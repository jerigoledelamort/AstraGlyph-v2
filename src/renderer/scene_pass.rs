// Scene pass: renders 3D geometry to a low-resolution offscreen texture.
// Each pixel in the output corresponds to one ASCII cell.
//
// Features:
// - Depth buffer for correct occlusion
// - Batched rendering: all meshes in a single render pass + single submit
// - Buffer caching: vertex/index buffers created once per entity, reused
// - Multiple light sources (directional + point), summed per fragment
// - Transparent materials (Glass) sorted back-to-front and alpha-blended
//   after the opaque pass

use std::collections::HashMap;
use crate::engine::core::{cast_slice, Result};
use crate::engine::math::{Mat4, Vec3};
use crate::graphics::buffer;
use crate::graphics::pipeline;
use crate::graphics::texture;
use crate::graphics::timing::{GpuPass, GpuTimer};
use crate::scene::component::MeshVertex;
use crate::scene::{Camera, Entity, MeshComponent, MaterialUniform};
use wgpu::{Device, Queue, Texture, TextureFormat, TextureView};

/// Maximum number of materials supported in the storage buffer.
const MAX_MATERIALS: usize = 256;

/// Maximum number of drawable objects (mesh instances) per frame.
const MAX_OBJECTS: usize = 1024;

/// Maximum number of simultaneous light sources.
pub const MAX_LIGHTS: usize = 8;

/// Resolution of the (single, simplified) shadow map.
const SHADOW_MAP_SIZE: u32 = 1024;

/// Vertex attribute layout for `MeshVertex` (position, normal, color, uv).
/// A `const` array gives it a `'static` lifetime so it can back more than
/// one `wgpu::VertexBufferLayout` (opaque + transparent pipelines) without
/// borrow-lifetime issues.
const VERTEX_ATTRS: [wgpu::VertexAttribute; 4] = [
    pipeline::vertex_attr(0, wgpu::VertexFormat::Float32x3, 0),
    pipeline::vertex_attr(1, wgpu::VertexFormat::Float32x3, std::mem::size_of::<f32>() as u64 * 3),
    pipeline::vertex_attr(2, wgpu::VertexFormat::Float32x3, std::mem::size_of::<f32>() as u64 * 6),
    pipeline::vertex_attr(3, wgpu::VertexFormat::Float32x2, std::mem::size_of::<f32>() as u64 * 9),
];

/// A single light source: directional or point, selected by `position.w`
/// (0.0 = directional, 1.0 = point). Layout matches WGSL std140: each vec3
/// is padded to 16 bytes (vec4). Total: 16 + 16 + 16 + 4 + 4 + 8(padding) = 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LightUniform {
    /// Light position in world space (point lights). `w` = light type
    /// (0.0 = directional, 1.0 = point).
    pub position: [f32; 4],
    /// Direction the light travels (directional lights). Unused for point lights.
    pub direction: [f32; 4],
    /// Light color (RGB, w = padding).
    pub color: [f32; 4],
    /// Per-light ambient contribution, summed into the material's base ambient.
    pub ambient: f32,
    /// Per-light intensity multiplier for diffuse + specular.
    pub diffuse: f32,
    /// Padding to reach 64 bytes.
    pub _padding: [f32; 2],
}

unsafe impl crate::engine::core::Pod for LightUniform {}

impl LightUniform {
    /// A directional light (e.g. sun/sky): `direction` points FROM the light
    /// source toward the scene.
    pub fn directional(direction: Vec3, color: Vec3, ambient: f32, intensity: f32) -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 0.0],
            direction: [direction.x, direction.y, direction.z, 0.0],
            color: [color.x, color.y, color.z, 0.0],
            ambient,
            diffuse: intensity,
            _padding: [0.0, 0.0],
        }
    }

    /// A point light at `position`, attenuated by distance.
    pub fn point(position: Vec3, color: Vec3, ambient: f32, intensity: f32) -> Self {
        Self {
            position: [position.x, position.y, position.z, 1.0],
            direction: [0.0, 0.0, 0.0, 0.0],
            color: [color.x, color.y, color.z, 0.0],
            ambient,
            diffuse: intensity,
            _padding: [0.0, 0.0],
        }
    }

    /// Build a simplified shadow-map view-projection matrix for this light:
    /// an orthographic frustum centered on the scene's bounding sphere,
    /// looking from the light toward `scene_center`. Works for both light
    /// types — for point lights the frustum is centered on the light's
    /// actual position; for directional lights it's pulled back along the
    /// light's direction far enough to enclose the scene.
    pub fn shadow_view_proj(&self, scene_center: Vec3, scene_radius: f32) -> Mat4 {
        let radius = scene_radius.max(0.5);
        let is_point = self.position[3] > 0.5;
        let eye = if is_point {
            Vec3::new(self.position[0], self.position[1], self.position[2])
        } else {
            let dir = Vec3::new(self.direction[0], self.direction[1], self.direction[2]).normalize();
            scene_center - dir * radius * 2.0
        };
        let forward = (scene_center - eye).normalize();
        let up = if forward.cross(Vec3::UNIT_Y).length_squared() < 1e-4 {
            Vec3::UNIT_X
        } else {
            Vec3::UNIT_Y
        };
        let view = Mat4::look_at(eye, scene_center, up);
        let proj = Mat4::orthographic(-radius, radius, -radius, radius, 0.05, radius * 4.0);
        proj.mul(view)
    }
}

/// Header uniform accompanying the `lights` storage buffer: how many of the
/// `MAX_LIGHTS` slots are actually populated.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct LightsMetaUniform {
    count: u32,
    _padding: [u32; 3],
}

unsafe impl crate::engine::core::Pod for LightsMetaUniform {}

/// Uniform block for the camera: view-projection matrix + camera position.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CameraUniform {
    pub view_proj: [f32; 16],
    pub camera_pos: [f32; 3],
    pub _padding: f32,
}

unsafe impl crate::engine::core::Pod for CameraUniform {}

/// Per-object (per mesh instance) GPU data: its model matrix, the matching
/// normal matrix, plus which material it uses. Indexed in the shaders via
/// `@builtin(instance_index)`.
///
/// Model matrix and material index live together — and are looked up by *object*
/// index rather than material index — because materials are deduplicated
/// (see `MaterialRegistry`): several objects legitimately share one material
/// slot, so a material index can no longer identify an object.
///
/// The normal matrix is the inverse-transpose of the model's upper 3x3, stored
/// as a full mat4x4 rather than a mat3x3: a WGSL mat3x3 is three vec4-padded
/// columns (48 bytes) whose layout is easy to get subtly wrong against a Rust
/// `[f32; 12]`, while a mat4x4 matches `[f32; 16]` exactly. Sixteen wasted
/// bytes per object buy an unambiguous layout.
///
/// 32 floats + u32 + 3 u32 padding = 144 bytes (mat4x4 alignment is 16).
/// This struct is read by BOTH `scene_vertex.wgsl` and `shadow_vertex.wgsl`;
/// their `Object` structs must be kept identical or one of them walks the
/// storage buffer at the wrong stride.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ObjectUniform {
    pub model: [f32; 16],
    /// Inverse-transpose of `model`'s upper 3x3, for transforming normals.
    /// Equal to the rotation part when the scale is uniform; diverges exactly
    /// when non-uniform scale would otherwise skew the normals.
    pub normal: [f32; 16],
    pub material_index: u32,
    pub _padding: [u32; 3],
}

unsafe impl crate::engine::core::Pod for ObjectUniform {}

impl ObjectUniform {
    /// Build an object entry from a world matrix and a material slot.
    ///
    /// The normal matrix is computed here, once per object per frame, rather
    /// than per vertex in the shader — WGSL has no matrix inverse, and even if
    /// it did, inverting in the vertex stage would repeat the work thousands of
    /// times. A singular model matrix (zero scale on some axis) has no inverse;
    /// the model matrix itself is used then, which renders *something* (the old
    /// behaviour) instead of NaN-ing every normal.
    pub fn new(model: Mat4, material_index: u32) -> Self {
        let normal = model
            .inverse_affine()
            .map(|inv| inv.transpose())
            .unwrap_or(model);
        // Zero the translation row the transpose moved into the fourth column:
        // normals are directions, and the shader multiplies with w = 0 anyway,
        // but keeping the matrix clean makes it comparable in tests and dumps.
        let mut normal_m = normal.m;
        normal_m[3] = 0.0;
        normal_m[7] = 0.0;
        normal_m[11] = 0.0;
        normal_m[12] = 0.0;
        normal_m[13] = 0.0;
        normal_m[14] = 0.0;
        normal_m[15] = 1.0;
        Self {
            model: model.m,
            normal: normal_m,
            material_index,
            _padding: [0, 0, 0],
        }
    }
}

/// Uniform block for the shadow map: the shadow-casting light's view-projection
/// matrix. Bound as VERTEX in the shadow pass and as FRAGMENT in the main
/// scene pass (same buffer, two bind groups).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ShadowUniform {
    view_proj: [f32; 16],
}

unsafe impl crate::engine::core::Pod for ShadowUniform {}

/// Cached GPU buffers for a single mesh, plus its precomputed centroid
/// (used to sort transparent meshes back-to-front without re-walking all
/// vertices every frame).
struct MeshBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    center: Vec3,
}

/// Scene render pipeline: renders meshes to an offscreen texture with depth testing.
/// Supports batched rendering (all meshes in one render pass), buffer caching,
/// multiple lights, and a separate blended pass for transparent materials.
pub struct ScenePipeline {
    opaque_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    glass_tint_pipeline: wgpu::RenderPipeline,
    /// Ray-traced counterpart of `opaque_pipeline`, present only when the device
    /// has ray query. It handles *every* material, transparent included: a
    /// refraction ray already reports what is behind the glass, so there is
    /// nothing left to alpha-blend against (see `scene_traced_fragment.wgsl`).
    traced_pipeline: Option<wgpu::RenderPipeline>,
    vp_buffer: wgpu::Buffer,
    lights_meta_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    object_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Kept so the bind group can be rebuilt when the scene's texture set
    /// changes (`set_texture_array`).
    bind_group_layout: wgpu::BindGroupLayout,
    /// The scene's texture array. Starts as a 1x1 white placeholder; owned
    /// here because binding 8/9 reference its view and sampler.
    texture_array: crate::graphics::texture_array::TextureArray,
    /// Offscreen render target texture (low-resolution).
    pub target_texture: Texture,
    pub target_view: TextureView,
    /// Depth texture for depth testing.
    depth_texture: Texture,
    depth_view: TextureView,
    #[allow(dead_code)]
    fmt: TextureFormat,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    /// Cache of vertex/index buffers keyed by entity ID.
    buffer_cache: HashMap<u64, MeshBuffers>,
    /// Camera position from the last `update_camera` call, used to sort
    /// transparent meshes back-to-front.
    camera_pos: Vec3,
    /// Draw calls issued by the most recent `render_batched`.
    ///
    /// Counted rather than derived from the mesh count, because they are not the
    /// same number: the rasterised path draws transparent meshes twice (tint, then
    /// surface) and the traced path draws everything once. A profiler reporting the
    /// mesh count as the draw count would hide exactly that difference.
    draw_calls: u32,
    /// Simplified shadow map: depth-only pass from light[0]'s point of view.
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    shadow_texture: Texture,
    shadow_view: TextureView,
    /// Held so `set_texture_array` can rebuild the bind group that references it.
    shadow_sampler: wgpu::Sampler,
    shadow_vp_buffer: wgpu::Buffer,
}

impl ScenePipeline {
    /// Build the scene pipelines.
    ///
    /// `traced_layout` is the ray-tracing bind group layout (group 1) from
    /// `RayTracer::layout`. Passing `None` — which is what a device without ray
    /// query must do — skips the traced pipeline entirely, so no WGSL containing
    /// ray queries is ever compiled on hardware that cannot run it.
    pub fn new(
        device: &Device,
        queue: &Queue,
        width: u32,
        height: u32,
        format: TextureFormat,
        traced_layout: Option<&wgpu::BindGroupLayout>,
    ) -> Result<Self> {
        // Load shaders from embedded WGSL. WGSL has no include directive, so the
        // shared shading code is prepended to each entry-point file here; both
        // fragment paths must agree on the surface model exactly.
        let vs_source = include_str!("../graphics/shaders/scene_vertex.wgsl");
        let fs_source = concat!(
            include_str!("../graphics/shaders/scene_shading.wgsl"),
            include_str!("../graphics/shaders/scene_fragment.wgsl"),
        );
        // `enable` directives must precede every other item in a WGSL module, so
        // the extension line has to be prepended here rather than living at the
        // top of either source file: `scene_shading.wgsl` is shared with the
        // rasterised path, which must stay compilable on a device that has no
        // ray query to enable.
        let traced_fs_source = concat!(
            "enable wgpu_ray_query;\n",
            include_str!("../graphics/shaders/scene_shading.wgsl"),
            include_str!("../graphics/shaders/scene_traced_fragment.wgsl"),
        );

        let vs_module = pipeline::compile_shader(device, "scene_vertex", vs_source);
        let fs_module = pipeline::compile_shader(device, "scene_fragment", fs_source);

        // Uniform buffers.
        let vp_buffer = buffer::uniform_buffer(device, "scene_camera", 80); // CameraUniform = 16+3+1 floats * 4 = 80 bytes
        let lights_meta_buffer = buffer::uniform_buffer(device, "scene_lights_meta", 16); // count + padding

        // Lights storage buffer (read-only, fixed capacity of MAX_LIGHTS).
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_lights"),
            size: (MAX_LIGHTS * std::mem::size_of::<LightUniform>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Material storage buffer (read-only, accessed by instance_index in shader).
        let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_materials"),
            size: (MAX_MATERIALS * std::mem::size_of::<MaterialUniform>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Per-object storage buffer (model matrix + material index), indexed by
        // instance_index in both the scene and shadow vertex shaders.
        let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_objects"),
            size: (MAX_OBJECTS * std::mem::size_of::<ObjectUniform>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Simplified shadow map resources (light[0]'s depth-only view).
        let shadow_vp_buffer = buffer::uniform_buffer(device, "scene_shadow_vp", 64);
        let shadow_texture = texture::sampled_depth_texture(device, "scene_shadow_map", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene_shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Bind group layout: 0 = vp buffer, 1 = lights meta, 2 = lights storage,
        // 3 = material storage, 4 = shadow map, 5 = shadow sampler, 6 = shadow view-proj,
        // 7 = objects, 8 = texture array, 9 = texture sampler.
        let texture_entries = crate::graphics::texture_array::TextureArray::layout_entries(8, 9);
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    texture_entries[0],
                    texture_entries[1],
                ],
            });

        // The scene starts with an empty (1x1 white placeholder) texture array;
        // `set_texture_array` swaps in the real one once the scene's textures
        // are decoded, rebuilding the bind group.
        let texture_array = crate::graphics::texture_array::TextureArray::new(device, queue, &[])?;

        let bind_group = Self::create_scene_bind_group(
            device,
            &bind_group_layout,
            &vp_buffer,
            &lights_meta_buffer,
            &lights_buffer,
            &material_buffer,
            &shadow_view,
            &shadow_sampler,
            &shadow_vp_buffer,
            &object_buffer,
            &texture_array,
        );

        // Pipeline layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex buffer layout for MeshVertex. Backed by a `'static` const array
        // so it can be reused for both the opaque and transparent pipelines.
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRS,
        };

        // Offscreen render target.
        let target_texture = texture::render_target_texture(
            device,
            "scene_render_target",
            width,
            height,
            format,
        );
        let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth texture.
        let depth_texture = texture::depth_texture(device, "scene_depth", width, height);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Shared descriptor builder for the pipeline variants: same vertex shader
        // and vertex format, differing in fragment module, pipeline layout, blend
        // state, depth-write and fragment entry point.
        let make_pipeline = |label: &str,
                             layout: &wgpu::PipelineLayout,
                             fragment: &wgpu::ShaderModule,
                             blend: wgpu::BlendState,
                             depth_write: bool,
                             entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(vertex_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: fragment,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // Opaque: writes depth, no blending (straight overwrite).
        let opaque_pipeline = make_pipeline(
            "scene_pipeline_opaque",
            &pipeline_layout,
            &fs_module,
            wgpu::BlendState::REPLACE,
            true,
            "main",
        );
        // Transparent: tested against opaque depth but doesn't write depth itself —
        // relies on back-to-front draw order (see render_batched) instead.
        let transparent_pipeline = make_pipeline(
            "scene_pipeline_transparent",
            &pipeline_layout,
            &fs_module,
            wgpu::BlendState::ALPHA_BLENDING,
            false,
            "main",
        );
        // Glass tint: multiply blend (result = destination * source), drawn before
        // the transparent surface pass. Alpha blending alone can only fade the
        // background toward the glass colour; multiplying is what actually filters
        // it, so objects seen through coloured glass take on its hue.
        let glass_tint_pipeline = make_pipeline(
            "scene_pipeline_glass_tint",
            &pipeline_layout,
            &fs_module,
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Dst,
                    dst_factor: wgpu::BlendFactor::Zero,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            false,
            "tint",
        );

        // Traced: one pipeline for every material. Compiled only when the caller
        // supplied a ray-tracing layout, because the WGSL contains ray queries
        // that a device without the feature cannot even parse.
        let traced_pipeline = traced_layout.map(|traced_layout| {
            let traced_fs_module =
                pipeline::compile_shader(device, "scene_traced_fragment", traced_fs_source);
            let traced_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("scene_traced_pipeline_layout"),
                    bind_group_layouts: &[Some(&bind_group_layout), Some(traced_layout)],
                    immediate_size: 0,
                });
            make_pipeline(
                "scene_pipeline_traced",
                &traced_pipeline_layout,
                &traced_fs_module,
                wgpu::BlendState::REPLACE,
                true,
                "main",
            )
        });

        // Shadow pass: depth-only, rendered from light[0]'s point of view.
        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // The shadow pass needs the same model matrices as the scene
                    // pass, otherwise shadows would be cast by untransformed geometry.
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_bind_group"),
            layout: &shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_vp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: object_buffer.as_entire_binding(),
                },
            ],
        });
        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pipeline_layout"),
            bind_group_layouts: &[Some(&shadow_bind_group_layout)],
            immediate_size: 0,
        });
        let shadow_vs_source = include_str!("../graphics/shaders/shadow_vertex.wgsl");
        let shadow_vs_module = pipeline::compile_shader(device, "shadow_vertex", shadow_vs_source);
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_vs_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout.clone())],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            opaque_pipeline,
            transparent_pipeline,
            glass_tint_pipeline,
            traced_pipeline,
            vp_buffer,
            lights_meta_buffer,
            lights_buffer,
            material_buffer,
            object_buffer,
            bind_group,
            bind_group_layout,
            texture_array,
            target_texture,
            target_view,
            depth_texture,
            depth_view,
            fmt: format,
            width,
            height,
            buffer_cache: HashMap::new(),
            camera_pos: Vec3::ZERO,
            draw_calls: 0,
            shadow_pipeline,
            shadow_bind_group,
            shadow_texture,
            shadow_view,
            shadow_sampler,
            shadow_vp_buffer,
        })
    }

    /// Build the group-0 bind group. Split out of `new` because the texture
    /// array can be replaced at runtime (scene reload), which requires
    /// rebuilding the whole group — bind groups are immutable.
    #[allow(clippy::too_many_arguments)]
    fn create_scene_bind_group(
        device: &Device,
        layout: &wgpu::BindGroupLayout,
        vp_buffer: &wgpu::Buffer,
        lights_meta_buffer: &wgpu::Buffer,
        lights_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        shadow_view: &TextureView,
        shadow_sampler: &wgpu::Sampler,
        shadow_vp_buffer: &wgpu::Buffer,
        object_buffer: &wgpu::Buffer,
        texture_array: &crate::graphics::texture_array::TextureArray,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: shadow_vp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(texture_array.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(texture_array.sampler()),
                },
            ],
        })
    }

    /// Replace the scene's texture array (scene load/reload) and rebuild the
    /// bind group that references it.
    pub fn set_texture_array(
        &mut self,
        device: &Device,
        texture_array: crate::graphics::texture_array::TextureArray,
    ) {
        self.texture_array = texture_array;
        self.bind_group = Self::create_scene_bind_group(
            device,
            &self.bind_group_layout,
            &self.vp_buffer,
            &self.lights_meta_buffer,
            &self.lights_buffer,
            &self.material_buffer,
            &self.shadow_view,
            // The shadow sampler lives only inside the bind group today; keep a
            // handle so the rebuild can reference it.
            &self.shadow_sampler,
            &self.shadow_vp_buffer,
            &self.object_buffer,
            &self.texture_array,
        );
    }

    /// The current texture array (for the profiler and for resolving
    /// per-material UV scales).
    pub fn texture_array(&self) -> &crate::graphics::texture_array::TextureArray {
        &self.texture_array
    }

    /// Upload the shadow-casting light's view-projection matrix (see
    /// `LightUniform::shadow_view_proj`).
    pub fn update_shadow_camera(&self, queue: &Queue, light_view_proj: Mat4) {
        buffer::write_buffer(queue, &self.shadow_vp_buffer, 0, &[ShadowUniform {
            view_proj: light_view_proj.m,
        }]);
    }

    /// Render the shadow map: scene depth from light[0]'s point of view.
    /// Takes the already-resolved `(entity, object_index)` draw list so casters
    /// are transformed by exactly the same model matrices as the scene pass.
    fn render_shadow_pass(
        &mut self,
        device: &Device,
        queue: &Queue,
        drawable: &[(Entity, u32)],
        timer: &mut GpuTimer,
    ) {
        if drawable.is_empty() {
            return;
        }

        let mut shadow_draws;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shadow_pass_encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timer.pass_writes(GpuPass::Shadow),
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.shadow_pipeline);
            rpass.set_bind_group(0, &self.shadow_bind_group, &[]);
            shadow_draws = 0;

            for (entity, object_index) in drawable {
                if let Some(bufs) = self.buffer_cache.get(&entity.id()) {
                    rpass.set_vertex_buffer(0, bufs.vertex_buffer.slice(..));
                    rpass.set_index_buffer(bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..bufs.index_count, 0, *object_index..*object_index + 1);
                    shadow_draws += 1;
                }
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
        self.draw_calls += shadow_draws;
    }

    /// Update the camera uniform (view-projection matrix + position).
    pub fn update_camera(&mut self, queue: &Queue, camera: &Camera) {
        let vp = camera.view_projection();
        let pos = camera.position;
        self.camera_pos = pos;
        buffer::write_buffer(queue, &self.vp_buffer, 0, &[CameraUniform {
            view_proj: vp.m,
            camera_pos: [pos.x, pos.y, pos.z],
            _padding: 0.0,
        }]);
    }

    /// Upload up to `MAX_LIGHTS` lights to the GPU (storage buffer + count header).
    pub fn update_lights(&self, queue: &Queue, lights: &[LightUniform]) {
        let count = lights.len().min(MAX_LIGHTS);
        let meta = LightsMetaUniform {
            count: count as u32,
            _padding: [0, 0, 0],
        };
        buffer::write_buffer(queue, &self.lights_meta_buffer, 0, &[meta]);
        if count > 0 {
            queue.write_buffer(&self.lights_buffer, 0, cast_slice(&lights[..count]));
        }
    }

    /// Upload all materials to the GPU storage buffer.
    pub fn upload_materials(&self, queue: &Queue, materials: &[MaterialUniform]) {
        let count = materials.len().min(MAX_MATERIALS);
        if count > 0 {
            queue.write_buffer(
                &self.material_buffer,
                0,
                cast_slice(&materials[..count]),
            );
        }
    }

    /// Upload per-object model matrices + material indices to the GPU.
    /// Entries beyond `MAX_OBJECTS` are dropped (the corresponding meshes will
    /// read a stale slot rather than crash — see `render_batched`, which clamps
    /// the draw list to the same limit).
    pub fn upload_objects(&self, queue: &Queue, objects: &[ObjectUniform]) {
        let count = objects.len().min(MAX_OBJECTS);
        if count > 0 {
            queue.write_buffer(&self.object_buffer, 0, cast_slice(&objects[..count]));
        }
    }

    /// How many objects a single frame can draw (size of the object storage buffer).
    pub const fn max_objects() -> usize {
        MAX_OBJECTS
    }

    /// Whether a traced pipeline was built, i.e. whether `render_batched` can
    /// honour a `Some(traced)` argument.
    pub fn supports_tracing(&self) -> bool {
        self.traced_pipeline.is_some()
    }

    /// Draw calls issued by the most recent frame, across every pass.
    pub fn draw_calls(&self) -> u32 {
        self.draw_calls
    }

    /// Bytes of GPU memory held by cached mesh buffers.
    ///
    /// An estimate of the vertex and index data only — not the render targets,
    /// depth buffer, shadow map or acceleration structures, whose sizes wgpu does
    /// not expose. Reported as "mesh bytes" rather than "GPU memory" so the number
    /// is not mistaken for the total.
    pub fn mesh_bytes(&self) -> u64 {
        self.buffer_cache
            .values()
            .map(|b| b.vertex_buffer.size() + b.index_buffer.size())
            .sum()
    }

    /// Meshes with cached GPU buffers.
    pub fn cached_meshes(&self) -> usize {
        self.buffer_cache.len()
    }

    /// The scene depth buffer. Exposed for CPU readback: screen-space effects
    /// (SSAO, depth-driven cell subdivision) consume it on the ASCII side.
    pub fn depth_texture(&self) -> &Texture {
        &self.depth_texture
    }

    /// Get or create cached GPU buffers (and centroid) for a mesh.
    fn get_or_create_buffers(
        &mut self,
        device: &Device,
        entity: Entity,
        mesh: &MeshComponent,
    ) -> &MeshBuffers {
        self.buffer_cache.entry(entity.id()).or_insert_with(|| {
            let vbuf = buffer::vertex_buffer(device, "mesh_vertices", &mesh.vertices);
            let ibuf = buffer::index_buffer(device, "mesh_indices", &mesh.indices);
            let center = if mesh.vertices.is_empty() {
                Vec3::ZERO
            } else {
                let sum = mesh
                    .vertices
                    .iter()
                    .fold(Vec3::ZERO, |acc, v| acc + v.position);
                sum / mesh.vertices.len() as f32
            };
            MeshBuffers {
                vertex_buffer: vbuf,
                index_buffer: ibuf,
                index_count: mesh.indices.len() as u32,
                center,
            }
        })
    }

    /// Render all meshes in a single render pass (batched): opaque meshes first
    /// (depth-tested, depth-written), then transparent meshes sorted
    /// back-to-front and alpha-blended on top.
    ///
    /// Each mesh is drawn with its *object* index as the instance start, so the
    /// shaders can look up its model matrix and material via `instance_index`.
    /// Buffers are cached per-entity: created on first use, reused thereafter.
    /// `objects` and `materials` must be the same slices passed to
    /// `upload_objects` / `upload_materials` this frame — they are used to
    /// resolve each mesh's world transform and opaque/transparent class.
    ///
    /// When `traced` is `Some`, the ray-traced pipeline replaces all three
    /// rasterised ones: every material goes through a single opaque draw, the
    /// shadow map pass is skipped (shadow rays replace it), and the glass tint
    /// pass is skipped (the refraction ray already carries what is behind the
    /// glass, filtered by its colour). Passing `Some` without having built the
    /// traced pipeline falls back to rasterising rather than failing.
    pub fn render_batched(
        &mut self,
        device: &Device,
        queue: &Queue,
        meshes: &[(Entity, &MeshComponent, u32)],
        objects: &[ObjectUniform],
        materials: &[MaterialUniform],
        traced: Option<&wgpu::BindGroup>,
        timer: &mut GpuTimer,
    ) -> Result<()> {
        if meshes.is_empty() {
            return Ok(());
        }
        let traced = traced.filter(|_| self.traced_pipeline.is_some());

        // Ensure all meshes have cached buffers.
        for (entity, mesh, _) in meshes {
            self.get_or_create_buffers(device, *entity, mesh);
        }

        // Drop anything past the object buffer's capacity rather than letting it
        // read a slot that was never uploaded.
        let drawable: Vec<(Entity, u32)> = meshes
            .iter()
            .filter(|(_, _, object_index)| (*object_index as usize) < MAX_OBJECTS.min(objects.len()))
            .map(|(entity, _, object_index)| (*entity, *object_index))
            .collect();
        if drawable.is_empty() {
            return Ok(());
        }

        self.draw_calls = 0;

        // Simplified shadow map: depth-only pass from light[0]'s point of view.
        // Must run before the main pass since the fragment shader samples it.
        // Skipped entirely when tracing: the traced shader never samples the map,
        // so rendering it would be pure cost.
        if traced.is_none() {
            self.render_shadow_pass(device, queue, &drawable, timer);
        }

        // Classify by material transparency; transparent meshes carry their
        // world-space distance to the camera for back-to-front sorting.
        let mut opaque: Vec<(Entity, u32)> = Vec::new();
        let mut transparent: Vec<(Entity, u32, f32)> = Vec::new();
        for (entity, object_index) in &drawable {
            let object = &objects[*object_index as usize];
            let is_transparent = materials
                .get(object.material_index as usize)
                .map(|m| m.transparency > 0.0)
                .unwrap_or(false);
            if is_transparent {
                // The cached centroid is in mesh-local space, so it has to go
                // through the model matrix before it can be sorted by depth.
                let local_center = self
                    .buffer_cache
                    .get(&entity.id())
                    .map(|b| b.center)
                    .unwrap_or(Vec3::ZERO);
                let world_center = Mat4::from_cols_array(object.model).transform_point(local_center);
                let dist_sq = (world_center - self.camera_pos).length_squared();
                transparent.push((*entity, *object_index, dist_sq));
            } else {
                opaque.push((*entity, *object_index));
            }
        }
        // Farthest first (back-to-front).
        transparent.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scene_pass_encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timer.pass_writes(GpuPass::Scene),
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_bind_group(0, &self.bind_group, &[]);

            let mut drawn = 0u32;
            if let (Some(traced_group), Some(traced_pipeline)) =
                (traced, self.traced_pipeline.as_ref())
            {
                // One pass, one pipeline, every material. Draw order no longer
                // matters: nothing is blended, so depth testing alone resolves
                // occlusion — including glass in front of glass, which the
                // refraction ray sees through rather than blending against.
                rpass.set_bind_group(1, traced_group, &[]);
                rpass.set_pipeline(traced_pipeline);
                for (entity, object_index) in &drawable {
                    if let Some(bufs) = self.buffer_cache.get(&entity.id()) {
                        rpass.set_vertex_buffer(0, bufs.vertex_buffer.slice(..));
                        rpass
                            .set_index_buffer(bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..bufs.index_count, 0, *object_index..*object_index + 1);
                        drawn += 1;
                    }
                }
            } else {
                rpass.set_pipeline(&self.opaque_pipeline);
                for (entity, object_index) in &opaque {
                    if let Some(bufs) = self.buffer_cache.get(&entity.id()) {
                        rpass.set_vertex_buffer(0, bufs.vertex_buffer.slice(..));
                        rpass
                            .set_index_buffer(bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..bufs.index_count, 0, *object_index..*object_index + 1);
                        drawn += 1;
                    }
                }

                // Transparent objects, farthest first. Each one is drawn twice: the
                // tint pass filters whatever is already behind it, then the surface
                // pass adds its own shading, reflections and rim. Interleaving the two
                // per object (rather than doing all tints then all surfaces) keeps the
                // back-to-front ordering correct when glass overlaps glass.
                for (entity, object_index, _) in &transparent {
                    if let Some(bufs) = self.buffer_cache.get(&entity.id()) {
                        rpass.set_vertex_buffer(0, bufs.vertex_buffer.slice(..));
                        rpass
                            .set_index_buffer(bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                        rpass.set_pipeline(&self.glass_tint_pipeline);
                        rpass.draw_indexed(0..bufs.index_count, 0, *object_index..*object_index + 1);

                        rpass.set_pipeline(&self.transparent_pipeline);
                        rpass.draw_indexed(0..bufs.index_count, 0, *object_index..*object_index + 1);
                        drawn += 2;
                    }
                }
            }
            self.draw_calls += drawn;
        }

        queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_uniform_directional_encodes_type_zero() {
        let l = LightUniform::directional(Vec3::new(0.0, -1.0, 0.0), Vec3::ONE, 0.1, 0.9);
        assert_eq!(l.position[3], 0.0);
        assert_eq!(l.direction[0], 0.0);
        assert_eq!(l.direction[1], -1.0);
    }

    #[test]
    fn light_uniform_point_encodes_type_one() {
        let l = LightUniform::point(Vec3::new(1.0, 2.0, 3.0), Vec3::ONE, 0.1, 0.9);
        assert_eq!(l.position[3], 1.0);
        assert_eq!(l.position[0], 1.0);
        assert_eq!(l.position[1], 2.0);
        assert_eq!(l.position[2], 3.0);
    }

    #[test]
    fn shadow_view_proj_point_light_maps_scene_center_near_ndc_origin() {
        // A point light directly above the scene center should map that
        // center to roughly the middle of the shadow map (NDC x,y ≈ 0).
        let light = LightUniform::point(Vec3::new(0.0, 10.0, 0.0), Vec3::ONE, 0.1, 1.0);
        let center = Vec3::ZERO;
        let radius = 5.0;
        let vp = light.shadow_view_proj(center, radius);
        let clip = vp.transform_vec4(crate::engine::math::Vec4::from_vec3(center, 1.0));
        assert!(clip.w.abs() > 1e-6);
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        assert!(ndc_x.abs() < 1e-4, "ndc_x = {ndc_x}");
        assert!(ndc_y.abs() < 1e-4, "ndc_y = {ndc_y}");
    }

    #[test]
    fn shadow_view_proj_directional_light_looks_toward_scene_center() {
        let light = LightUniform::directional(Vec3::new(0.0, -1.0, 0.0), Vec3::ONE, 0.1, 1.0);
        let center = Vec3::new(1.0, 2.0, 3.0);
        let vp = light.shadow_view_proj(center, 4.0);
        let clip = vp.transform_vec4(crate::engine::math::Vec4::from_vec3(center, 1.0));
        assert!(clip.w.abs() > 1e-6);
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        assert!(ndc_x.abs() < 1e-4, "ndc_x = {ndc_x}");
        assert!(ndc_y.abs() < 1e-4, "ndc_y = {ndc_y}");
    }

    #[test]
    fn lights_meta_uniform_is_pod_sized_correctly() {
        assert_eq!(std::mem::size_of::<LightsMetaUniform>(), 16);
        assert_eq!(std::mem::size_of::<LightUniform>(), 64);
    }

    #[test]
    fn object_uniform_layout_matches_wgsl_expectations() {
        // 2 * mat4x4<f32> (128) + u32 (4) + 3 * u32 padding (12) = 144, a
        // multiple of the 16-byte alignment a mat4x4 member forces on the struct.
        assert_eq!(std::mem::size_of::<ObjectUniform>(), 144);
        assert_eq!(std::mem::size_of::<ObjectUniform>() % 16, 0);
    }

    #[test]
    fn object_uniform_preserves_model_matrix_and_material() {
        let model = Mat4::translation(1.0, 2.0, 3.0);
        let obj = ObjectUniform::new(model, 7);
        assert_eq!(obj.material_index, 7);
        // The stored matrix must round-trip so the renderer can re-derive world
        // positions on the CPU (used for transparent depth sorting).
        let restored = Mat4::from_cols_array(obj.model);
        assert_eq!(restored, model);
        assert_eq!(restored.transform_point(Vec3::ZERO), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn normal_matrix_equals_rotation_for_rigid_transforms() {
        // For rotation + translation (no scale) the inverse-transpose IS the
        // rotation, so the normal matrix must match the model's upper 3x3.
        let model = Mat4::translation(5.0, -2.0, 1.0).mul(Mat4::rotation_y(0.9));
        let obj = ObjectUniform::new(model, 0);
        let n = Mat4::from_cols_array(obj.normal);
        let dir = Vec3::new(0.3, 0.8, -0.5).normalize();
        let want = model.transform_dir(dir);
        let got = n.transform_dir(dir);
        assert!((want - got).length() < 1e-5, "want {want}, got {got}");
        // And translation must not leak into it.
        assert_eq!(n.transform_dir(Vec3::ZERO), Vec3::ZERO);
    }

    /// The plan's own acceptance test for 0.1: an ellipsoid from scale (2,1,1)
    /// must keep the normal at its "north pole" pointing straight up, and a
    /// side normal must be *corrected*, not just scaled. Using the model matrix
    /// (the old behaviour) fails the side-normal check.
    #[test]
    fn normal_matrix_corrects_normals_under_non_uniform_scale() {
        let model = Mat4::scaling(2.0, 1.0, 1.0);
        let obj = ObjectUniform::new(model, 0);
        let n = Mat4::from_cols_array(obj.normal);

        // Top of the sphere: normal (0,1,0) must stay (0,1,0).
        let top = n.transform_dir(Vec3::UNIT_Y).normalize();
        assert!((top - Vec3::UNIT_Y).length() < 1e-6, "top normal became {top}");

        // A 45-degree normal on the unit sphere. On the ellipsoid the surface
        // flattens along x, so the true normal leans *away* from x — the
        // inverse-transpose divides the x component by the scale.
        let slanted = Vec3::new(1.0, 1.0, 0.0).normalize();
        let corrected = n.transform_dir(slanted).normalize();
        let expected = Vec3::new(0.5, 1.0, 0.0).normalize();
        assert!(
            (corrected - expected).length() < 1e-5,
            "expected {expected}, got {corrected}"
        );
        // The model matrix would have produced the opposite lean — assert the
        // difference so this test cannot silently pass with the old code.
        let wrong = model.transform_dir(slanted).normalize();
        assert!((corrected - wrong).length() > 0.1, "normal matrix must differ from model matrix here");
    }

    #[test]
    fn normal_matrix_survives_a_singular_model_matrix() {
        // Zero scale has no inverse; the fallback keeps rendering (with the old
        // incorrect-under-scale normals) rather than poisoning the frame with NaN.
        let model = Mat4::scaling(1.0, 0.0, 1.0);
        let obj = ObjectUniform::new(model, 0);
        assert!(obj.normal.iter().all(|v| v.is_finite()));
    }
}
