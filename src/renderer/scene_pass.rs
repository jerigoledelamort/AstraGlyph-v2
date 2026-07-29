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
use crate::scene::component::MeshVertex;
use crate::scene::{Camera, Entity, MeshComponent, MaterialUniform};
use wgpu::{Device, Queue, Texture, TextureFormat, TextureView};

/// Maximum number of materials supported in the storage buffer.
const MAX_MATERIALS: usize = 256;

/// Maximum number of simultaneous light sources.
pub const MAX_LIGHTS: usize = 8;

/// Resolution of the (single, simplified) shadow map.
const SHADOW_MAP_SIZE: u32 = 1024;

/// Vertex attribute layout for `MeshVertex` (position, normal, color).
/// A `const` array gives it a `'static` lifetime so it can back more than
/// one `wgpu::VertexBufferLayout` (opaque + transparent pipelines) without
/// borrow-lifetime issues.
const VERTEX_ATTRS: [wgpu::VertexAttribute; 3] = [
    pipeline::vertex_attr(0, wgpu::VertexFormat::Float32x3, 0),
    pipeline::vertex_attr(1, wgpu::VertexFormat::Float32x3, std::mem::size_of::<f32>() as u64 * 3),
    pipeline::vertex_attr(2, wgpu::VertexFormat::Float32x3, std::mem::size_of::<f32>() as u64 * 6),
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
    vp_buffer: wgpu::Buffer,
    lights_meta_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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
    /// Simplified shadow map: depth-only pass from light[0]'s point of view.
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    shadow_texture: Texture,
    shadow_view: TextureView,
    shadow_vp_buffer: wgpu::Buffer,
}

impl ScenePipeline {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Self> {
        // Load shaders from embedded WGSL.
        let vs_source = include_str!("../graphics/shaders/scene_vertex.wgsl");
        let fs_source = include_str!("../graphics/shaders/scene_fragment.wgsl");

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
        // 3 = material storage, 4 = shadow map, 5 = shadow sampler, 6 = shadow view-proj.
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
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bind_group"),
            layout: &bind_group_layout,
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
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: shadow_vp_buffer.as_entire_binding(),
                },
            ],
        });

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

        // Shared descriptor builder for the opaque and transparent pipeline variants:
        // same shaders/layout/vertex format, different blend + depth-write state.
        let make_pipeline = |label: &str, blend: wgpu::BlendState, depth_write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(vertex_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_module,
                    entry_point: Some("main"),
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
        let opaque_pipeline = make_pipeline("scene_pipeline_opaque", wgpu::BlendState::REPLACE, true);
        // Transparent: tested against opaque depth but doesn't write depth itself —
        // relies on back-to-front draw order (see render_batched) instead.
        let transparent_pipeline =
            make_pipeline("scene_pipeline_transparent", wgpu::BlendState::ALPHA_BLENDING, false);

        // Shadow pass: depth-only, rendered from light[0]'s point of view.
        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_bind_group"),
            layout: &shadow_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_vp_buffer.as_entire_binding(),
            }],
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
            vp_buffer,
            lights_meta_buffer,
            lights_buffer,
            material_buffer,
            bind_group,
            target_texture,
            target_view,
            depth_texture,
            depth_view,
            fmt: format,
            width,
            height,
            buffer_cache: HashMap::new(),
            camera_pos: Vec3::ZERO,
            shadow_pipeline,
            shadow_bind_group,
            shadow_texture,
            shadow_view,
            shadow_vp_buffer,
        })
    }

    /// Upload the shadow-casting light's view-projection matrix (see
    /// `LightUniform::shadow_view_proj`).
    pub fn update_shadow_camera(&self, queue: &Queue, light_view_proj: Mat4) {
        buffer::write_buffer(queue, &self.shadow_vp_buffer, 0, &[ShadowUniform {
            view_proj: light_view_proj.m,
        }]);
    }

    /// Render the shadow map: scene depth from light[0]'s point of view.
    fn render_shadow_pass(
        &mut self,
        device: &Device,
        queue: &Queue,
        meshes: &[(Entity, &MeshComponent, u32)],
    ) {
        if meshes.is_empty() {
            return;
        }

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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.shadow_pipeline);
            rpass.set_bind_group(0, &self.shadow_bind_group, &[]);

            for (entity, _mesh, _material_index) in meshes {
                if let Some(bufs) = self.buffer_cache.get(&entity.id()) {
                    rpass.set_vertex_buffer(0, bufs.vertex_buffer.slice(..));
                    rpass.set_index_buffer(bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..bufs.index_count, 0, 0..1);
                }
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
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
    /// Each mesh is drawn with its material_index as the instance start,
    /// so the shader can look up the correct material via `instance_index`.
    /// Buffers are cached per-entity: created on first use, reused thereafter.
    /// `materials` must be the same slice passed to `upload_materials` this
    /// frame — used to classify each mesh as opaque or transparent.
    pub fn render_batched(
        &mut self,
        device: &Device,
        queue: &Queue,
        meshes: &[(Entity, &MeshComponent, u32)],
        materials: &[MaterialUniform],
    ) -> Result<()> {
        if meshes.is_empty() {
            return Ok(());
        }

        // Ensure all meshes have cached buffers.
        for (entity, mesh, _) in meshes {
            self.get_or_create_buffers(device, *entity, mesh);
        }

        // Simplified shadow map: depth-only pass from light[0]'s point of view.
        // Must run before the main pass since the fragment shader samples it.
        self.render_shadow_pass(device, queue, meshes);

        // Classify by material transparency; transparent meshes carry their
        // squared distance to the camera for back-to-front sorting.
        let mut opaque: Vec<(Entity, u32)> = Vec::new();
        let mut transparent: Vec<(Entity, u32, f32)> = Vec::new();
        for (entity, _mesh, material_index) in meshes {
            let is_transparent = materials
                .get(*material_index as usize)
                .map(|m| m.transparency > 0.0)
                .unwrap_or(false);
            if is_transparent {
                let center = self
                    .buffer_cache
                    .get(&entity.id())
                    .map(|b| b.center)
                    .unwrap_or(Vec3::ZERO);
                let dist_sq = (center - self.camera_pos).length_squared();
                transparent.push((*entity, *material_index, dist_sq));
            } else {
                opaque.push((*entity, *material_index));
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_bind_group(0, &self.bind_group, &[]);

            rpass.set_pipeline(&self.opaque_pipeline);
            for (entity, material_index) in &opaque {
                if let Some(bufs) = self.buffer_cache.get(&entity.id()) {
                    rpass.set_vertex_buffer(0, bufs.vertex_buffer.slice(..));
                    rpass.set_index_buffer(bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..bufs.index_count, 0, *material_index..*material_index + 1);
                }
            }

            rpass.set_pipeline(&self.transparent_pipeline);
            for (entity, material_index, _) in &transparent {
                if let Some(bufs) = self.buffer_cache.get(&entity.id()) {
                    rpass.set_vertex_buffer(0, bufs.vertex_buffer.slice(..));
                    rpass.set_index_buffer(bufs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..bufs.index_count, 0, *material_index..*material_index + 1);
                }
            }
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
}
