// Scene pass: renders 3D geometry to a low-resolution offscreen texture.
// Each pixel in the output corresponds to one ASCII cell.

use crate::engine::core::Result;
use crate::graphics::buffer;
use crate::graphics::pipeline;
use crate::scene::component::MeshVertex;
use crate::scene::{Camera, MeshComponent};
use wgpu::{Device, Queue, Texture, TextureFormat, TextureView};

/// Light parameters passed to the scene fragment shader.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LightUniform {
    pub direction: [f32; 3],
    pub ambient: f32,
    pub diffuse: f32,
}

unsafe impl crate::engine::core::Pod for LightUniform {}

/// Uniform block for the view-projection matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ViewProjUniform {
    pub matrix: [f32; 16],
}

unsafe impl crate::engine::core::Pod for ViewProjUniform {}

/// Scene render pipeline: renders meshes to an offscreen texture.
pub struct ScenePipeline {
    pipeline: wgpu::RenderPipeline,
    vp_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Offscreen render target texture (low-resolution).
    pub target_texture: Texture,
    pub target_view: TextureView,
    fmt: TextureFormat,
    width: u32,
    height: u32,
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
        let vp_buffer = buffer::uniform_buffer(device, "scene_vp", 64); // mat4 = 16 floats * 4 bytes
        let light_buffer = buffer::uniform_buffer(device, "scene_light", 32); // LightUniform = 4 floats * 4 bytes

        // Bind group layout: 0 = vp buffer, 1 = light buffer.
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
                    resource: light_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex buffer layout for MeshVertex.
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                pipeline::vertex_attr(0, wgpu::VertexFormat::Float32x3, 0),
                pipeline::vertex_attr(
                    1,
                    wgpu::VertexFormat::Float32x3,
                    std::mem::size_of::<f32>() as u64 * 3,
                ),
            ],
        };

        // Offscreen render target.
        let target_texture = crate::graphics::texture::render_target_texture(
            device,
            "scene_render_target",
            width,
            height,
            format,
        );
        let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout)],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            vp_buffer,
            light_buffer,
            bind_group,
            target_texture,
            target_view,
            fmt: format,
            width,
            height,
        })
    }

    /// Update the view-projection uniform.
    pub fn update_view_proj(&self, queue: &Queue, camera: &Camera) {
        let vp = camera.view_projection();
        buffer::write_buffer(queue, &self.vp_buffer, 0, &[ViewProjUniform { matrix: vp.m }]);
    }

    /// Update the light uniform.
    pub fn update_light(&self, queue: &Queue, light: &LightUniform) {
        buffer::write_buffer(queue, &self.light_buffer, 0, &[*light]);
    }

    /// Render a mesh to the offscreen target.
    pub fn render(
        &self,
        device: &Device,
        queue: &Queue,
        mesh: &MeshComponent,
    ) -> Result<()> {
        // Create vertex and index buffers for this mesh.
        // In a more optimized version, these would be pre-created or cached.
        let vbuf = buffer::vertex_buffer(device, "mesh_vertices", &mesh.vertices);
        let ibuf = buffer::index_buffer(device, "mesh_indices", &mesh.indices);

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
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, vbuf.slice(..));
            rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}