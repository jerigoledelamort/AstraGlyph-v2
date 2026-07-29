// Composite pass: renders ASCII glyphs to the screen surface.
// Uses a glyph atlas texture and per-cell instance data.

use crate::ascii::{build_combined_atlas, combined_glyph_count, GLYPH_SIZE};
use crate::engine::core::{cast_slice, Pod, Result};
use crate::graphics::pipeline;
use wgpu::{Device, Queue, TextureFormat};

/// Per-cell instance data sent to the GPU as a storage buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct InstanceData {
    pub ndc_x: f32,
    pub ndc_y: f32,
    pub width: f32,
    pub height: f32,
    pub glyph_index: u32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
}

unsafe impl Pod for InstanceData {}

/// Composite pipeline: draws glyph quads to the screen.
pub struct CompositePipeline {
    pipeline: wgpu::RenderPipeline,
    glyph_texture: wgpu::Texture,
    glyph_view: wgpu::TextureView,
    glyph_sampler: wgpu::Sampler,
    instance_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl CompositePipeline {
pub fn new(device: &Device, screen_format: TextureFormat, max_instances: u32) -> Result<Self> {
        // The atlas holds the shading glyphs AND the text font (see ascii/mod.rs),
        // so text and scene cells can be drawn from one texture.
        let glyph_count_val = combined_glyph_count() as u32;
        let atlas_width = glyph_count_val * GLYPH_SIZE;

        // Create empty glyph atlas texture; data uploaded later via upload_atlas().
        let glyph_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glyph_atlas"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: GLYPH_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // We'll upload atlas data via the queue in `upload_atlas`.
        // For now, create the view and sampler.
        let glyph_view = glyph_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let glyph_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("glyph_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Instance buffer (sized for the actual number of cells).
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: (max_instances as u64) * std::mem::size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout: 0 = instance buffer, 1 = glyph texture, 2 = sampler.
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("composite_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        // Also visible to the vertex stage, which reads the atlas
                        // dimensions to derive the glyph count for UV slicing.
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&glyph_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&glyph_sampler),
                },
            ],
        });

        // Shaders.
        let vs_source = include_str!("../graphics/shaders/composite_vertex.wgsl");
        let fs_source = include_str!("../graphics/shaders/composite_fragment.wgsl");
        let vs_module = pipeline::compile_shader(device, "composite_vertex", vs_source);
        let fs_module = pipeline::compile_shader(device, "composite_fragment", fs_source);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: screen_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            glyph_texture,
            glyph_view,
            glyph_sampler,
            instance_buffer,
            bind_group,
        })
    }

    /// Upload the glyph atlas data to the texture.
    /// Converts RGBA atlas data to single-channel R8.
    pub fn upload_atlas(&self, queue: &Queue) {
        let rgba_atlas = build_combined_atlas();
        let glyph_count_val = combined_glyph_count() as u32;
        let atlas_width = glyph_count_val * GLYPH_SIZE;

        // Convert RGBA (4 bytes per pixel) to R8 (1 byte per pixel) by extracting the red channel.
        let r8_data: Vec<u8> = rgba_atlas
            .chunks(4)
            .map(|pixel| pixel[0]) // R channel (255 for visible, 0 for invisible)
            .collect();

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.glyph_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &r8_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_width),
                rows_per_image: Some(GLYPH_SIZE),
            },
            wgpu::Extent3d {
                width: atlas_width,
                height: GLYPH_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Update instance data from the cell grid.
    pub fn update_instances(
        &self,
        queue: &Queue,
        instances: &[InstanceData],
    ) {
        queue.write_buffer(&self.instance_buffer, 0, cast_slice(instances));
    }

    /// Render glyphs to the screen.
    pub fn render(
        &self,
        device: &Device,
        queue: &Queue,
        view: &wgpu::TextureView,
        instance_count: u32,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite_encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            // 6 vertices per quad, instance_count instances.
            rpass.draw(0..6, 0..instance_count);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}