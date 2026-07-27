// Texture creation utilities.

use wgpu::{Device, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};

/// Create a 2D render target texture for intermediate rendering (e.g. ASCII pass).
pub fn render_target_texture(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}