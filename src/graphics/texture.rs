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

/// Create a depth texture for depth testing in the scene pass.
///
/// `COPY_SRC` is included so the depth buffer can be read back to the CPU —
/// screen-space effects such as SSAO and the depth-driven cell subdivision run
/// on the CPU side of the ASCII pipeline.
pub fn depth_texture(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
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
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Create a depth texture that is also sampled elsewhere (e.g. a shadow map:
/// written as a depth attachment in one pass, read via `textureSampleCompare`
/// in another).
pub fn sampled_depth_texture(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
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
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}