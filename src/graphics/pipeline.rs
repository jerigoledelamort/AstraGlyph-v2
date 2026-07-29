// Pipeline and shader module utilities.

use wgpu::Device;

/// Compile a WGSL shader source into a ShaderModule.
pub fn shader_module(device: &Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    })
}

/// Helper to load shader source and compile it.
///
/// Shader sources are embedded as `&str` in the binary (no file I/O at runtime).
pub fn compile_shader(device: &Device, label: &str, source: &str) -> wgpu::ShaderModule {
    shader_module(device, label, source)
}

/// Build a simple vertex attribute descriptor from a format and offset.
pub const fn vertex_attr(location: u32, format: wgpu::VertexFormat, offset: u64) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        shader_location: location,
        format,
        offset,
    }
}

/// Create a render pipeline from a descriptor builder.
#[allow(dead_code)]
pub fn render_pipeline(
    device: &Device,
_label: &str,
    desc: &wgpu::RenderPipelineDescriptor,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(desc)
}