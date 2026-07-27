// Buffer creation utilities — no wgpu::util, no bytemuck.

use crate::engine::core::{cast_slice, Pod};
use wgpu::{Buffer, Device};

/// Create a buffer initialized with data, using mapped memory.
fn create_buffer_init(
    device: &Device,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: contents.len() as u64,
        usage,
        mapped_at_creation: true,
    });
    {
        let mut mapping = buffer.slice(..).get_mapped_range_mut()
            .expect("failed to map buffer for writing");
        mapping.copy_from_slice(contents);
    }
    buffer.unmap();
    buffer
}

/// Create a vertex buffer initialized with data.
pub fn vertex_buffer(device: &Device, label: &str, data: &[impl Pod]) -> Buffer {
    create_buffer_init(
        device,
        label,
        cast_slice(data),
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    )
}

/// Create an index buffer initialized with data.
pub fn index_buffer(device: &Device, label: &str, data: &[u32]) -> Buffer {
    create_buffer_init(
        device,
        label,
        cast_slice(data),
        wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    )
}

/// Create a uniform buffer of the given size (uninitialized).
pub fn uniform_buffer(device: &Device, label: &str, size: u64) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Write data to a buffer via the queue.
pub fn write_buffer(queue: &wgpu::Queue, buffer: &Buffer, offset: u64, data: &[impl Pod]) {
    queue.write_buffer(buffer, offset, cast_slice(data));
}