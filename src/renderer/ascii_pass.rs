// ASCII pass: reads back the scene render target and converts pixels to
// glyph instance data for the composite pass.

use std::sync::mpsc;
use crate::engine::core::Result;
use crate::renderer::composite_pass::InstanceData;
use wgpu::{Buffer, Device, Extent3d, Queue};

/// Handles GPU texture readback and ASCII conversion.
pub struct AsciiProcessor {
    /// Staging buffer for texture readback (CPU-visible).
    readback_buffer: Buffer,
    /// Buffer size in bytes (padded to 256-byte rows).
    buffer_size: u64,
    width: u32,
    height: u32,
    bytes_per_row: u32,
}

impl AsciiProcessor {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        // wgpu requires buffer rows to be padded to 256 bytes.
        let bytes_per_row = (width * 4 + 255) & !255; // round up to 256
        let buffer_size = (bytes_per_row * height) as u64;

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ascii_readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            readback_buffer,
            buffer_size,
            width,
            height,
            bytes_per_row,
        }
    }

    /// Copy the scene render target into the readback buffer and read pixels.
    ///
    /// This blocks until the GPU finishes the copy and the buffer is mapped.
    pub fn read_pixels(
        &self,
        device: &Device,
        queue: &Queue,
        source: &wgpu::Texture,
    ) -> Result<Vec<[u8; 4]>> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ascii_readback_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer and wait for completion.
        let (tx, rx) = mpsc::channel();
        self.readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        // Poll the device until the mapping is done.
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|e| {
            crate::engine::core::EngineError::Graphics(format!("poll error: {e}"))
        })?;

        rx.recv()
            .map_err(|e| {
                crate::engine::core::EngineError::Graphics(format!(
                    "channel error: {e}"
                ))
            })?
            .map_err(|e| {
                crate::engine::core::EngineError::Graphics(format!(
                    "map error: {e}"
                ))
            })?;

        // Read the mapped data, skipping padding bytes.
        let mapped = self.readback_buffer.slice(..).get_mapped_range()
            .expect("failed to map readback buffer for reading");
        let mut pixels = Vec::with_capacity((self.width * self.height) as usize);

        for row in 0..self.height {
            let row_start = (row * self.bytes_per_row) as usize;
            for col in 0..self.width {
                let offset = row_start + (col * 4) as usize;
                pixels.push([
                    mapped[offset],
                    mapped[offset + 1],
                    mapped[offset + 2],
                    mapped[offset + 3],
                ]);
            }
        }

        drop(mapped);
        self.readback_buffer.unmap();

        Ok(pixels)
    }

    /// Convert raw pixels to glyph instance data for the composite pass.
    ///
    /// Each pixel becomes one glyph quad positioned on the screen.
    pub fn pixels_to_instances(
        &self,
        pixels: &[[u8; 4]],
        screen_w: u32,
        screen_h: u32,
    ) -> Vec<InstanceData> {
        let cols = self.width;
        let rows = self.height;

        // Compute the glyph quad size in NDC to fill the screen while
        // maintaining the cell grid aspect ratio.
        let cell_w_ndc = 2.0 / cols as f32;
        let cell_h_ndc = 2.0 / rows as f32;

        let mut instances = Vec::with_capacity(pixels.len());

        for (i, pixel) in pixels.iter().enumerate() {
            let col = (i as u32) % cols;
            let row = (i as u32) / cols;

            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let luminance = 0.299 * r + 0.587 * g + 0.114 * b;

            let glyph_index = crate::ascii::glyph_atlas::brightness_to_index(luminance);

            // NDC position: x goes left to right, y goes top to bottom.
            // Top-left of screen is (-1, 1), bottom-right is (1, -1).
            let ndc_x = -1.0 + col as f32 * cell_w_ndc;
            let ndc_y = 1.0 - row as f32 * cell_h_ndc;

            instances.push(InstanceData {
                ndc_x,
                ndc_y,
                width: cell_w_ndc,
                height: cell_h_ndc,
                glyph_index,
                color_r: r,
                color_g: g,
                color_b: b,
            });
        }

        instances
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}