// ASCII pass: reads back the scene render target and converts pixels to
// glyph instance data for the composite pass.
//
// Readback is double-buffered: each frame we kick off an async map on one
// buffer and non-blockingly poll the other buffer submitted the previous
// frame. The CPU never blocks waiting for the GPU — at the cost of the
// composited image trailing the scene by up to ~1 frame, which is
// imperceptible at interactive framerates.

use std::sync::mpsc;
use crate::renderer::composite_pass::InstanceData;
use wgpu::{Buffer, Device, Extent3d, Queue};

/// One of the two ping-ponged readback buffers.
struct ReadbackSlot {
    buffer: Buffer,
    /// Set once `map_async` has been issued; cleared once the result has
    /// been consumed (successfully or not). While set, this buffer must not
    /// be copied into or re-mapped.
    pending: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// Handles GPU texture readback and ASCII conversion.
pub struct AsciiProcessor {
    slots: [ReadbackSlot; 2],
    /// Index into `slots` that will receive the next copy + map_async.
    write_index: usize,
    /// Buffer size in bytes (padded to 256-byte rows).
    #[allow(dead_code)]
    buffer_size: u64,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    /// Most recently fully-read frame. Reused whenever neither slot has a
    /// freshly completed map yet, so callers always get a full frame instead
    /// of a partial/stale one.
    last_pixels: Vec<[u8; 4]>,
}

impl AsciiProcessor {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        // wgpu requires buffer rows to be padded to 256 bytes.
        let bytes_per_row = (width * 4 + 255) & !255; // round up to 256
        let buffer_size = (bytes_per_row * height) as u64;

        let make_buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };

        Self {
            slots: [
                ReadbackSlot { buffer: make_buffer("ascii_readback_buffer_0"), pending: None },
                ReadbackSlot { buffer: make_buffer("ascii_readback_buffer_1"), pending: None },
            ],
            write_index: 0,
            buffer_size,
            width,
            height,
            bytes_per_row,
            last_pixels: vec![[0, 0, 0, 255]; (width * height) as usize],
        }
    }

    /// Advance the double-buffered readback by one frame:
    /// - Non-blockingly check the slot submitted on a previous frame; if its
    ///   `map_async` has completed, extract its pixels into `last_pixels`.
    /// - Kick off a fresh copy + `map_async` on the other slot (unless it is
    ///   still waiting on a previous map — a rare backpressure case).
    ///
    /// Returns the most recently completed full frame. This never blocks on
    /// the GPU; the result may lag the current frame by roughly one frame.
    pub fn read_pixels(
        &mut self,
        device: &Device,
        queue: &Queue,
        source: &wgpu::Texture,
    ) -> Vec<[u8; 4]> {
        // Give any in-flight map_async callbacks a chance to fire, without blocking.
        let _ = device.poll(wgpu::PollType::Poll);

        let read_index = 1 - self.write_index;
        if let Some(rx) = &self.slots[read_index].pending {
            if let Ok(result) = rx.try_recv() {
                self.slots[read_index].pending = None;
                if result.is_ok() {
                    self.extract_pixels(read_index);
                }
                self.slots[read_index].buffer.unmap();
            }
            // else: still pending — keep serving `last_pixels` from before.
        }

        // Only submit new work into a slot that isn't already awaiting a map.
        if self.slots[self.write_index].pending.is_none() {
            self.submit_copy(device, queue, source, self.write_index);
            self.write_index = read_index;
        }

        self.last_pixels.clone()
    }

    /// Copy the scene texture into `slot_index`'s buffer and start an async map.
    fn submit_copy(&mut self, device: &Device, queue: &Queue, source: &wgpu::Texture, slot_index: usize) {
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
                buffer: &self.slots[slot_index].buffer,
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

        let (tx, rx) = mpsc::channel();
        self.slots[slot_index]
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.slots[slot_index].pending = Some(rx);
    }

    /// Read the mapped bytes out of `slot_index`'s buffer into `last_pixels`,
    /// skipping row-padding bytes. Assumes the buffer is currently mapped.
    fn extract_pixels(&mut self, slot_index: usize) {
        let mapped = self.slots[slot_index].buffer.slice(..).get_mapped_range()
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
        self.last_pixels = pixels;
    }

    /// Convert raw pixels to glyph instance data for the composite pass.
    ///
    /// Each pixel becomes one glyph quad positioned on the screen.
    pub fn pixels_to_instances(
        &self,
        pixels: &[[u8; 4]],
        _screen_w: u32,
        _screen_h: u32,
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
