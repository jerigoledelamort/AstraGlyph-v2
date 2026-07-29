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

/// How a cell's colour is turned into a glyph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GlyphStyle {
    /// Unicode quadrant block elements driven by 2x2 subpixels — twice the
    /// effective resolution, reads as tone and silhouette.
    #[default]
    Blocks,
    /// The classic brightness ramp (' ' . : - = + * # % @ and shade blocks),
    /// one glyph per averaged cell.
    Ramp,
}

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
    /// Depth readback, ping-ponged exactly like the colour slots above.
    /// Depth32Float is 4 bytes per pixel, so the row padding differs from the
    /// colour path only in that it holds f32 rather than RGBA8.
    depth_slots: [ReadbackSlot; 2],
    depth_write_index: usize,
    depth_bytes_per_row: u32,
    /// Most recently completed depth frame, 0..1 window-space depth
    /// (1.0 = far plane / nothing drawn).
    last_depth: Vec<f32>,
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

        // Depth32Float is also 4 bytes per pixel, so the padded row size matches.
        let depth_bytes_per_row = bytes_per_row;
        let depth_buffer_size = (depth_bytes_per_row * height) as u64;
        let make_depth_buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: depth_buffer_size,
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
            depth_slots: [
                ReadbackSlot { buffer: make_depth_buffer("ascii_depth_readback_0"), pending: None },
                ReadbackSlot { buffer: make_depth_buffer("ascii_depth_readback_1"), pending: None },
            ],
            depth_write_index: 0,
            depth_bytes_per_row,
            // 1.0 = far plane, i.e. "nothing here", the correct neutral value
            // for SSAO and subdivision before the first frame lands.
            last_depth: vec![1.0; (width * height) as usize],
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

    /// Advance the double-buffered *depth* readback by one frame, mirroring
    /// `read_pixels`. Returns the most recently completed depth frame as
    /// window-space depth in 0..1 (1.0 = far plane / nothing drawn).
    ///
    /// Never blocks; the result may lag the current frame by roughly one frame,
    /// which is harmless for the screen-space effects that consume it.
    pub fn read_depth(
        &mut self,
        device: &Device,
        queue: &Queue,
        depth_source: &wgpu::Texture,
    ) -> Vec<f32> {
        let _ = device.poll(wgpu::PollType::Poll);

        let read_index = 1 - self.depth_write_index;
        if let Some(rx) = &self.depth_slots[read_index].pending {
            if let Ok(result) = rx.try_recv() {
                self.depth_slots[read_index].pending = None;
                if result.is_ok() {
                    self.extract_depth(read_index);
                }
                self.depth_slots[read_index].buffer.unmap();
            }
        }

        if self.depth_slots[self.depth_write_index].pending.is_none() {
            self.submit_depth_copy(device, queue, depth_source, self.depth_write_index);
            self.depth_write_index = read_index;
        }

        self.last_depth.clone()
    }

    /// Copy the depth texture into `slot_index`'s buffer and start an async map.
    fn submit_depth_copy(
        &mut self,
        device: &Device,
        queue: &Queue,
        source: &wgpu::Texture,
        slot_index: usize,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ascii_depth_readback_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.depth_slots[slot_index].buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.depth_bytes_per_row),
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
        self.depth_slots[slot_index]
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.depth_slots[slot_index].pending = Some(rx);
    }

    /// Decode the mapped depth bytes into `last_depth`, skipping row padding.
    /// As with `extract_pixels`, a failed mapping keeps the previous frame rather
    /// than bringing the application down.
    fn extract_depth(&mut self, slot_index: usize) {
        let Ok(mapped) = self.depth_slots[slot_index].buffer.slice(..).get_mapped_range() else {
            return;
        };
        let mut depth = Vec::with_capacity((self.width * self.height) as usize);

        for row in 0..self.height {
            let row_start = (row * self.depth_bytes_per_row) as usize;
            for col in 0..self.width {
                let offset = row_start + (col * 4) as usize;
                let bytes = [
                    mapped[offset],
                    mapped[offset + 1],
                    mapped[offset + 2],
                    mapped[offset + 3],
                ];
                depth.push(f32::from_le_bytes(bytes));
            }
        }

        drop(mapped);
        self.last_depth = depth;
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
    /// skipping row-padding bytes.
    ///
    /// A failed mapping is not fatal: it happens when the surface or device goes
    /// through a bad state (resize, minimize, device loss) and the buffer is torn
    /// down under us. Keeping the previous frame is strictly better than
    /// panicking the whole application.
    fn extract_pixels(&mut self, slot_index: usize) {
        let Ok(mapped) = self.slots[slot_index].buffer.slice(..).get_mapped_range() else {
            return;
        };
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

    /// Convert a 2x-supersampled pixel buffer into one glyph instance per cell.
    ///
    /// The processor's own `width`/`height` are the SUBPIXEL dimensions, so the
    /// cell grid is half that in each axis. Two styles are supported:
    ///
    /// - `Blocks`: each cell's 2x2 subpixel block picks one of 16 quadrant block
    ///   glyphs, doubling the effective resolution (see `ascii::blocks`).
    /// - `Ramp`: the 2x2 block is averaged down to one colour and mapped onto the
    ///   brightness ramp — the classic look, but now anti-aliased by the
    ///   supersampling rather than point-sampled.
    pub fn subpixels_to_instances(
        &self,
        pixels: &[[u8; 4]],
        style: GlyphStyle,
    ) -> Vec<InstanceData> {
        let cols = self.width / 2;
        let rows = self.height / 2;
        if cols == 0 || rows == 0 {
            return Vec::new();
        }

        let cell_w_ndc = 2.0 / cols as f32;
        let cell_h_ndc = 2.0 / rows as f32;
        let mut instances = Vec::with_capacity((cols * rows) as usize);

        for row in 0..rows {
            for col in 0..cols {
                let sub = crate::ascii::blocks::gather_subpixels(
                    pixels, self.width, self.height, col, row,
                );

                let (glyph_index, color) = match style {
                    GlyphStyle::Blocks => {
                        let cell = crate::ascii::blocks::classify(&sub);
                        (crate::ascii::block_glyph_index(cell.pattern), cell.color)
                    }
                    GlyphStyle::Ramp => {
                        let avg = crate::ascii::blocks::average_subpixels(&sub);
                        let lum = crate::ascii::luminance(avg);
                        (crate::ascii::glyph_atlas::brightness_to_index(lum), avg)
                    }
                };

                instances.push(InstanceData {
                    ndc_x: -1.0 + col as f32 * cell_w_ndc,
                    ndc_y: 1.0 - row as f32 * cell_h_ndc,
                    width: cell_w_ndc,
                    height: cell_h_ndc,
                    glyph_index,
                    color_r: color[0],
                    color_g: color[1],
                    color_b: color[2],
                });
            }
        }

        instances
    }

    /// Same as `subpixels_to_instances`, but following a non-uniform tile layout
    /// (see `ascii::grid_layout`): merged tiles become one larger glyph.
    ///
    /// A merged tile covers several cells, so there is no single 2x2 block to
    /// resolve into quadrants — its whole area is averaged and drawn solid
    /// (blocks) or mapped onto the ramp. Span-1 tiles keep the per-cell treatment.
    pub fn subpixels_to_instances_tiled(
        &self,
        pixels: &[[u8; 4]],
        tiles: &[crate::ascii::Tile],
        style: GlyphStyle,
    ) -> Vec<InstanceData> {
        let cols = self.width / 2;
        let rows = self.height / 2;
        if cols == 0 || rows == 0 {
            return Vec::new();
        }

        let cell_w_ndc = 2.0 / cols as f32;
        let cell_h_ndc = 2.0 / rows as f32;
        let mut instances = Vec::with_capacity(tiles.len());

        for tile in tiles {
            let (glyph_index, color) = if tile.span == 1 {
                let sub = crate::ascii::blocks::gather_subpixels(
                    pixels, self.width, self.height, tile.col, tile.row,
                );
                self.style_glyph(&sub, style)
            } else {
                let avg = self.average_tile(pixels, tile);
                match style {
                    GlyphStyle::Blocks => (crate::ascii::block_glyph_index(0b1111), avg),
                    GlyphStyle::Ramp => (
                        crate::ascii::glyph_atlas::brightness_to_index(crate::ascii::luminance(avg)),
                        avg,
                    ),
                }
            };

            let span = tile.span as f32;
            instances.push(InstanceData {
                ndc_x: -1.0 + tile.col as f32 * cell_w_ndc,
                ndc_y: 1.0 - tile.row as f32 * cell_h_ndc,
                width: cell_w_ndc * span,
                height: cell_h_ndc * span,
                glyph_index,
                color_r: color[0],
                color_g: color[1],
                color_b: color[2],
            });
        }

        instances
    }

    /// Glyph index and colour for one cell's 2x2 subpixel block.
    fn style_glyph(
        &self,
        sub: &crate::ascii::blocks::Subpixels,
        style: GlyphStyle,
    ) -> (u32, [f32; 3]) {
        match style {
            GlyphStyle::Blocks => {
                let cell = crate::ascii::blocks::classify(sub);
                (crate::ascii::block_glyph_index(cell.pattern), cell.color)
            }
            GlyphStyle::Ramp => {
                let avg = crate::ascii::blocks::average_subpixels(sub);
                let lum = crate::ascii::luminance(avg);
                (crate::ascii::glyph_atlas::brightness_to_index(lum), avg)
            }
        }
    }

    /// Mean colour of every subpixel a merged tile covers.
    fn average_tile(&self, pixels: &[[u8; 4]], tile: &crate::ascii::Tile) -> [f32; 3] {
        let mut sum = [0.0f32; 3];
        let mut count = 0.0f32;
        for dy in 0..(tile.span * 2) {
            for dx in 0..(tile.span * 2) {
                let x = tile.col * 2 + dx;
                let y = tile.row * 2 + dy;
                if x >= self.width || y >= self.height {
                    continue;
                }
                let idx = (y as usize) * (self.width as usize) + (x as usize);
                if let Some(p) = pixels.get(idx) {
                    sum[0] += p[0] as f32 / 255.0;
                    sum[1] += p[1] as f32 / 255.0;
                    sum[2] += p[2] as f32 / 255.0;
                    count += 1.0;
                }
            }
        }
        if count > 0.0 {
            [sum[0] / count, sum[1] / count, sum[2] / count]
        } else {
            [0.0; 3]
        }
    }

    /// Turn the opaque cells of a UI overlay into glyph instances.
    ///
    /// These are meant to be appended AFTER the scene instances: the composite
    /// pass draws instances in order with alpha blending, so later quads land on
    /// top. Emitting the overlay as extra instances (rather than rewriting scene
    /// cells) keeps it independent of the scene's cell layout — it works
    /// unchanged whether the grid is uniform or has merged tiles.
    pub fn overlay_to_instances(&self, overlay: &crate::ascii::Overlay) -> Vec<InstanceData> {
        let cols = overlay.cols();
        let rows = overlay.rows();
        if cols == 0 || rows == 0 {
            return Vec::new();
        }

        // Overlay cells are laid out on the same grid as the scene cells.
        let cell_w_ndc = 2.0 / self.width as f32;
        let cell_h_ndc = 2.0 / self.height as f32;

        let mut instances = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let Some(cell) = overlay.cell(col, row) else {
                    continue;
                };
                if !cell.opaque {
                    continue;
                }
                instances.push(InstanceData {
                    ndc_x: -1.0 + col as f32 * cell_w_ndc,
                    ndc_y: 1.0 - row as f32 * cell_h_ndc,
                    width: cell_w_ndc,
                    height: cell_h_ndc,
                    glyph_index: cell.glyph_index,
                    color_r: cell.color[0],
                    color_g: cell.color[1],
                    color_b: cell.color[2],
                });
            }
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

#[cfg(test)]
mod tests {
    use crate::ascii::{compute_tiles, SubdivisionPolicy, Tile};

    /// NDC placement maths, mirrored from `pixels_to_instances_tiled` so it can
    /// be checked without a GPU device (constructing an AsciiProcessor needs one).
    fn tile_rect(tile: &Tile, cols: u32, rows: u32) -> (f32, f32, f32, f32) {
        let cell_w = 2.0 / cols as f32;
        let cell_h = 2.0 / rows as f32;
        (
            -1.0 + tile.col as f32 * cell_w,
            1.0 - tile.row as f32 * cell_h,
            cell_w * tile.span as f32,
            cell_h * tile.span as f32,
        )
    }

    #[test]
    fn merged_tiles_tile_the_screen_without_gaps() {
        // A flat far frame merges into 2x2 tiles; the resulting quads must still
        // exactly cover NDC -1..1 on both axes.
        let (cols, rows) = (8u32, 8u32);
        let depth = vec![1.0f32; (cols * rows) as usize];
        let policy = SubdivisionPolicy { merge_depth: 0.9, depth_tolerance: 0.01, max_span: 2 };
        let tiles = compute_tiles(&depth, cols, rows, &policy);

        let mut area = 0.0f32;
        for tile in &tiles {
            let (x, y, w, h) = tile_rect(tile, cols, rows);
            assert!(x >= -1.0 - 1e-6 && x + w <= 1.0 + 1e-6, "tile out of NDC on x: {x} + {w}");
            assert!(y <= 1.0 + 1e-6 && y - h >= -1.0 - 1e-6, "tile out of NDC on y: {y} - {h}");
            area += w * h;
        }
        assert!((area - 4.0).abs() < 1e-4, "quads must cover the full 2x2 NDC area, got {area}");
    }

    #[test]
    fn a_merged_tile_is_twice_the_size_of_a_base_cell() {
        let (cols, rows) = (4u32, 4u32);
        let base = tile_rect(&Tile { col: 0, row: 0, span: 1 }, cols, rows);
        let merged = tile_rect(&Tile { col: 0, row: 0, span: 2 }, cols, rows);
        assert!((merged.2 - base.2 * 2.0).abs() < 1e-6);
        assert!((merged.3 - base.3 * 2.0).abs() < 1e-6);
        // Both anchor at the same top-left corner.
        assert!((merged.0 - base.0).abs() < 1e-6);
        assert!((merged.1 - base.1).abs() < 1e-6);
    }

    #[test]
    fn first_cell_starts_at_the_top_left_of_ndc() {
        let (x, y, _, _) = tile_rect(&Tile { col: 0, row: 0, span: 1 }, 10, 10);
        assert!((x + 1.0).abs() < 1e-6, "x should start at -1");
        assert!((y - 1.0).abs() < 1e-6, "y should start at +1");
    }
}
