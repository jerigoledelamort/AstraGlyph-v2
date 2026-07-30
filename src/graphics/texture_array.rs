// Texture array: GPU storage for every texture the scene samples.
//
// One `TextureViewDimension::D2Array` binding rather than one binding per
// texture, because a WGSL fragment shader cannot index an array of *bindings*
// without `BINDING_ARRAY` (not universally available), but it can index the
// layers of one array texture with a plain `u32` — which is exactly what a
// material carries (`texture_index`). The alternative, an atlas, was rejected:
// tiling (UV > 1 with a Repeat sampler) is the bread-and-butter of level
// texturing, and an atlas can only fake it with wrap arithmetic in the shader,
// which breaks at mip transitions.
//
// The constraint an array imposes is that all layers share one size. Smaller
// images are padded up to the array size by *edge-clamping* rather than
// stretching: resampling would need a filter kernel and would blur a texture
// nobody asked to blur. UVs are rescaled at sample time instead — each layer
// records the fraction of the array surface it actually covers, and the shader
// multiplies. Repeat-tiling then happens in the shader (fract before rescale).
//
// Mip levels are computed on the CPU with a 2x2 box filter at upload time.
// At a 240x136 subpixel target almost every textured surface is minified;
// sampling mip 0 there aliases into per-pixel noise, which the ASCII stage
// amplifies into flickering garbage glyphs. A box filter is not Lanczos, but
// the output is quantized to character cells — the difference is unobservable.

use crate::engine::core::Result;
use wgpu::{Device, Queue};

/// Sentinel for "this material has no texture". Kept in sync with the WGSL
/// constant of the same name in `scene_shading.wgsl`.
pub const NO_TEXTURE: u32 = 0xFFFF_FFFF;

/// One texture prepared for upload: RGBA8 pixels plus its true size.
#[derive(Clone)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, RGBA8, row-major.
    pub pixels: Vec<u8>,
}

impl TextureData {
    /// Wrap raw RGBA8 bytes. Returns `None` when the byte count disagrees with
    /// the dimensions — a truncated buffer uploaded anyway would show as
    /// garbage rows, attributed to the decoder rather than the caller.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        if pixels.len() != (width as usize) * (height as usize) * 4 {
            return None;
        }
        Some(Self { width, height, pixels })
    }
}

/// Number of mip levels for a `width x height` base image.
pub fn mip_level_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// Downsample one RGBA8 image to half size with a 2x2 box filter.
///
/// Odd dimensions round up (`div_ceil`), matching wgpu's mip chain sizing; the
/// edge texel is repeated where a 2x2 block would run off the image. Alpha is
/// averaged like the colour channels — for the alpha-test materials this array
/// serves, averaging is what makes distant foliage fade out rather than
/// shimmer (each mip's 0.5 threshold cuts through an averaged edge).
pub fn downsample_rgba(width: u32, height: u32, pixels: &[u8]) -> (u32, u32, Vec<u8>) {
    let out_w = width.div_ceil(2).max(1);
    let out_h = height.div_ceil(2).max(1);
    let mut out = Vec::with_capacity((out_w * out_h * 4) as usize);
    for y in 0..out_h {
        for x in 0..out_w {
            let x0 = (x * 2).min(width - 1);
            let y0 = (y * 2).min(height - 1);
            let x1 = (x * 2 + 1).min(width - 1);
            let y1 = (y * 2 + 1).min(height - 1);
            for channel in 0..4usize {
                let mut sum = 0u32;
                for (sx, sy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                    sum += pixels[((sy * width + sx) * 4) as usize + channel] as u32;
                }
                out.push(((sum + 2) / 4) as u8);
            }
        }
    }
    (out_w, out_h, out)
}

/// Pad an RGBA8 image to `target_w x target_h` by repeating its edge texels.
///
/// Edge-clamp rather than zero-fill: the padded region *is* sampled when a
/// bilinear tap lands on the boundary of the covered region, and a black
/// border there would bleed dark seams into every tile edge.
pub fn pad_rgba(
    width: u32,
    height: u32,
    pixels: &[u8],
    target_w: u32,
    target_h: u32,
) -> Vec<u8> {
    debug_assert!(target_w >= width && target_h >= height);
    let mut out = Vec::with_capacity((target_w * target_h * 4) as usize);
    for y in 0..target_h {
        let sy = y.min(height - 1);
        for x in 0..target_w {
            let sx = x.min(width - 1);
            let base = ((sy * width + sx) * 4) as usize;
            out.extend_from_slice(&pixels[base..base + 4]);
        }
    }
    out
}

/// Where one texture landed in the array: its layer plus the fraction of the
/// layer surface its real pixels cover (1.0 for a texture that matched the
/// array size exactly). The shader multiplies UVs by this before sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureSlot {
    pub layer: u32,
    pub scale_u: f32,
    pub scale_v: f32,
}

/// GPU texture array plus its sampler and the per-layer UV scales.
pub struct TextureArray {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    slots: Vec<TextureSlot>,
    width: u32,
    height: u32,
}

impl TextureArray {
    /// Build the array from every texture the scene uses, in index order.
    ///
    /// `textures` may be empty: a 1x1 white placeholder array is created so the
    /// bind group always has a valid view, and `NO_TEXTURE` materials never
    /// sample it anyway. The array dimensions are the maximum over the inputs;
    /// smaller inputs are edge-padded and their `TextureSlot` records the UV
    /// scale that undoes the padding.
    pub fn new(device: &Device, queue: &Queue, textures: &[TextureData]) -> Result<Self> {
        let (width, height) = textures
            .iter()
            .fold((1u32, 1u32), |(w, h), t| (w.max(t.width), h.max(t.height)));
        let layer_count = textures.len().max(1) as u32;
        let mip_count = mip_level_count(width, height);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_texture_array"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layer_count,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mut slots = Vec::with_capacity(textures.len());
        for (layer, data) in textures.iter().enumerate() {
            slots.push(TextureSlot {
                layer: layer as u32,
                scale_u: data.width as f32 / width as f32,
                scale_v: data.height as f32 / height as f32,
            });
            // Mip 0: the (padded) source. Further mips: repeated box filter of
            // the *padded* image, so the mip chain sizing matches the texture's.
            let mut level_pixels = pad_rgba(data.width, data.height, &data.pixels, width, height);
            let (mut lw, mut lh) = (width, height);
            for mip in 0..mip_count {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &level_pixels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(lw * 4),
                        rows_per_image: Some(lh),
                    },
                    wgpu::Extent3d {
                        width: lw,
                        height: lh,
                        depth_or_array_layers: 1,
                    },
                );
                if mip + 1 < mip_count {
                    let (nw, nh, next) = downsample_rgba(lw, lh, &level_pixels);
                    level_pixels = next;
                    lw = nw;
                    lh = nh;
                }
            }
        }
        // No textures: upload one white texel so an accidental sample is a
        // visible "untextured white" rather than uninitialized memory.
        if textures.is_empty() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255, 255, 255, 255],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("scene_texture_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        // Repeat + trilinear. Repeat because tiling is the point of the array
        // (see the module comment); trilinear because the mips exist to be
        // blended between, and Nearest mip transitions pop visibly even
        // through the glyph quantizer.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene_texture_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
            slots,
            width,
            height,
        })
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Where texture `index` (the order given to `new`) landed.
    pub fn slot(&self, index: usize) -> Option<TextureSlot> {
        self.slots.get(index).copied()
    }

    pub fn layer_count(&self) -> usize {
        self.slots.len()
    }

    /// Array surface dimensions (every layer's padded size).
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Bytes of GPU memory the array occupies, mips included (computed, since
    /// wgpu does not report texture sizes). For the profiler.
    pub fn byte_size(&self) -> u64 {
        let mut total = 0u64;
        let (mut w, mut h) = (self.width, self.height);
        for _ in 0..mip_level_count(self.width, self.height) {
            total += w as u64 * h as u64 * 4;
            w = w.div_ceil(2).max(1);
            h = h.div_ceil(2).max(1);
        }
        total * self.slots.len().max(1) as u64
    }

    /// The two bind group layout entries (texture + sampler) for the given
    /// binding slots. Kept here so the scene pass cannot drift from the view
    /// dimension and sample type the array actually has.
    pub fn layout_entries(
        texture_binding: u32,
        sampler_binding: u32,
    ) -> [wgpu::BindGroupLayoutEntry; 2] {
        [
            wgpu::BindGroupLayoutEntry {
                binding: texture_binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: sampler_binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_count_is_log2_plus_one() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(2, 2), 2);
        assert_eq!(mip_level_count(256, 256), 9);
        assert_eq!(mip_level_count(256, 64), 9, "driven by the larger axis");
        assert_eq!(mip_level_count(100, 100), 7, "non-power-of-two rounds down: 64..127 -> 7");
        assert_eq!(mip_level_count(0, 0), 1, "degenerate input still gets one level");
    }

    #[test]
    fn downsample_averages_2x2_blocks() {
        // A 2x2 image of four solid values averages to their mean.
        let pixels = vec![
            0, 0, 0, 255, /**/ 100, 0, 0, 255, //
            0, 200, 0, 255, /**/ 0, 0, 40, 255,
        ];
        let (w, h, out) = downsample_rgba(2, 2, &pixels);
        assert_eq!((w, h), (1, 1));
        assert_eq!(out, vec![25, 50, 10, 255]);
    }

    #[test]
    fn downsample_rounds_odd_sizes_up_and_clamps_the_edge() {
        // 3x1: blocks are (0,1) and (2,2-clamped).
        let pixels = vec![
            10, 0, 0, 255, /**/ 30, 0, 0, 255, /**/ 100, 0, 0, 255,
        ];
        let (w, h, out) = downsample_rgba(3, 1, &pixels);
        assert_eq!((w, h), (2, 1));
        assert_eq!(out[0], 20, "average of 10 and 30");
        assert_eq!(out[4], 100, "edge block repeats the last texel");
    }

    #[test]
    fn downsample_chain_terminates_at_1x1() {
        let (mut w, mut h) = (256u32, 64u32);
        let mut pixels = vec![128u8; (w * h * 4) as usize];
        let mut steps = 0;
        while w > 1 || h > 1 {
            let (nw, nh, next) = downsample_rgba(w, h, &pixels);
            assert!(nw < w || nh < h, "each step must shrink");
            w = nw;
            h = nh;
            pixels = next;
            steps += 1;
            assert!(steps < 32, "chain must terminate");
        }
        assert_eq!(pixels.len(), 4);
        assert_eq!(steps + 1, mip_level_count(256, 64) as usize);
    }

    #[test]
    fn pad_clamps_edges_rather_than_zero_filling() {
        // 1x1 red padded to 2x2 must be red everywhere.
        let red = vec![200u8, 10, 10, 255];
        let out = pad_rgba(1, 1, &red, 2, 2);
        assert_eq!(out.len(), 16);
        for texel in out.chunks(4) {
            assert_eq!(texel, &[200, 10, 10, 255]);
        }
    }

    #[test]
    fn pad_preserves_the_original_region() {
        // 2x1 image (red, green) padded to 3x2.
        let pixels = vec![255, 0, 0, 255, /**/ 0, 255, 0, 255];
        let out = pad_rgba(2, 1, &pixels, 3, 2);
        let texel = |x: usize, y: usize| &out[(y * 3 + x) * 4..(y * 3 + x) * 4 + 4];
        assert_eq!(texel(0, 0), &[255, 0, 0, 255]);
        assert_eq!(texel(1, 0), &[0, 255, 0, 255]);
        assert_eq!(texel(2, 0), &[0, 255, 0, 255], "x clamps to the last column");
        assert_eq!(texel(0, 1), &[255, 0, 0, 255], "y clamps to the last row");
    }

    #[test]
    fn no_texture_sentinel_matches_the_scene_side_constant() {
        // Duplicated on purpose (module dependency direction); must never drift.
        assert_eq!(NO_TEXTURE, crate::scene::component::NO_TEXTURE);
    }

    #[test]
    fn texture_data_rejects_mismatched_sizes() {
        assert!(TextureData::new(2, 2, vec![0; 16]).is_some());
        assert!(TextureData::new(2, 2, vec![0; 15]).is_none());
        assert!(TextureData::new(0, 2, vec![]).is_none());
    }
}
