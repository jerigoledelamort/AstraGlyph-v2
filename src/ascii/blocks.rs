// Unicode Block Elements (U+2580..U+259F) rendering.
//
// Motivation: the brightness-ramp glyphs (' ' . : - = + * # % @) read as
// *texture* — the eye sees repeated character shapes rather than the underlying
// image. Block elements read as tone and silhouette instead, and more
// importantly they let one cell carry FOUR independent samples: the quadrant
// blocks encode every combination of upper-left / upper-right / lower-left /
// lower-right being filled.
//
// So the scene is rendered at 2x the glyph grid resolution and each cell
// consumes its own 2x2 subpixel block. That doubles the effective image
// resolution horizontally and vertically at the same glyph count — which is the
// detail the fixed-size render target could not otherwise provide.
//
// Quadrant bit layout (matches `Quadrants::PATTERN_*` below):
//   bit 0 = upper-left, bit 1 = upper-right, bit 2 = lower-left, bit 3 = lower-right
// All 16 combinations exist as real characters, so the output stays printable
// text rather than an invented encoding.

use crate::ascii::glyph_atlas::{Glyph, GLYPH_BYTES, GLYPH_SIZE};

/// Number of block glyphs (every 2x2 quadrant combination).
pub const BLOCK_COUNT: usize = 16;

/// The 16 quadrant combinations, indexed by their 4-bit pattern.
///
/// Listed for documentation and for tests that check the mapping; the atlas
/// bitmaps are generated from the bit pattern, not from these characters.
pub const BLOCK_CHARS: [char; BLOCK_COUNT] = [
    ' ',        // 0b0000 nothing
    '▘',        // 0b0001 U+2598 upper-left
    '▝',        // 0b0010 U+259D upper-right
    '▀',        // 0b0011 U+2580 upper half
    '▖',        // 0b0100 U+2596 lower-left
    '▌',        // 0b0101 U+258C left half
    '▞',        // 0b0110 U+259E upper-right + lower-left
    '▛',        // 0b0111 U+259B all but lower-right
    '▗',        // 0b1000 U+2597 lower-right
    '▚',        // 0b1001 U+259A upper-left + lower-right
    '▐',        // 0b1010 U+2590 right half
    '▜',        // 0b1011 U+259C all but lower-left
    '▄',        // 0b1100 U+2584 lower half
    '▙',        // 0b1101 U+2599 all but upper-right
    '▟',        // 0b1110 U+259F all but upper-left
    '█',        // 0b1111 U+2588 full block
];

/// Bit for the upper-left quadrant.
pub const UL: u8 = 0b0001;
/// Bit for the upper-right quadrant.
pub const UR: u8 = 0b0010;
/// Bit for the lower-left quadrant.
pub const LL: u8 = 0b0100;
/// Bit for the lower-right quadrant.
pub const LR: u8 = 0b1000;

/// Render the block glyph for a 4-bit quadrant `pattern` into the atlas's 8x8
/// `Glyph` format. Bits above the low four are ignored.
pub fn render_block(pattern: u8) -> Glyph {
    let mut glyph: Glyph = [[0u8; 4]; GLYPH_BYTES];
    let size = GLYPH_SIZE as usize;
    let half = size / 2;

    let fill = |x0: usize, y0: usize, g: &mut Glyph| {
        for y in y0..(y0 + half) {
            for x in x0..(x0 + half) {
                g[y * size + x] = [255, 255, 255, 255];
            }
        }
    };

    if pattern & UL != 0 {
        fill(0, 0, &mut glyph);
    }
    if pattern & UR != 0 {
        fill(half, 0, &mut glyph);
    }
    if pattern & LL != 0 {
        fill(0, half, &mut glyph);
    }
    if pattern & LR != 0 {
        fill(half, half, &mut glyph);
    }

    glyph
}

/// Build the block section as flat RGBA bytes, same layout as
/// `glyph_atlas::build_atlas()`.
pub fn build_block_atlas() -> Vec<u8> {
    let mut atlas = Vec::with_capacity(BLOCK_COUNT * GLYPH_BYTES * 4);
    for pattern in 0..BLOCK_COUNT as u8 {
        let glyph = render_block(pattern);
        for pixel in &glyph {
            atlas.extend_from_slice(pixel);
        }
    }
    atlas
}

/// The four subpixel colours of one cell, in quadrant order:
/// upper-left, upper-right, lower-left, lower-right.
pub type Subpixels = [[f32; 3]; 4];

/// What a cell should be drawn as: which quadrant glyph, and the colour of the
/// filled quadrants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockCell {
    /// 4-bit quadrant pattern (index into the block section of the atlas).
    pub pattern: u8,
    /// Colour for the filled quadrants.
    pub color: [f32; 3],
    /// Colour for the EMPTY quadrants.
    ///
    /// A quadrant glyph only covers part of its cell, so the remainder needs a
    /// colour too — otherwise it shows whatever the target was cleared to
    /// (black), and every cell straddling a gradient punches a dark hole into the
    /// image. Filling it with the mean of the unlit subpixels is what makes the
    /// result read as a continuous 2x-resolution image.
    pub background: [f32; 3],
    /// True when the cell had no meaningful internal contrast, so the pattern
    /// was chosen as "all four" and the colour carries the whole tone. Callers
    /// that prefer a shading-ramp glyph for flat areas can branch on this.
    pub flat: bool,
}

/// Luminance, matching the coefficients used across the pipeline (BT.601).
fn luminance(rgb: [f32; 3]) -> f32 {
    0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
}

/// Minimum luminance spread within a cell before it is treated as having real
/// internal detail. Below this the cell is flat and gets a solid block, because
/// thresholding noise would produce arbitrary quadrant patterns that flicker
/// between frames.
const FLAT_EPSILON: f32 = 1.0 / 255.0;

/// Choose the quadrant pattern and colour for one cell from its four subpixels.
///
/// The threshold is the cell's own midpoint (min+max)/2, so the decision is
/// local: a cell straddling a silhouette edge splits along that edge, while a
/// cell inside a smooth gradient stays solid. The colour is the average of the
/// quadrants that ended up filled, which keeps edges from darkening.
pub fn classify(sub: &Subpixels) -> BlockCell {
    let lum = [
        luminance(sub[0]),
        luminance(sub[1]),
        luminance(sub[2]),
        luminance(sub[3]),
    ];

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for l in lum {
        let l = if l.is_finite() { l } else { 0.0 };
        if l < min {
            min = l;
        }
        if l > max {
            max = l;
        }
    }

    // Flat cell: no interior detail to resolve. Fill it completely and let the
    // colour carry the tone — with true colour that is the most faithful
    // representation available, and it cannot flicker.
    if !(max - min > FLAT_EPSILON) {
        let mean = average(&[sub[0], sub[1], sub[2], sub[3]]);
        return BlockCell {
            pattern: 0b1111,
            color: mean,
            // Fully covered, so the background is never sampled; keeping it equal
            // to the foreground means a stray mask value can't darken the cell.
            background: mean,
            flat: true,
        };
    }

    let threshold = (min + max) * 0.5;
    let bits = [UL, UR, LL, LR];
    let mut pattern = 0u8;
    let mut filled: Vec<[f32; 3]> = Vec::with_capacity(4);
    let mut empty: Vec<[f32; 3]> = Vec::with_capacity(4);
    for i in 0..4 {
        let l = if lum[i].is_finite() { lum[i] } else { 0.0 };
        if l >= threshold {
            pattern |= bits[i];
            filled.push(sub[i]);
        } else {
            empty.push(sub[i]);
        }
    }

    // `max` is above the threshold by construction, so at least one quadrant is
    // always filled and `filled` is never empty. `empty` can be empty, in which
    // case the background is unused and mirrors the foreground.
    let color = average(&filled);
    let background = if empty.is_empty() {
        color
    } else {
        average(&empty)
    };

    BlockCell { pattern, color, background, flat: false }
}

fn average(colors: &[[f32; 3]]) -> [f32; 3] {
    if colors.is_empty() {
        return [0.0; 3];
    }
    let mut sum = [0.0f32; 3];
    for c in colors {
        for i in 0..3 {
            sum[i] += if c[i].is_finite() { c[i] } else { 0.0 };
        }
    }
    let n = colors.len() as f32;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Extract the 2x2 subpixel block for cell `(col, row)` from a buffer rendered
/// at exactly twice the cell grid resolution.
///
/// `src_width`/`src_height` are the *subpixel* buffer dimensions. Coordinates
/// past the edge clamp to the last available subpixel, so an odd-sized buffer
/// degrades gracefully instead of panicking.
pub fn gather_subpixels(
    pixels: &[[u8; 4]],
    src_width: u32,
    src_height: u32,
    col: u32,
    row: u32,
) -> Subpixels {
    let sample = |x: u32, y: u32| -> [f32; 3] {
        if src_width == 0 || src_height == 0 {
            return [0.0; 3];
        }
        let cx = x.min(src_width - 1);
        let cy = y.min(src_height - 1);
        let idx = (cy as usize) * (src_width as usize) + (cx as usize);
        match pixels.get(idx) {
            Some(p) => [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ],
            None => [0.0; 3],
        }
    };

    let x0 = col * 2;
    let y0 = row * 2;
    [
        sample(x0, y0),
        sample(x0 + 1, y0),
        sample(x0, y0 + 1),
        sample(x0 + 1, y0 + 1),
    ]
}

/// Average a cell's 2x2 block down to a single colour — the downsampling path
/// used by the classic brightness-ramp style, which gains free anti-aliasing
/// from the 2x render target.
pub fn average_subpixels(sub: &Subpixels) -> [f32; 3] {
    average(&[sub[0], sub[1], sub[2], sub[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ink_count(glyph: &Glyph) -> usize {
        glyph.iter().filter(|p| p[3] > 0).count()
    }

    /// Is the given quadrant of the rendered glyph filled?
    fn quadrant_filled(glyph: &Glyph, qx: usize, qy: usize) -> bool {
        let size = GLYPH_SIZE as usize;
        let half = size / 2;
        let mut any = false;
        for y in (qy * half)..(qy * half + half) {
            for x in (qx * half)..(qx * half + half) {
                if glyph[y * size + x][3] > 0 {
                    any = true;
                }
            }
        }
        any
    }

    #[test]
    fn char_table_covers_all_sixteen_patterns() {
        assert_eq!(BLOCK_CHARS.len(), BLOCK_COUNT);
        assert_eq!(BLOCK_CHARS[0], ' ');
        assert_eq!(BLOCK_CHARS[0b1111], '█');
        assert_eq!(BLOCK_CHARS[0b0011], '▀', "bits 0|1 are the upper half");
        assert_eq!(BLOCK_CHARS[0b1100], '▄', "bits 2|3 are the lower half");
        assert_eq!(BLOCK_CHARS[0b0101], '▌', "bits 0|2 are the left half");
        assert_eq!(BLOCK_CHARS[0b1010], '▐', "bits 1|3 are the right half");
        // Every entry must be distinct.
        for i in 0..BLOCK_COUNT {
            for j in (i + 1)..BLOCK_COUNT {
                assert_ne!(BLOCK_CHARS[i], BLOCK_CHARS[j], "duplicate char at {i}/{j}");
            }
        }
    }

    #[test]
    fn rendered_quadrants_match_their_bits() {
        for pattern in 0..BLOCK_COUNT as u8 {
            let glyph = render_block(pattern);
            // Quadrant coordinates: (qx, qy) with qy=0 on top.
            assert_eq!(quadrant_filled(&glyph, 0, 0), pattern & UL != 0, "UL of {pattern:#06b}");
            assert_eq!(quadrant_filled(&glyph, 1, 0), pattern & UR != 0, "UR of {pattern:#06b}");
            assert_eq!(quadrant_filled(&glyph, 0, 1), pattern & LL != 0, "LL of {pattern:#06b}");
            assert_eq!(quadrant_filled(&glyph, 1, 1), pattern & LR != 0, "LR of {pattern:#06b}");
        }
    }

    #[test]
    fn empty_and_full_patterns_are_exactly_that() {
        assert_eq!(ink_count(&render_block(0b0000)), 0);
        assert_eq!(ink_count(&render_block(0b1111)), GLYPH_BYTES);
        // Each single quadrant covers exactly a quarter of the cell.
        for bit in [UL, UR, LL, LR] {
            assert_eq!(ink_count(&render_block(bit)), GLYPH_BYTES / 4, "bit {bit:#06b}");
        }
        // Halves cover exactly half.
        for pat in [0b0011u8, 0b1100, 0b0101, 0b1010, 0b1001, 0b0110] {
            assert_eq!(ink_count(&render_block(pat)), GLYPH_BYTES / 2, "pattern {pat:#06b}");
        }
    }

    #[test]
    fn high_bits_are_ignored() {
        assert_eq!(render_block(0b1111_0000), render_block(0));
        assert_eq!(render_block(0b1111_1111), render_block(0b1111));
    }

    #[test]
    fn atlas_length_matches_the_block_count() {
        assert_eq!(build_block_atlas().len(), BLOCK_COUNT * GLYPH_BYTES * 4);
    }

    #[test]
    fn flat_cell_becomes_a_solid_block_carrying_its_tone() {
        let grey = [0.4, 0.4, 0.4];
        let cell = classify(&[grey, grey, grey, grey]);
        assert_eq!(cell.pattern, 0b1111);
        assert!(cell.flat);
        for c in 0..3 {
            assert!((cell.color[c] - 0.4).abs() < 1e-6);
        }
    }

    #[test]
    fn horizontal_edge_splits_into_upper_and_lower_halves() {
        let bright = [1.0, 1.0, 1.0];
        let dark = [0.0, 0.0, 0.0];
        // Top two subpixels bright, bottom two dark.
        let cell = classify(&[bright, bright, dark, dark]);
        assert_eq!(cell.pattern, UL | UR, "expected the upper half");
        assert_eq!(BLOCK_CHARS[cell.pattern as usize], '▀');
        assert!(!cell.flat);
        // Colour comes from the filled (bright) quadrants only, so the edge does
        // not get dimmed by the dark half.
        for c in 0..3 {
            assert!((cell.color[c] - 1.0).abs() < 1e-6, "got {:?}", cell.color);
        }
    }

    #[test]
    fn vertical_edge_splits_into_left_and_right_halves() {
        let bright = [1.0, 1.0, 1.0];
        let dark = [0.0, 0.0, 0.0];
        let cell = classify(&[bright, dark, bright, dark]);
        assert_eq!(cell.pattern, UL | LL);
        assert_eq!(BLOCK_CHARS[cell.pattern as usize], '▌');
    }

    #[test]
    fn single_bright_subpixel_becomes_a_single_quadrant() {
        let bright = [1.0, 1.0, 1.0];
        let dark = [0.05, 0.05, 0.05];
        assert_eq!(classify(&[bright, dark, dark, dark]).pattern, UL);
        assert_eq!(classify(&[dark, bright, dark, dark]).pattern, UR);
        assert_eq!(classify(&[dark, dark, bright, dark]).pattern, LL);
        assert_eq!(classify(&[dark, dark, dark, bright]).pattern, LR);
    }

    #[test]
    fn diagonal_pattern_is_resolved() {
        let bright = [1.0, 1.0, 1.0];
        let dark = [0.0, 0.0, 0.0];
        let cell = classify(&[bright, dark, dark, bright]);
        assert_eq!(cell.pattern, UL | LR);
        assert_eq!(BLOCK_CHARS[cell.pattern as usize], '▚');
    }

    /// The regression this guards: a partially covered cell must supply a colour
    /// for its EMPTY quadrants too. When it did not, every cell straddling a
    /// gradient rendered its uncovered part as the cleared background, which
    /// showed up as black streaks across smooth surfaces like the ground plane.
    #[test]
    fn partial_cells_carry_a_background_from_the_unlit_subpixels() {
        let bright = [0.8, 0.8, 0.8];
        let dim = [0.2, 0.2, 0.2];
        let cell = classify(&[bright, bright, dim, dim]);

        assert_eq!(cell.pattern, UL | UR);
        // Foreground from the lit half...
        for c in cell.color {
            assert!((c - 0.8).abs() < 1e-6, "foreground {:?}", cell.color);
        }
        // ...and background from the unlit half, NOT black.
        for c in cell.background {
            assert!((c - 0.2).abs() < 1e-6, "background {:?}", cell.background);
        }
        assert_ne!(cell.background, [0.0, 0.0, 0.0], "background must not be a black hole");
    }

    #[test]
    fn background_matches_foreground_when_every_quadrant_is_lit() {
        // Flat cell: fully covered, so the background is never sampled. Keeping it
        // equal to the foreground means a stray mask value cannot darken the cell.
        let cell = classify(&[[0.5; 3], [0.5; 3], [0.5; 3], [0.5; 3]]);
        assert_eq!(cell.pattern, 0b1111);
        assert_eq!(cell.color, cell.background);
    }

    #[test]
    fn background_is_finite_for_hostile_input() {
        let nan = [f32::NAN; 3];
        for s in [[nan, [1.0; 3], nan, [0.0; 3]], [nan, nan, nan, nan]] {
            let cell = classify(&s);
            assert!(cell.background.iter().all(|c| c.is_finite()), "{cell:?}");
        }
    }

    #[test]
    fn classify_always_fills_at_least_one_quadrant() {
        // Whatever the input, the brightest subpixel is at or above the midpoint,
        // so an all-empty pattern (which would render an invisible cell) is
        // impossible.
        let samples = [
            [[0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]],
            [[1.0; 3], [0.0; 3], [0.5; 3], [0.25; 3]],
            [[0.01, 0.02, 0.03], [0.0; 3], [0.0; 3], [0.0; 3]],
        ];
        for s in samples {
            assert_ne!(classify(&s).pattern, 0, "pattern must not be empty for {s:?}");
        }
    }

    #[test]
    fn classify_survives_non_finite_input() {
        let nan = [f32::NAN, f32::NAN, f32::NAN];
        let inf = [f32::INFINITY, 0.0, 0.0];
        for s in [
            [nan, nan, nan, nan],
            [inf, [0.0; 3], [0.0; 3], [0.0; 3]],
            [nan, [1.0; 3], inf, [0.5; 3]],
        ] {
            let cell = classify(&s);
            assert!(cell.color.iter().all(|c| c.is_finite()), "colour went non-finite: {cell:?}");
            assert!(cell.pattern < 16);
        }
    }

    #[test]
    fn gather_subpixels_reads_the_right_2x2_block() {
        // 4x2 subpixel buffer = a 2x1 cell grid.
        let px = vec![
            [10u8, 0, 0, 255], [20, 0, 0, 255], [30, 0, 0, 255], [40, 0, 0, 255],
            [50, 0, 0, 255], [60, 0, 0, 255], [70, 0, 0, 255], [80, 0, 0, 255],
        ];
        let cell0 = gather_subpixels(&px, 4, 2, 0, 0);
        assert!((cell0[0][0] - 10.0 / 255.0).abs() < 1e-6, "UL");
        assert!((cell0[1][0] - 20.0 / 255.0).abs() < 1e-6, "UR");
        assert!((cell0[2][0] - 50.0 / 255.0).abs() < 1e-6, "LL");
        assert!((cell0[3][0] - 60.0 / 255.0).abs() < 1e-6, "LR");

        let cell1 = gather_subpixels(&px, 4, 2, 1, 0);
        assert!((cell1[0][0] - 30.0 / 255.0).abs() < 1e-6);
        assert!((cell1[3][0] - 80.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn gather_subpixels_clamps_out_of_range_and_empty_buffers() {
        let px = vec![[255u8, 255, 255, 255]];
        // Asking for a cell past the end clamps to the only subpixel.
        let sub = gather_subpixels(&px, 1, 1, 5, 5);
        for s in sub {
            assert_eq!(s, [1.0, 1.0, 1.0]);
        }
        // An empty buffer yields black rather than panicking.
        let sub = gather_subpixels(&[], 0, 0, 0, 0);
        for s in sub {
            assert_eq!(s, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn average_subpixels_is_the_mean() {
        let sub: Subpixels = [[0.0; 3], [1.0; 3], [0.0; 3], [1.0; 3]];
        let avg = average_subpixels(&sub);
        for c in avg {
            assert!((c - 0.5).abs() < 1e-6);
        }
    }
}
