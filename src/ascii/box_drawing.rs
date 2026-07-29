// Box Drawing characters (U+2500..U+257F) plus a few geometric arrows.
//
// Why these and not the CJK/emoji set the roadmap also mentions: an 8x8 cell
// physically cannot hold a legible CJK ideograph — those need 16x16 at minimum,
// which would mean a second, larger glyph cell and a different atlas geometry.
// Box drawing, by contrast, fits 8x8 perfectly (it is line art on a grid) and is
// immediately useful: menu and console panels need real frames, and ASCII
// substitutes like '+' and '-' leave visible gaps at the corners.
//
// Every glyph here is generated from line SEGMENTS rather than a hand-typed
// bitmap. Line art is exactly the case where that pays off: the segments encode
// the intent ("a horizontal stroke from the centre to the right edge"), so
// neighbouring pieces are guaranteed to meet, which is the whole point of box
// drawing. A typo in a hand-drawn bitmap would show up as a one-pixel gap in a
// frame corner.
//
// Stroke geometry in the 8x8 cell:
//   single lines run along index 4 (the centre axis)
//   double lines run along indices 3 and 5, straddling that axis

use crate::ascii::glyph_atlas::{Glyph, GLYPH_BYTES, GLYPH_SIZE};

/// Centre axis for single-width strokes.
const MID: usize = 4;
/// The two axes used by double-width strokes.
const LO: usize = 3;
const HI: usize = 5;

/// Characters provided by this module, in atlas order.
pub const BOX_CHARS: &[char] = &[
    // Single line.
    '─', '│', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼',
    // Double line.
    '═', '║', '╔', '╗', '╚', '╝', '╠', '╣', '╦', '╩', '╬',
    // Geometric arrows, for menu markers and scroll indicators.
    '▲', '▼', '◄', '►',
];

/// Number of glyphs in this section.
pub const BOX_COUNT: usize = 26;

/// One stroke: a horizontal or vertical run of pixels.
#[derive(Clone, Copy)]
enum Segment {
    /// Row `y`, columns `x0..=x1`.
    Horizontal { y: usize, x0: usize, x1: usize },
    /// Column `x`, rows `y0..=y1`.
    Vertical { x: usize, y0: usize, y1: usize },
}

const LAST: usize = GLYPH_SIZE as usize - 1;

/// Full-width horizontal stroke on row `y`.
const fn h_full(y: usize) -> Segment {
    Segment::Horizontal { y, x0: 0, x1: LAST }
}
/// Horizontal stroke from `x0` to the right edge.
const fn h_right(y: usize, x0: usize) -> Segment {
    Segment::Horizontal { y, x0, x1: LAST }
}
/// Horizontal stroke from the left edge to `x1`.
const fn h_left(y: usize, x1: usize) -> Segment {
    Segment::Horizontal { y, x0: 0, x1 }
}
/// Full-height vertical stroke on column `x`.
const fn v_full(x: usize) -> Segment {
    Segment::Vertical { x, y0: 0, y1: LAST }
}
/// Vertical stroke from `y0` to the bottom edge.
const fn v_down(x: usize, y0: usize) -> Segment {
    Segment::Vertical { x, y0, y1: LAST }
}
/// Vertical stroke from the top edge to `y1`.
const fn v_up(x: usize, y1: usize) -> Segment {
    Segment::Vertical { x, y0: 0, y1 }
}

/// Segments for the glyph at `index` (matching [`BOX_CHARS`]).
///
/// Returns an owned list: this runs once per glyph while building the atlas at
/// startup, so a handful of tiny allocations buys a far more readable table than
/// 22 separate `const` arrays would.
fn segments(index: usize) -> Vec<Segment> {
    match index {
        // '─' '│'
        0 => vec![h_full(MID)],
        1 => vec![v_full(MID)],
        // '┌' '┐' '└' '┘' — quarter turns meeting at the centre.
        2 => vec![h_right(MID, MID), v_down(MID, MID)],
        3 => vec![h_left(MID, MID), v_down(MID, MID)],
        4 => vec![h_right(MID, MID), v_up(MID, MID)],
        5 => vec![h_left(MID, MID), v_up(MID, MID)],
        // '├' '┤' '┬' '┴' '┼' — tees and the cross.
        6 => vec![v_full(MID), h_right(MID, MID)],
        7 => vec![v_full(MID), h_left(MID, MID)],
        8 => vec![h_full(MID), v_down(MID, MID)],
        9 => vec![h_full(MID), v_up(MID, MID)],
        10 => vec![h_full(MID), v_full(MID)],

        // '═' '║'
        11 => vec![h_full(LO), h_full(HI)],
        12 => vec![v_full(LO), v_full(HI)],
        // '╔' — the outer stroke turns early, the inner one late, so the corner
        // closes instead of leaving a notch.
        13 => vec![h_right(LO, LO), h_right(HI, HI), v_down(LO, LO), v_down(HI, HI)],
        // '╗'
        14 => vec![h_left(LO, HI), h_left(HI, LO), v_down(HI, LO), v_down(LO, HI)],
        // '╚'
        15 => vec![h_right(HI, LO), h_right(LO, HI), v_up(LO, HI), v_up(HI, LO)],
        // '╝'
        16 => vec![h_left(HI, HI), h_left(LO, LO), v_up(HI, HI), v_up(LO, LO)],
        // '╠' '╣' '╦' '╩' '╬'
        17 => vec![v_full(LO), v_full(HI), h_right(LO, HI), h_right(HI, HI)],
        18 => vec![v_full(LO), v_full(HI), h_left(LO, LO), h_left(HI, LO)],
        19 => vec![h_full(LO), h_full(HI), v_down(LO, HI), v_down(HI, HI)],
        20 => vec![h_full(LO), h_full(HI), v_up(LO, LO), v_up(HI, LO)],
        21 => vec![h_full(LO), h_full(HI), v_full(LO), v_full(HI)],

        _ => Vec::new(),
    }
}

/// Render the glyph at `index` into the atlas's 8x8 format.
pub fn render_box_glyph(index: usize) -> Glyph {
    let mut glyph: Glyph = [[0u8; 4]; GLYPH_BYTES];
    let size = GLYPH_SIZE as usize;

    // Arrows are solid triangles, not strokes, so they are filled directly.
    match index {
        22 => return triangle_up(),
        23 => return triangle_down(),
        24 => return triangle_left(),
        25 => return triangle_right(),
        _ => {}
    }

    for seg in &segments(index) {
        match *seg {
            Segment::Horizontal { y, x0, x1 } => {
                if y >= size {
                    continue;
                }
                for x in x0..=x1.min(LAST) {
                    glyph[y * size + x] = [255, 255, 255, 255];
                }
            }
            Segment::Vertical { x, y0, y1 } => {
                if x >= size {
                    continue;
                }
                for y in y0..=y1.min(LAST) {
                    glyph[y * size + x] = [255, 255, 255, 255];
                }
            }
        }
    }

    glyph
}

fn set(glyph: &mut Glyph, x: usize, y: usize) {
    let size = GLYPH_SIZE as usize;
    if x < size && y < size {
        glyph[y * size + x] = [255, 255, 255, 255];
    }
}

/// '▲' — a triangle widening toward the bottom.
fn triangle_up() -> Glyph {
    let mut g: Glyph = [[0u8; 4]; GLYPH_BYTES];
    for row in 0..4usize {
        // Row 0 is a single pixel pair at the centre; each row widens by one.
        let half = row + 1;
        for x in (MID - half)..=(MID + half - 1) {
            set(&mut g, x, row + 2);
        }
    }
    g
}

/// '▼' — the same triangle mirrored vertically.
fn triangle_down() -> Glyph {
    let mut g: Glyph = [[0u8; 4]; GLYPH_BYTES];
    for row in 0..4usize {
        let half = 4 - row;
        for x in (MID - half)..=(MID + half - 1) {
            set(&mut g, x, row + 2);
        }
    }
    g
}

/// '◄' — a triangle widening toward the right.
fn triangle_left() -> Glyph {
    let mut g: Glyph = [[0u8; 4]; GLYPH_BYTES];
    for col in 0..4usize {
        let half = col + 1;
        for y in (MID - half)..=(MID + half - 1) {
            set(&mut g, col + 2, y);
        }
    }
    g
}

/// '►' — the same triangle mirrored horizontally.
fn triangle_right() -> Glyph {
    let mut g: Glyph = [[0u8; 4]; GLYPH_BYTES];
    for col in 0..4usize {
        let half = 4 - col;
        for y in (MID - half)..=(MID + half - 1) {
            set(&mut g, col + 2, y);
        }
    }
    g
}

/// Index of `c` within this section, or `None` if it is not a box glyph.
pub fn box_index(c: char) -> Option<usize> {
    BOX_CHARS.iter().position(|&candidate| candidate == c)
}

/// Build the box-drawing section as flat RGBA bytes, same layout as
/// `glyph_atlas::build_atlas()`.
pub fn build_box_atlas() -> Vec<u8> {
    let mut atlas = Vec::with_capacity(BOX_COUNT * GLYPH_BYTES * 4);
    for index in 0..BOX_COUNT {
        let glyph = render_box_glyph(index);
        for pixel in &glyph {
            atlas.extend_from_slice(pixel);
        }
    }
    atlas
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(glyph: &Glyph, x: usize, y: usize) -> bool {
        glyph[y * GLYPH_SIZE as usize + x][3] > 0
    }

    fn ink(glyph: &Glyph) -> usize {
        glyph.iter().filter(|p| p[3] > 0).count()
    }

    #[test]
    fn char_table_length_matches_the_count() {
        assert_eq!(BOX_CHARS.len(), BOX_COUNT);
        for i in 0..BOX_CHARS.len() {
            for j in (i + 1)..BOX_CHARS.len() {
                assert_ne!(BOX_CHARS[i], BOX_CHARS[j], "duplicate char at {i}/{j}");
            }
        }
    }

    #[test]
    fn box_index_finds_known_chars_and_rejects_others() {
        assert_eq!(box_index('─'), Some(0));
        assert_eq!(box_index('┼'), Some(10));
        assert_eq!(box_index('╬'), Some(21));
        assert_eq!(box_index('►'), Some(25));
        assert_eq!(box_index('A'), None);
        assert_eq!(box_index(' '), None);
    }

    #[test]
    fn every_glyph_has_ink() {
        for i in 0..BOX_COUNT {
            assert!(ink(&render_box_glyph(i)) > 0, "glyph {i} ({}) is blank", BOX_CHARS[i]);
        }
    }

    #[test]
    fn horizontal_and_vertical_lines_span_the_cell() {
        let h = render_box_glyph(0); // '─'
        for x in 0..8 {
            assert!(lit(&h, x, MID), "horizontal line missing at x={x}");
        }
        assert_eq!(ink(&h), 8, "a single line must be exactly one pixel thick");

        let v = render_box_glyph(1); // '│'
        for y in 0..8 {
            assert!(lit(&v, MID, y), "vertical line missing at y={y}");
        }
        assert_eq!(ink(&v), 8);
    }

    /// The reason segments are used instead of hand-drawn bitmaps: adjacent
    /// pieces of a frame have to meet exactly at the centre axis.
    #[test]
    fn corners_touch_the_centre_and_only_their_own_two_edges() {
        // '┌' opens right and down.
        let tl = render_box_glyph(2);
        assert!(lit(&tl, MID, MID), "corner must cover the centre");
        assert!(lit(&tl, 7, MID), "'┌' must reach the right edge");
        assert!(lit(&tl, MID, 7), "'┌' must reach the bottom edge");
        assert!(!lit(&tl, 0, MID), "'┌' must not reach the left edge");
        assert!(!lit(&tl, MID, 0), "'┌' must not reach the top edge");

        // '┘' opens left and up — the opposite corner.
        let br = render_box_glyph(5);
        assert!(lit(&br, 0, MID));
        assert!(lit(&br, MID, 0));
        assert!(!lit(&br, 7, MID));
        assert!(!lit(&br, MID, 7));
    }

    #[test]
    fn tees_keep_their_through_line_and_one_stub() {
        // '├': full vertical, stub to the right.
        let t = render_box_glyph(6);
        for y in 0..8 {
            assert!(lit(&t, MID, y), "'├' vertical broken at y={y}");
        }
        assert!(lit(&t, 7, MID), "'├' stub must reach the right edge");
        assert!(!lit(&t, 0, MID), "'├' must not reach the left edge");

        // '┬': full horizontal, stub downward.
        let d = render_box_glyph(8);
        for x in 0..8 {
            assert!(lit(&d, x, MID));
        }
        assert!(lit(&d, MID, 7));
        assert!(!lit(&d, MID, 0));
    }

    #[test]
    fn cross_spans_both_axes_fully() {
        let cross = render_box_glyph(10); // '┼'
        for i in 0..8 {
            assert!(lit(&cross, i, MID));
            assert!(lit(&cross, MID, i));
        }
    }

    #[test]
    fn double_lines_use_two_distinct_axes() {
        let h = render_box_glyph(11); // '═'
        for x in 0..8 {
            assert!(lit(&h, x, LO), "upper stroke missing at x={x}");
            assert!(lit(&h, x, HI), "lower stroke missing at x={x}");
            assert!(!lit(&h, x, MID), "the gap between the strokes must stay clear");
        }
        assert_eq!(ink(&h), 16);

        let v = render_box_glyph(12); // '║'
        for y in 0..8 {
            assert!(lit(&v, LO, y));
            assert!(lit(&v, HI, y));
            assert!(!lit(&v, MID, y));
        }
    }

    #[test]
    fn double_corner_closes_without_a_notch() {
        // '╔': the outer strokes must form a continuous corner, and the inner
        // ones must meet it — a notch here is the classic double-line artefact.
        let g = render_box_glyph(13);
        assert!(lit(&g, LO, LO), "outer corner pixel missing");
        assert!(lit(&g, 7, LO), "outer stroke must reach the right edge");
        assert!(lit(&g, LO, 7), "outer stroke must reach the bottom edge");
        assert!(lit(&g, HI, HI), "inner corner pixel missing");
        assert!(lit(&g, 7, HI), "inner stroke must reach the right edge");
        assert!(lit(&g, HI, 7), "inner stroke must reach the bottom edge");
        // Nothing above or to the left of the outer corner.
        assert!(!lit(&g, 0, LO));
        assert!(!lit(&g, LO, 0));
    }

    #[test]
    fn double_cross_covers_all_four_axes() {
        let g = render_box_glyph(21); // '╬'
        for i in 0..8 {
            assert!(lit(&g, i, LO));
            assert!(lit(&g, i, HI));
            assert!(lit(&g, LO, i));
            assert!(lit(&g, HI, i));
        }
    }

    #[test]
    fn arrows_point_the_right_way() {
        // '▲' is narrow at the top, wide at the bottom.
        let up = render_box_glyph(22);
        let top_width = (0..8).filter(|&x| lit(&up, x, 2)).count();
        let bottom_width = (0..8).filter(|&x| lit(&up, x, 5)).count();
        assert!(bottom_width > top_width, "'▲' should widen downward: {top_width} -> {bottom_width}");

        // '▼' is the mirror.
        let down = render_box_glyph(23);
        let top_width = (0..8).filter(|&x| lit(&down, x, 2)).count();
        let bottom_width = (0..8).filter(|&x| lit(&down, x, 5)).count();
        assert!(top_width > bottom_width, "'▼' should narrow downward");

        // '◄' widens to the right, '►' to the left.
        let left = render_box_glyph(24);
        let left_col = (0..8).filter(|&y| lit(&left, 2, y)).count();
        let right_col = (0..8).filter(|&y| lit(&left, 5, y)).count();
        assert!(right_col > left_col, "'◄' should widen rightward");

        let right = render_box_glyph(25);
        let left_col = (0..8).filter(|&y| lit(&right, 2, y)).count();
        let right_col = (0..8).filter(|&y| lit(&right, 5, y)).count();
        assert!(left_col > right_col, "'►' should widen leftward");
    }

    #[test]
    fn out_of_range_index_renders_blank_without_panicking() {
        assert_eq!(ink(&render_box_glyph(999)), 0);
    }

    #[test]
    fn atlas_length_matches_the_glyph_count() {
        assert_eq!(build_box_atlas().len(), BOX_COUNT * GLYPH_BYTES * 4);
    }
}
