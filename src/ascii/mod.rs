// ASCII module: glyph atlas, bitmap font, cell grid, dynamic grid layout,
// colour modes, UI overlay.

pub mod blocks;
pub mod box_drawing;
pub mod color;
pub mod font5x7;
pub mod glyph_atlas;
pub mod grid_layout;
pub mod overlay;

pub use glyph_atlas::{build_atlas, glyph_count, GLYPH_SIZE};
#[allow(unused_imports)]
pub use grid_layout::{average_block_color, compute_tiles, SubdivisionPolicy, Tile};
#[allow(unused_imports)]
pub use overlay::{Overlay, OverlayCell, SceneCell};
#[allow(unused_imports)]
pub use color::{luminance, quantize_buffer, ColorMode};

// --- Combined atlas ---
//
// The composite pass samples one texture, so every glyph source has to live in
// the same atlas. Layout, in order:
//   [0 .. glyph_count())                      shading ramp (indices unchanged,
//                                             so brightness->index still works)
//   [FONT_GLYPH_OFFSET .. +CHAR_COUNT)        printable ASCII text font
//   [BLOCK_GLYPH_OFFSET .. +BLOCK_COUNT)      quadrant block elements
//   [BOX_GLYPH_OFFSET .. +BOX_COUNT)          box drawing + arrows

/// Atlas index of the first font glyph (i.e. of `font5x7::FIRST_CHAR`).
pub const FONT_GLYPH_OFFSET: u32 = 14;

/// Atlas index of the first block-element glyph (quadrant pattern 0).
pub const BLOCK_GLYPH_OFFSET: u32 = FONT_GLYPH_OFFSET + font5x7::CHAR_COUNT as u32;

/// Atlas index of the first box-drawing glyph.
pub const BOX_GLYPH_OFFSET: u32 = BLOCK_GLYPH_OFFSET + blocks::BLOCK_COUNT as u32;

/// Total glyphs in the combined atlas.
pub fn combined_glyph_count() -> usize {
    glyph_count() + font5x7::CHAR_COUNT + blocks::BLOCK_COUNT + box_drawing::BOX_COUNT
}

/// Build the combined atlas: shading glyphs, the text font, the block elements,
/// then box drawing — in the same flat RGBA layout `build_atlas()` uses.
pub fn build_combined_atlas() -> Vec<u8> {
    let mut atlas = build_atlas();
    atlas.extend_from_slice(&font5x7::build_font_atlas());
    atlas.extend_from_slice(&blocks::build_block_atlas());
    atlas.extend_from_slice(&box_drawing::build_box_atlas());
    atlas
}

/// Atlas index for a box-drawing character, or `None` if it is not one.
pub fn box_glyph_index(c: char) -> Option<u32> {
    box_drawing::box_index(c).map(|i| BOX_GLYPH_OFFSET + i as u32)
}

/// Atlas index for a 4-bit quadrant pattern.
pub fn block_glyph_index(pattern: u8) -> u32 {
    BLOCK_GLYPH_OFFSET + (pattern & 0b1111) as u32
}

/// Convert the per-glyph atlas layout into the row-major single-channel data a
/// GPU texture upload expects.
///
/// `build_*_atlas()` emits glyphs one after another: all 64 pixels of glyph 0,
/// then all 64 of glyph 1, and so on. A texture of `count * GLYPH_SIZE` by
/// `GLYPH_SIZE` pixels is instead addressed row by row: row `y` holds row `y` of
/// EVERY glyph, side by side.
///
/// Uploading the per-glyph layout directly therefore scrambles the atlas — every
/// sampled cell picks up fragments of unrelated glyphs, which is what made the
/// output look like arbitrary noise and the HUD unreadable. This function does
/// the transpose and takes the red channel (coverage is stored in every channel;
/// the texture is R8Unorm).
pub fn atlas_to_row_major_r8(atlas_rgba: &[u8], glyph_count: usize) -> Vec<u8> {
    let size = GLYPH_SIZE as usize;
    let width = glyph_count * size;
    let mut out = vec![0u8; width * size];

    for glyph in 0..glyph_count {
        for y in 0..size {
            for x in 0..size {
                // Source: glyph-major, 4 bytes per pixel.
                let src = ((glyph * size * size) + y * size + x) * 4;
                // Destination: row-major across the whole atlas strip.
                let dst = y * width + glyph * size + x;
                if let (Some(&v), Some(slot)) = (atlas_rgba.get(src), out.get_mut(dst)) {
                    *slot = v;
                }
            }
        }
    }

    out
}

/// Atlas index for a text character, or `None` when the font does not cover it.
pub fn text_glyph_index(c: char) -> Option<u32> {
    font5x7::glyph_index(c).map(|i| FONT_GLYPH_OFFSET + i as u32)
}

/// `char -> glyph index` mapping for [`Overlay`], usable as its `GlyphMapFn`.
///
/// Resolves printable ASCII through the text font and box-drawing/arrow
/// characters through their own section, so UI code can write '┌' and '▲'
/// literally. Anything else falls back to the space glyph rather than to a
/// shading block, so an unexpected character leaves a gap instead of a bright
/// artefact in the middle of a panel.
pub fn overlay_glyph_of(c: char) -> u32 {
    if let Some(index) = text_glyph_index(c) {
        return index;
    }
    if let Some(index) = box_glyph_index(c) {
        return index;
    }
    glyph_atlas::SPACE_INDEX
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascii::glyph_atlas::GLYPH_BYTES;

    #[test]
    fn font_offset_matches_the_shading_glyph_count() {
        // If a shading glyph is ever added, this must move with it — otherwise
        // every text glyph silently shifts.
        assert_eq!(FONT_GLYPH_OFFSET as usize, glyph_count());
    }

    #[test]
    fn combined_atlas_has_all_sections() {
        let atlas = build_combined_atlas();
        assert_eq!(
            combined_glyph_count(),
            glyph_count() + font5x7::CHAR_COUNT + blocks::BLOCK_COUNT + box_drawing::BOX_COUNT
        );
        assert_eq!(atlas.len(), combined_glyph_count() * GLYPH_BYTES * 4);

        // The shading section must be byte-identical to the original atlas, so
        // existing brightness->index mappings are unaffected.
        let shading = build_atlas();
        assert_eq!(&atlas[..shading.len()], &shading[..]);
    }

    #[test]
    fn block_indices_land_inside_the_block_section() {
        assert_eq!(BLOCK_GLYPH_OFFSET as usize, glyph_count() + font5x7::CHAR_COUNT);
        assert_eq!(block_glyph_index(0), BLOCK_GLYPH_OFFSET);
        assert_eq!(block_glyph_index(0b1111), BLOCK_GLYPH_OFFSET + 15);
        // High bits are masked off, so a stray value cannot index past the section.
        assert_eq!(block_glyph_index(0xFF), BLOCK_GLYPH_OFFSET + 15);
        assert!((block_glyph_index(0b1111) as usize) < combined_glyph_count());
    }

    #[test]
    fn the_block_and_box_sections_match_their_atlases() {
        let atlas = build_combined_atlas();

        let blocks_bytes = blocks::build_block_atlas();
        let block_offset = (BLOCK_GLYPH_OFFSET as usize) * GLYPH_BYTES * 4;
        assert_eq!(
            &atlas[block_offset..block_offset + blocks_bytes.len()],
            &blocks_bytes[..]
        );

        let box_bytes = box_drawing::build_box_atlas();
        let box_offset = (BOX_GLYPH_OFFSET as usize) * GLYPH_BYTES * 4;
        assert_eq!(&atlas[box_offset..], &box_bytes[..], "box section must end the atlas");
    }

    #[test]
    fn box_glyph_indices_land_inside_the_box_section() {
        assert_eq!(
            BOX_GLYPH_OFFSET as usize,
            glyph_count() + font5x7::CHAR_COUNT + blocks::BLOCK_COUNT
        );
        let corner = box_glyph_index('┌').expect("box chars must resolve");
        assert!(corner >= BOX_GLYPH_OFFSET);
        assert!((corner as usize) < combined_glyph_count());
        assert_eq!(box_glyph_index('A'), None, "letters belong to the font section");

        // The overlay map must reach both sections, since UI code writes text and
        // frame characters side by side.
        assert_eq!(overlay_glyph_of('┌'), corner);
        assert_eq!(overlay_glyph_of('A'), text_glyph_index('A').unwrap());
        assert_eq!(overlay_glyph_of('▲'), box_glyph_index('▲').unwrap());
    }

    #[test]
    fn text_glyph_index_lands_inside_the_font_section() {
        let a = text_glyph_index('A').expect("'A' must be covered");
        assert!(a >= FONT_GLYPH_OFFSET);
        assert!((a as usize) < combined_glyph_count());

        assert_eq!(text_glyph_index(' '), Some(FONT_GLYPH_OFFSET));
        assert_eq!(text_glyph_index('~'), Some(FONT_GLYPH_OFFSET + 94));
        assert_eq!(text_glyph_index('\n'), None);
    }

    #[test]
    fn overlay_glyph_of_falls_back_to_space_for_unsupported_chars() {
        assert_eq!(overlay_glyph_of('A'), text_glyph_index('A').unwrap());
        assert_eq!(overlay_glyph_of('Я'), glyph_atlas::SPACE_INDEX);
        assert_eq!(overlay_glyph_of('\n'), glyph_atlas::SPACE_INDEX);
    }

    /// The bug this guards against shipped from the very first commit: the atlas
    /// was uploaded glyph-major into a row-major texture, so every sampled cell
    /// showed slices of unrelated glyphs. Nothing caught it because the atlas
    /// *content* was correct — only its arrangement for the GPU was wrong.
    #[test]
    fn row_major_conversion_places_each_glyph_row_side_by_side() {
        let atlas = build_combined_atlas();
        let count = combined_glyph_count();
        let rows = atlas_to_row_major_r8(&atlas, count);
        let size = GLYPH_SIZE as usize;
        let width = count * size;

        assert_eq!(rows.len(), width * size, "one byte per texel");

        // Check a glyph whose shape is unambiguous: the solid block must be fully
        // set on every row, at its own horizontal slot.
        let solid = block_glyph_index(0b1111) as usize;
        for y in 0..size {
            for x in 0..size {
                assert_eq!(
                    rows[y * width + solid * size + x],
                    255,
                    "solid block hole at ({x},{y})"
                );
            }
        }

        // The upper-half block must be set on the top four rows and clear below —
        // this is what distinguishes a correct transpose from a plausible-looking
        // but wrong one.
        let upper = block_glyph_index(blocks::UL | blocks::UR) as usize;
        for y in 0..size {
            let expected = if y < size / 2 { 255 } else { 0 };
            for x in 0..size {
                assert_eq!(
                    rows[y * width + upper * size + x],
                    expected,
                    "upper-half block wrong at ({x},{y})"
                );
            }
        }

        // The space glyph stays empty, and a text glyph keeps its own bitmap.
        let space = glyph_atlas::SPACE_INDEX as usize;
        for y in 0..size {
            for x in 0..size {
                assert_eq!(rows[y * width + space * size + x], 0);
            }
        }
        let a = text_glyph_index('A').unwrap() as usize;
        let expected_a = font5x7::render_glyph('A').unwrap();
        for y in 0..size {
            for x in 0..size {
                assert_eq!(
                    rows[y * width + a * size + x],
                    expected_a[y * size + x][0],
                    "glyph 'A' differs at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn row_major_conversion_tolerates_a_short_input() {
        // Must not panic if the caller passes a mismatched count.
        let out = atlas_to_row_major_r8(&[255, 255, 255, 255], 4);
        assert_eq!(out.len(), 4 * 8 * 8);
    }

    #[test]
    fn the_font_section_bytes_match_the_font_atlas() {
        let atlas = build_combined_atlas();
        let font = font5x7::build_font_atlas();
        let offset = glyph_count() * GLYPH_BYTES * 4;
        assert_eq!(&atlas[offset..offset + font.len()], &font[..]);
    }
}

