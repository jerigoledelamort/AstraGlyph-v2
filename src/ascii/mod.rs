// ASCII module: glyph atlas, bitmap font, cell grid, dynamic grid layout,
// colour modes, UI overlay.

pub mod blocks;
pub mod cell_grid;
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

/// Atlas index of the first font glyph (i.e. of `font5x7::FIRST_CHAR`).
pub const FONT_GLYPH_OFFSET: u32 = 14;

/// Atlas index of the first block-element glyph (quadrant pattern 0).
pub const BLOCK_GLYPH_OFFSET: u32 = FONT_GLYPH_OFFSET + font5x7::CHAR_COUNT as u32;

/// Total glyphs in the combined atlas.
pub fn combined_glyph_count() -> usize {
    glyph_count() + font5x7::CHAR_COUNT + blocks::BLOCK_COUNT
}

/// Build the combined atlas: shading glyphs, then the text font, then the block
/// elements, in the same flat RGBA layout `build_atlas()` uses.
pub fn build_combined_atlas() -> Vec<u8> {
    let mut atlas = build_atlas();
    atlas.extend_from_slice(&font5x7::build_font_atlas());
    atlas.extend_from_slice(&blocks::build_block_atlas());
    atlas
}

/// Atlas index for a 4-bit quadrant pattern.
pub fn block_glyph_index(pattern: u8) -> u32 {
    BLOCK_GLYPH_OFFSET + (pattern & 0b1111) as u32
}

/// Atlas index for a text character, or `None` when the font does not cover it.
pub fn text_glyph_index(c: char) -> Option<u32> {
    font5x7::glyph_index(c).map(|i| FONT_GLYPH_OFFSET + i as u32)
}

/// `char -> glyph index` mapping for [`Overlay`], usable as its `GlyphMapFn`.
///
/// Unsupported characters fall back to the space glyph rather than to a shading
/// block, so an unexpected character leaves a gap instead of a bright artefact.
pub fn overlay_glyph_of(c: char) -> u32 {
    text_glyph_index(c).unwrap_or(glyph_atlas::SPACE_INDEX)
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
    fn combined_atlas_has_all_three_sections() {
        let atlas = build_combined_atlas();
        assert_eq!(
            combined_glyph_count(),
            glyph_count() + font5x7::CHAR_COUNT + blocks::BLOCK_COUNT
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
    fn the_block_section_bytes_match_the_block_atlas() {
        let atlas = build_combined_atlas();
        let blocks_bytes = blocks::build_block_atlas();
        let offset = (BLOCK_GLYPH_OFFSET as usize) * GLYPH_BYTES * 4;
        assert_eq!(&atlas[offset..], &blocks_bytes[..]);
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

    #[test]
    fn the_font_section_bytes_match_the_font_atlas() {
        let atlas = build_combined_atlas();
        let font = font5x7::build_font_atlas();
        let offset = glyph_count() * GLYPH_BYTES * 4;
        assert_eq!(&atlas[offset..offset + font.len()], &font[..]);
    }
}
