// Hand-authored 5x7 bitmap font covering printable ASCII (0x20..=0x7E),
// rasterized into the engine's 8x8 glyph format.
//
// Why a hand-coded font: the procedural atlas in glyph_atlas.rs holds only
// shading glyphs, so nothing in the engine could draw text — which blocks the
// HUD, menus and debug console. ROADMAP lists TTF loading as a "fallback if no
// procedural font", so the procedural path is the primary one and this table is
// it; a TTF parser is an optional addition on top.
//
// Bitmap encoding: one byte per row, 7 rows per glyph, 5 significant bits.
// Bit 4 (0b10000) is the LEFTMOST column and bit 0 (0b00001) the rightmost, so a
// binary literal in the source reads left-to-right exactly as the glyph looks.
// Every row byte must therefore be <= 0b11111; a test enforces that over the
// whole table.

use crate::ascii::glyph_atlas::{Glyph, GLYPH_BYTES, GLYPH_SIZE};

/// First character covered by the font (space).
pub const FIRST_CHAR: char = ' ';
/// Last character covered by the font (tilde).
pub const LAST_CHAR: char = '~';
/// Number of glyphs in the table.
pub const CHAR_COUNT: usize = 95;
/// Glyph cell width in pixels.
pub const FONT_WIDTH: usize = 5;
/// Glyph cell height in pixels.
pub const FONT_HEIGHT: usize = 7;

/// Left padding when placing the 5x7 pattern into the 8x8 cell.
const PAD_X: usize = 1;
/// Top padding when placing the 5x7 pattern into the 8x8 cell.
const PAD_Y: usize = 0;

/// The font, in ASCII order starting at `FIRST_CHAR`.
#[rustfmt::skip]
const FONT: [[u8; FONT_HEIGHT]; CHAR_COUNT] = [
    // ' '
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // '!'
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
    // '"'
    [0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // '#'
    [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
    // '$'
    [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100],
    // '%'
    [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
    // '&'
    [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
    // '\''
    [0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // '('
    [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    // ')'
    [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
    // '*'
    [0b00000, 0b00100, 0b10101, 0b01110, 0b10101, 0b00100, 0b00000],
    // '+'
    [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
    // ','
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00100, 0b01000],
    // '-'
    [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
    // '.'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
    // '/'
    [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
    // '0'
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    // '1'
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // '2'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    // '3'
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    // '4'
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    // '5'
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    // '6'
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    // '7'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    // '8'
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    // '9'
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    // ':'
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
    // ';'
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b00100, 0b01000],
    // '<'
    [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
    // '='
    [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
    // '>'
    [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
    // '?'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100],
    // '@'
    [0b01110, 0b10001, 0b00001, 0b01101, 0b10101, 0b10101, 0b01110],
    // 'A'
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    // 'B'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    // 'C'
    [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    // 'D'
    [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
    // 'E'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    // 'F'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    // 'G'
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
    // 'H'
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    // 'I'
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'J'
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
    // 'K'
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    // 'L'
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    // 'M'
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    // 'N'
    [0b10001, 0b11001, 0b11001, 0b10101, 0b10011, 0b10011, 0b10001],
    // 'O'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'P'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    // 'Q'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10011, 0b01101],
    // 'R'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    // 'S'
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    // 'T'
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    // 'U'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'V'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    // 'W'
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
    // 'X'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    // 'Y'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    // 'Z'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    // '['
    [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
    // '\\'
    [0b10000, 0b01000, 0b01000, 0b00100, 0b00010, 0b00010, 0b00001],
    // ']'
    [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
    // '^'
    [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
    // '_'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
    // '`'
    [0b01000, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // 'a'
    [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
    // 'b'
    [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
    // 'c'
    [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
    // 'd'
    [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111],
    // 'e'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
    // 'f'
    [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
    // 'g' (descender)
    [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
    // 'h'
    [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
    // 'i'
    [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'j' (descender)
    [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
    // 'k'
    [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
    // 'l'
    [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'm'
    [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
    // 'n'
    [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
    // 'o'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'p' (descender)
    [0b00000, 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000],
    // 'q' (descender)
    [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001],
    // 'r'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b10000, 0b10000, 0b10000],
    // 's'
    [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
    // 't'
    [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
    // 'u'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10001, 0b01111],
    // 'v'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    // 'w'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
    // 'x'
    [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
    // 'y' (descender)
    [0b00000, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
    // 'z'
    [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
    // '{'
    [0b00110, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00110],
    // '|'
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    // '}'
    [0b01100, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b01100],
    // '~'
    [0b00000, 0b00000, 0b01001, 0b10101, 0b10010, 0b00000, 0b00000],
];

/// Index of `c` in the font table, or `None` if the font does not cover it.
pub fn glyph_index(c: char) -> Option<usize> {
    if c < FIRST_CHAR || c > LAST_CHAR {
        return None;
    }
    Some(c as usize - FIRST_CHAR as usize)
}

/// The raw 5x7 bitmap rows for `c` (bit 4 = leftmost column).
pub fn glyph_rows(c: char) -> Option<[u8; FONT_HEIGHT]> {
    glyph_index(c).map(|i| FONT[i])
}

/// Render `c` into the atlas's 8x8 `Glyph` format.
///
/// The 5x7 pattern is placed with `PAD_X` pixels of left padding and `PAD_Y` of
/// top padding, leaving the rightmost two columns and the bottom row clear —
/// that inter-character gap is what keeps adjacent cells readable as text.
pub fn render_glyph(c: char) -> Option<Glyph> {
    let rows = glyph_rows(c)?;
    let mut glyph: Glyph = [[0u8; 4]; GLYPH_BYTES];
    let size = GLYPH_SIZE as usize;

    for (ry, row) in rows.iter().enumerate() {
        let y = ry + PAD_Y;
        if y >= size {
            break;
        }
        for cx in 0..FONT_WIDTH {
            let x = cx + PAD_X;
            if x >= size {
                break;
            }
            // Bit 4 is the leftmost column.
            let bit = 1u8 << (FONT_WIDTH - 1 - cx);
            if row & bit != 0 {
                glyph[y * size + x] = [255, 255, 255, 255];
            }
        }
    }

    Some(glyph)
}

/// Build the whole font as flat RGBA bytes, matching the layout of
/// `glyph_atlas::build_atlas()`: `CHAR_COUNT` glyphs of `GLYPH_SIZE` squared.
pub fn build_font_atlas() -> Vec<u8> {
    let mut atlas = Vec::with_capacity(CHAR_COUNT * GLYPH_BYTES * 4);
    for i in 0..CHAR_COUNT {
        let c = char::from_u32(FIRST_CHAR as u32 + i as u32).unwrap_or(' ');
        let glyph = render_glyph(c).unwrap_or([[0u8; 4]; GLYPH_BYTES]);
        for pixel in &glyph {
            atlas.extend_from_slice(pixel);
        }
    }
    atlas
}

/// Width of `text` in cells. Characters the font does not cover still occupy a
/// cell (they render blank), so the width equals the character count — layout
/// stays predictable regardless of content.
pub fn text_width(text: &str) -> usize {
    text.chars().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Does the rendered glyph have any ink on this 8x8 row?
    fn row_has_ink(glyph: &Glyph, y: usize) -> bool {
        let size = GLYPH_SIZE as usize;
        (0..size).any(|x| glyph[y * size + x][3] > 0)
    }

    fn ink_count(glyph: &Glyph) -> usize {
        glyph.iter().filter(|p| p[3] > 0).count()
    }

    #[test]
    fn table_length_matches_the_character_range() {
        assert_eq!(FONT.len(), CHAR_COUNT);
        assert_eq!(LAST_CHAR as usize - FIRST_CHAR as usize + 1, CHAR_COUNT);
    }

    #[test]
    fn every_row_fits_in_five_bits() {
        for (i, rows) in FONT.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                assert!(
                    *row <= 0b11111,
                    "glyph {i} (char {:?}) row {r} = {row:#07b} exceeds 5 bits",
                    char::from_u32(FIRST_CHAR as u32 + i as u32).unwrap()
                );
            }
        }
    }

    #[test]
    fn covered_range_is_contiguous_and_ordered() {
        assert_eq!(glyph_index(' '), Some(0));
        assert_eq!(glyph_index('A'), Some('A' as usize - 0x20));
        assert_eq!(glyph_index('~'), Some(CHAR_COUNT - 1));
        for i in 0..CHAR_COUNT {
            let c = char::from_u32(0x20 + i as u32).unwrap();
            assert_eq!(glyph_index(c), Some(i), "index mismatch for {c:?}");
            assert!(glyph_rows(c).is_some());
            assert!(render_glyph(c).is_some());
        }
    }

    #[test]
    fn uncovered_characters_return_none_without_panicking() {
        for c in ['\n', '\t', '\u{0}', '\u{7F}', 'Я', '€', '\u{10FFFF}'] {
            assert_eq!(glyph_index(c), None, "{c:?} should be uncovered");
            assert_eq!(glyph_rows(c), None);
            assert!(render_glyph(c).is_none());
        }
    }

    #[test]
    fn space_is_completely_blank() {
        let glyph = render_glyph(' ').unwrap();
        assert_eq!(ink_count(&glyph), 0);
        assert!(glyph.iter().all(|p| p[3] == 0));
    }

    #[test]
    fn every_non_space_glyph_has_ink() {
        for i in 1..CHAR_COUNT {
            let c = char::from_u32(0x20 + i as u32).unwrap();
            let glyph = render_glyph(c).unwrap();
            assert!(ink_count(&glyph) > 0, "{c:?} rendered blank");
        }
    }

    #[test]
    fn hyphen_has_ink_only_on_its_middle_row() {
        let glyph = render_glyph('-').unwrap();
        for y in 0..GLYPH_SIZE as usize {
            let expected = y == 3 + PAD_Y;
            assert_eq!(
                row_has_ink(&glyph, y),
                expected,
                "hyphen row {y}: expected ink={expected}"
            );
        }
    }

    #[test]
    fn underscore_sits_on_the_last_font_row() {
        let glyph = render_glyph('_').unwrap();
        assert!(row_has_ink(&glyph, FONT_HEIGHT - 1 + PAD_Y));
        assert!(!row_has_ink(&glyph, 0));
    }

    #[test]
    fn capital_a_has_a_hollow_middle() {
        // The crossbar row is solid, but the rows above and below must have gaps
        // — a solid block would mean the glyph is a filled rectangle.
        let rows = glyph_rows('A').unwrap();
        assert_eq!(rows[3], 0b11111, "crossbar should be solid");
        for r in [1usize, 2, 4, 5, 6] {
            assert_ne!(rows[r], 0b11111, "row {r} of 'A' should not be solid");
        }
    }

    #[test]
    fn descenders_reach_the_bottom_row_and_x_height_letters_do_not() {
        for c in ['g', 'j', 'p', 'q', 'y'] {
            let rows = glyph_rows(c).unwrap();
            assert_ne!(rows[FONT_HEIGHT - 1], 0, "{c:?} should have a descender");
        }
        for c in ['a', 'e', 'o', 'x', 'c', 'm', 'n'] {
            let rows = glyph_rows(c).unwrap();
            assert_eq!(rows[0], 0, "{c:?} should not reach the ascender row");
        }
    }

    #[test]
    fn no_two_different_characters_share_a_bitmap() {
        // Catches the most likely authoring error: a copy-pasted row block left
        // unedited between two letters.
        for i in 0..CHAR_COUNT {
            for j in (i + 1)..CHAR_COUNT {
                if FONT[i] == FONT[j] {
                    let a = char::from_u32(0x20 + i as u32).unwrap();
                    let b = char::from_u32(0x20 + j as u32).unwrap();
                    panic!("{a:?} and {b:?} have identical bitmaps");
                }
            }
        }
    }

    #[test]
    fn rendering_respects_the_padding_and_cell_bounds() {
        let size = GLYPH_SIZE as usize;
        for i in 0..CHAR_COUNT {
            let c = char::from_u32(0x20 + i as u32).unwrap();
            let glyph = render_glyph(c).unwrap();
            for y in 0..size {
                for x in 0..size {
                    if glyph[y * size + x][3] == 0 {
                        continue;
                    }
                    assert!(
                        x >= PAD_X && x < PAD_X + FONT_WIDTH,
                        "{c:?} has ink at x={x}, outside the padded 5px band"
                    );
                    assert!(
                        y >= PAD_Y && y < PAD_Y + FONT_HEIGHT,
                        "{c:?} has ink at y={y}, outside the padded 7px band"
                    );
                }
            }
        }
    }

    #[test]
    fn bit_order_is_left_to_right() {
        // '1' has its stem left of centre on the top row (0b00100 -> x = 2),
        // which pins down the bit-to-column mapping.
        let glyph = render_glyph('1').unwrap();
        let size = GLYPH_SIZE as usize;
        assert!(glyph[0 * size + (2 + PAD_X)][3] > 0, "expected ink at column 2");
        assert_eq!(glyph[0 * size + (0 + PAD_X)][3], 0, "column 0 should be clear");

        // '/' rises to the right: its top row has ink at the far right, its
        // bottom row at the far left. If the bit order were mirrored this would
        // read as a backslash.
        let slash = render_glyph('/').unwrap();
        assert!(slash[0 * size + (4 + PAD_X)][3] > 0, "'/' top row should be rightmost");
        assert!(slash[6 * size + (0 + PAD_X)][3] > 0, "'/' bottom row should be leftmost");
    }

    #[test]
    fn atlas_length_and_contents_match_the_glyphs() {
        let atlas = build_font_atlas();
        assert_eq!(atlas.len(), CHAR_COUNT * GLYPH_BYTES * 4);

        // Spot-check that 'A' lands at its own index.
        let idx = glyph_index('A').unwrap();
        let offset = idx * GLYPH_BYTES * 4;
        let expected = render_glyph('A').unwrap();
        for (p, pixel) in expected.iter().enumerate() {
            for c in 0..4 {
                assert_eq!(
                    atlas[offset + p * 4 + c],
                    pixel[c],
                    "atlas byte mismatch for 'A' at pixel {p} channel {c}"
                );
            }
        }

        // Space at index 0 must be all zeros.
        assert!(atlas[..GLYPH_BYTES * 4].iter().all(|b| *b == 0));
    }

    #[test]
    fn text_width_counts_characters_including_unsupported_ones() {
        assert_eq!(text_width(""), 0);
        assert_eq!(text_width("FPS: 60"), 7);
        // Unsupported chars still occupy a cell so layout stays predictable.
        assert_eq!(text_width("aЯb"), 3);
    }
}
