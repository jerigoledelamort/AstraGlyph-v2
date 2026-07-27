// Procedural glyph atlas — generates bitmap patterns for ASCII/Unicode characters.
// No external font library; all glyphs are 8x8 bitmaps generated in code.

/// Character size in pixels (8x8).
pub const GLYPH_SIZE: u32 = 8;
pub const GLYPH_BYTES: usize = GLYPH_SIZE as usize * GLYPH_SIZE as usize;

/// A single 8x8 glyph stored as a row-major RGBA bitmap.
pub type Glyph = [[u8; 4]; GLYPH_BYTES];

/// Brightness ramp characters ordered from dark to bright.
pub const BRIGHTNESS_RAMP: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

/// Unicode block characters for additional shading levels.
pub const BLOCK_CHARS: &[char] = &['░', '▒', '▓', '█'];

/// All characters in the atlas, ordered by index.
pub const ALL_CHARS: &[char] = &[
    ' ', '.', ':', '-', '=', '+', '*', '#', '%', '@',
    '░', '▒', '▓', '█',
];

/// Index of the space character (all black).
pub const SPACE_INDEX: u32 = 0;

/// Generate the 8x8 bitmap for a character at the given brightness level.
///
/// Each character is an 8x8 RGBA image where:
/// - R, G, B = 255 (white, tinted by color at render time)
/// - A = intensity (0-255)
fn generate_glyph(char_index: usize) -> Glyph {
    let mut glyph = [[0u8; 4]; GLYPH_BYTES];

    let set = |g: &mut Glyph, x: usize, y: usize, alpha: u8| {
        let idx = y * GLYPH_SIZE as usize + x;
        g[idx] = [255, 255, 255, alpha];
    };

    match char_index {
        // ' ' — empty
        0 => {}

        // '.' — single dot in lower-center
        1 => {
            set(&mut glyph, 3, 5, 255);
            set(&mut glyph, 4, 5, 255);
        }

        // ':' — two vertical dots
        2 => {
            set(&mut glyph, 3, 2, 255);
            set(&mut glyph, 4, 2, 255);
            set(&mut glyph, 3, 5, 255);
            set(&mut glyph, 4, 5, 255);
        }

        // '-' — horizontal line in middle
        3 => {
            for x in 2..6 {
                set(&mut glyph, x, 3, 255);
                set(&mut glyph, x, 4, 255);
            }
        }

        // '=' — two horizontal lines
        4 => {
            for x in 2..6 {
                set(&mut glyph, x, 2, 255);
                set(&mut glyph, x, 3, 255);
                set(&mut glyph, x, 5, 255);
                set(&mut glyph, x, 6, 255);
            }
        }

        // '+' — cross
        5 => {
            for x in 2..6 {
                set(&mut glyph, x, 3, 255);
                set(&mut glyph, x, 4, 255);
            }
            for y in 2..6 {
                set(&mut glyph, 3, y, 255);
                set(&mut glyph, 4, y, 255);
            }
        }

        // '*' — star / asterisk
        6 => {
            set(&mut glyph, 3, 1, 255);
            set(&mut glyph, 4, 1, 255);
            set(&mut glyph, 3, 2, 255);
            set(&mut glyph, 4, 2, 255);
            for x in 1..7 {
                set(&mut glyph, x, 3, 255);
                set(&mut glyph, x, 4, 255);
            }
            set(&mut glyph, 3, 5, 255);
            set(&mut glyph, 4, 5, 255);
            set(&mut glyph, 3, 6, 255);
            set(&mut glyph, 4, 6, 255);
        }

        // '#' — hash / grid
        7 => {
            for y in 0..8 {
                set(&mut glyph, 1, y, 255);
                set(&mut glyph, 2, y, 255);
                set(&mut glyph, 5, y, 255);
                set(&mut glyph, 6, y, 255);
            }
            for x in 0..8 {
                set(&mut glyph, x, 2, 255);
                set(&mut glyph, x, 3, 255);
                set(&mut glyph, x, 5, 255);
                set(&mut glyph, x, 6, 255);
            }
        }

        // '%' — checkerboard-ish
        8 => {
            for y in 0..8 {
                for x in 0..8 {
                    if (x + y) % 2 == 0 {
                        set(&mut glyph, x, y, 255);
                    }
                }
            }
        }

        // '@' — filled circle with hole
        9 => {
            // Outer ring
            for y in 0..8 {
                for x in 0..8 {
                    let dx = x as f32 - 3.5;
                    let dy = y as f32 - 3.5;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= 3.5 && dist >= 2.0 {
                        set(&mut glyph, x, y, 255);
                    }
                }
            }
            // Fill center partially
            set(&mut glyph, 3, 3, 255);
            set(&mut glyph, 4, 3, 255);
            set(&mut glyph, 3, 4, 255);
            set(&mut glyph, 4, 4, 255);
        }

        // '░' — light shade (25% — sparse dots)
        10 => {
            for y in 0..8 {
                for x in 0..8 {
                    if x % 2 == 0 && y % 2 == 0 {
                        set(&mut glyph, x, y, 255);
                    }
                }
            }
        }

        // '▒' — medium shade (50% — checkerboard)
        11 => {
            for y in 0..8 {
                for x in 0..8 {
                    if (x + y) % 2 == 0 {
                        set(&mut glyph, x, y, 255);
                    }
                }
            }
        }

        // '▓' — dark shade (75% — dense dots)
        12 => {
            for y in 0..8 {
                for x in 0..8 {
                    if !(x % 2 == 1 && y % 2 == 1) {
                        set(&mut glyph, x, y, 255);
                    }
                }
            }
        }

        // '█' — full block (100%)
        13 => {
            for y in 0..8 {
                for x in 0..8 {
                    set(&mut glyph, x, y, 255);
                }
            }
        }

        _ => {}
    }

    glyph
}

/// Build the full glyph atlas as a flat RGBA byte buffer.
///
/// Layout: `ALL_CHARS.len()` glyphs, each 8x8 pixels, row-major.
/// Total size: `ALL_CHARS.len() * GLYPH_SIZE * GLYPH_SIZE * 4` bytes.
pub fn build_atlas() -> Vec<u8> {
    let count = ALL_CHARS.len();
    let mut atlas = Vec::with_capacity(count * GLYPH_BYTES * 4);
    for i in 0..count {
        let glyph = generate_glyph(i);
        for pixel in &glyph {
            atlas.extend_from_slice(pixel);
        }
    }
    atlas
}

/// Total number of glyphs in the atlas.
pub fn glyph_count() -> usize {
    ALL_CHARS.len()
}

/// Map a brightness value (0.0 - 1.0) to a glyph index.
///
/// Uses the full character set (brightness ramp + block chars) for
/// a smooth gradient.
pub fn brightness_to_index(brightness: f32) -> u32 {
    let clamped = brightness.clamp(0.0, 1.0);
    let count = ALL_CHARS.len() as f32;
    ((clamped * (count - 1.0)).round() as u32).min(ALL_CHARS.len() as u32 - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_size() {
        let atlas = build_atlas();
        let expected = glyph_count() * GLYPH_BYTES * 4;
        assert_eq!(atlas.len(), expected);
    }

    #[test]
    fn brightness_mapping() {
        assert_eq!(brightness_to_index(0.0), 0);  // darkest → space
        let last = ALL_CHARS.len() as u32 - 1;
        assert_eq!(brightness_to_index(1.0), last); // brightest → full block
    }

    #[test]
    fn brightness_monotonic() {
        let mut prev = 0u32;
        for i in 0..=100 {
            let b = i as f32 / 100.0;
            let idx = brightness_to_index(b);
            assert!(idx >= prev, "non-monotonic at brightness={b}");
            prev = idx;
        }
    }

    #[test]
    fn space_glyph_is_empty() {
        let atlas = build_atlas();
        // First glyph (space) should be all zeros.
        for i in 0..GLYPH_BYTES * 4 {
            assert_eq!(atlas[i], 0, "space glyph not empty at byte {i}");
        }
    }

    #[test]
    fn full_block_glyph_is_filled() {
        let atlas = build_atlas();
        let offset = (ALL_CHARS.len() - 1) * GLYPH_BYTES * 4;
        for i in 0..GLYPH_BYTES * 4 {
            assert_eq!(atlas[offset + i], 255, "full block not filled at byte {i}");
        }
    }
}