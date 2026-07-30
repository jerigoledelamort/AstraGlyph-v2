// ASCII colour output modes — quantization of a per-cell RGB triple down to the
// fidelity levels a terminal-style renderer targets: 24-bit TrueColor, the
// xterm-256 palette, the 16 system colours, a grey ramp, or a single foreground
// colour (ROADMAP 3.1 "Color modes: ANSI 256, TrueColor (16M), indexed palette").
//
// Design notes:
// - Pure CPU maths: no GPU, no WGSL, no external crates. The scene pass renders
//   one pixel per ASCII cell (120x68 today) and the ASCII pass already reads
//   those pixels back to the CPU, so quantizing a whole frame here costs a few
//   thousand cheap integer operations and stays fully unit-testable.
// - `quantize` maps a colour to the RGB the mode can actually display, so the
//   composite pass (which takes f32 RGB per glyph instance) renders every mode
//   unchanged. `palette_index` returns the terminal colour index for the modes
//   that have one — that is what a future terminal backend would emit.
// - Nearest-colour metric: squared Euclidean distance in 8-bit RGB space. Not
//   perceptually uniform, but deterministic, dependency-free and the de-facto
//   standard for xterm-256 mapping. Ties always resolve to the lower palette
//   index, which makes every mapping reproducible across runs and platforms.
// - The xterm tables are the real ones: the 6x6x6 cube uses the *irregular*
//   component levels 0/95/135/175/215/255 (indices 16..=231) and the grey ramp
//   is 8 + 10*i for i in 0..24 (indices 232..=255). Assuming evenly spaced cube
//   levels is the classic bug here, so the tests pin the exact numbers.
// - `rgb_to_ansi256` never returns a system colour (0..=15): those are
//   user-configurable in most terminals, and several of them duplicate other
//   palette entries (black == cube 16, white == cube 231, grey == ramp 244),
//   which would make the palette round-trip ambiguous. Use `ColorMode::Ansi16`
//   when the system colours are wanted.
// - `luminance` uses the ITU-R BT.601 coefficients 0.299/0.587/0.114, exactly
//   matching `renderer::ascii_pass`, so a grey-scaled
//   frame picks the same glyphs as a coloured one.
// - All inputs are sanitized (NaN -> 0.0, clamp to 0..=1) before use: the shading
//   pipeline can produce out-of-range or non-finite values, and neither a panic
//   nor a NaN may reach the composite pass.

/// Component levels of the xterm 6x6x6 colour cube (indices 16..=231).
/// Deliberately irregular — this is the xterm specification, not a ramp.
const CUBE_LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];

/// First index of the 6x6x6 colour cube.
const CUBE_FIRST: u8 = 16;

/// First index of the 24-step grey ramp.
const GREY_FIRST: u8 = 232;

/// Value of grey ramp step 0.
const GREY_BASE: u8 = 8;

/// Distance between two consecutive grey ramp steps.
const GREY_STEP: u8 = 10;

/// Number of steps in the grey ramp (8, 18, ... 238).
const GREY_COUNT: u8 = 24;

/// The 16 standard system colours (xterm / VGA defaults).
const ANSI16_PALETTE: [[u8; 3]; 16] = [
    [0, 0, 0],       // 0  black
    [128, 0, 0],     // 1  maroon
    [0, 128, 0],     // 2  green
    [128, 128, 0],   // 3  olive
    [0, 0, 128],     // 4  navy
    [128, 0, 128],   // 5  purple
    [0, 128, 128],   // 6  teal
    [192, 192, 192], // 7  silver
    [128, 128, 128], // 8  grey
    [255, 0, 0],     // 9  red
    [0, 255, 0],     // 10 lime
    [255, 255, 0],   // 11 yellow
    [0, 0, 255],     // 12 blue
    [255, 0, 255],   // 13 fuchsia
    [0, 255, 255],   // 14 aqua
    [255, 255, 255], // 15 white
];

/// The single foreground colour used by [`ColorMode::Monochrome`].
pub const MONOCHROME_FOREGROUND: [f32; 3] = [1.0, 1.0, 1.0];

/// Colour fidelity of the ASCII output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColorMode {
    /// 24-bit colour: the cell colour is displayed as-is (only sanitized).
    #[default]
    TrueColor,
    /// The xterm-256 palette: 16 system colours + a 6x6x6 cube + 24 greys.
    /// Quantization only ever selects from the cube and the grey ramp.
    Ansi256,
    /// The 16 standard system colours.
    Ansi16,
    /// Luminance only, quantized to the 24-step xterm grey ramp.
    Grayscale,
    /// A single foreground colour; luminance drives only the glyph choice.
    Monochrome,
}

impl ColorMode {
    /// Every mode, in declaration order. Useful for cycling modes at runtime.
    pub const ALL: [ColorMode; 5] = [
        ColorMode::TrueColor,
        ColorMode::Ansi256,
        ColorMode::Ansi16,
        ColorMode::Grayscale,
        ColorMode::Monochrome,
    ];

    /// Quantize one cell colour to what this mode can actually display.
    ///
    /// Input is 0..=1 RGB; the output is the displayable 0..=1 RGB, ready for
    /// the composite pass. Non-finite and out-of-range components are sanitized
    /// (NaN becomes 0.0, everything else is clamped), so the result is always
    /// finite and inside 0..=1.
    pub fn quantize(&self, rgb: [f32; 3]) -> [f32; 3] {
        match self {
            ColorMode::TrueColor => sanitize(rgb),
            ColorMode::Ansi256 => ansi256_to_rgb(rgb_to_ansi256(rgb)),
            ColorMode::Ansi16 => ansi16_to_rgb(rgb_to_ansi16(rgb)),
            ColorMode::Grayscale => {
                let level = grey_level(grey_step_of(rgb));
                from_bytes([level, level, level])
            }
            ColorMode::Monochrome => MONOCHROME_FOREGROUND,
        }
    }

    /// The terminal colour index for this colour, for modes that have one.
    ///
    /// Returns `None` for [`ColorMode::TrueColor`] (emits raw RGB) and
    /// [`ColorMode::Monochrome`] (never changes the foreground colour).
    pub fn palette_index(&self, rgb: [f32; 3]) -> Option<u8> {
        match self {
            ColorMode::TrueColor | ColorMode::Monochrome => None,
            ColorMode::Ansi256 => Some(rgb_to_ansi256(rgb)),
            ColorMode::Ansi16 => Some(rgb_to_ansi16(rgb)),
            ColorMode::Grayscale => Some(GREY_FIRST + grey_step_of(rgb)),
        }
    }
}

/// Quantize a whole frame in place — one call per frame for the renderer.
pub fn quantize_buffer(mode: ColorMode, pixels: &mut [[f32; 3]]) {
    for pixel in pixels.iter_mut() {
        *pixel = mode.quantize(*pixel);
    }
}

/// Perceived brightness, ITU-R BT.601: `0.299 R + 0.587 G + 0.114 B`.
///
/// Input is sanitized first and the result is clamped to 0..=1, so the return
/// value is always a usable brightness even for garbage input.
pub fn luminance(rgb: [f32; 3]) -> f32 {
    let c = sanitize(rgb);
    (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]).clamp(0.0, 1.0)
}

/// Exact RGB of an xterm-256 palette entry (the inverse of [`rgb_to_ansi256`]).
///
/// - `0..=15`   — system colours (see [`ansi16_to_rgb`])
/// - `16..=231` — colour cube: `index = 16 + 36*r + 6*g + b`, components taken
///   from the levels 0/95/135/175/215/255
/// - `232..=255` — grey ramp: `8 + 10*(index - 232)`
pub fn ansi256_to_rgb(index: u8) -> [f32; 3] {
    if index < CUBE_FIRST {
        ansi16_to_rgb(index)
    } else if index < GREY_FIRST {
        let cube = (index - CUBE_FIRST) as usize;
        from_bytes([
            CUBE_LEVELS[cube / 36],
            CUBE_LEVELS[(cube % 36) / 6],
            CUBE_LEVELS[cube % 6],
        ])
    } else {
        let level = grey_level(index - GREY_FIRST);
        from_bytes([level, level, level])
    }
}

/// Exact RGB of one of the 16 standard system colours.
///
/// Only the low 4 bits of `index` are used, so out-of-range indices wrap
/// instead of panicking.
pub fn ansi16_to_rgb(index: u8) -> [f32; 3] {
    from_bytes(ANSI16_PALETTE[(index % 16) as usize])
}

/// Nearest xterm-256 entry to `rgb`, by squared Euclidean distance in 8-bit RGB.
///
/// Only the colour cube (16..=231) and the grey ramp (232..=255) are searched;
/// the system colours are never returned (see the module notes). Ties resolve to
/// the lower index, so this is the exact nearest-neighbour result over 16..=255
/// and `rgb_to_ansi256(ansi256_to_rgb(i)) == i` holds for every `i` in that range.
pub fn rgb_to_ansi256(rgb: [f32; 3]) -> u8 {
    let c = to_bytes(rgb);

    // Squared distance separates per channel, so the nearest cube colour is the
    // nearest level in each channel independently.
    let r = nearest_cube_level(c[0]);
    let g = nearest_cube_level(c[1]);
    let b = nearest_cube_level(c[2]);
    let cube_rgb = [CUBE_LEVELS[r], CUBE_LEVELS[g], CUBE_LEVELS[b]];
    let cube_index = CUBE_FIRST + (36 * r + 6 * g + b) as u8;
    let cube_dist = dist2(c, cube_rgb);

    // For a grey (v, v, v) the distance is `3*v^2 - 2*v*sum + const`, a convex
    // parabola in `v` centred on `sum / 3`; the nearest ramp entry is therefore
    // the one minimizing |3*v - sum|, computed exactly in integers.
    let sum = c[0] as u32 + c[1] as u32 + c[2] as u32;
    let step = nearest_grey_step(sum);
    let grey = grey_level(step);
    let grey_dist = dist2(c, [grey, grey, grey]);

    // Tie goes to the cube, whose indices are all below the grey ramp's.
    if grey_dist < cube_dist {
        GREY_FIRST + step
    } else {
        cube_index
    }
}

/// Nearest system colour to `rgb`, by squared Euclidean distance in 8-bit RGB.
///
/// Always returns an index in 0..=15; ties resolve to the lower index.
pub fn rgb_to_ansi16(rgb: [f32; 3]) -> u8 {
    let c = to_bytes(rgb);
    let mut best = 0u8;
    let mut best_dist = u32::MAX;
    for (index, entry) in ANSI16_PALETTE.iter().enumerate() {
        let dist = dist2(c, *entry);
        if dist < best_dist {
            best_dist = dist;
            best = index as u8;
        }
    }
    best
}

/// Replace NaN with 0.0 and clamp every component into 0..=1.
fn sanitize(rgb: [f32; 3]) -> [f32; 3] {
    let fix = |c: f32| if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) };
    [fix(rgb[0]), fix(rgb[1]), fix(rgb[2])]
}

/// Sanitize and convert to 8-bit components (round-to-nearest).
fn to_bytes(rgb: [f32; 3]) -> [u8; 3] {
    let c = sanitize(rgb);
    [
        (c[0] * 255.0).round() as u8,
        (c[1] * 255.0).round() as u8,
        (c[2] * 255.0).round() as u8,
    ]
}

/// Convert 8-bit components back to 0..=1 floats.
fn from_bytes(rgb: [u8; 3]) -> [f32; 3] {
    [
        rgb[0] as f32 / 255.0,
        rgb[1] as f32 / 255.0,
        rgb[2] as f32 / 255.0,
    ]
}

/// Squared Euclidean distance between two 8-bit colours.
/// Maximum value is 3 * 255^2 = 195075, so `u32` cannot overflow.
fn dist2(a: [u8; 3], b: [u8; 3]) -> u32 {
    let mut sum = 0u32;
    for i in 0..3 {
        let d = a[i].abs_diff(b[i]) as u32;
        sum += d * d;
    }
    sum
}

/// Index into [`CUBE_LEVELS`] of the level closest to `component`.
/// Ties resolve to the lower level index.
fn nearest_cube_level(component: u8) -> usize {
    let mut best = 0usize;
    let mut best_dist = u32::MAX;
    for (index, level) in CUBE_LEVELS.iter().enumerate() {
        let dist = component.abs_diff(*level) as u32;
        if dist < best_dist {
            best_dist = dist;
            best = index;
        }
    }
    best
}

/// Grey ramp step whose triple `(v, v, v)` is closest to a colour with the given
/// channel sum. Ties resolve to the lower step.
fn nearest_grey_step(channel_sum: u32) -> u8 {
    let mut best = 0u8;
    let mut best_dist = u32::MAX;
    for step in 0..GREY_COUNT {
        let dist = (grey_level(step) as u32 * 3).abs_diff(channel_sum);
        if dist < best_dist {
            best_dist = dist;
            best = step;
        }
    }
    best
}

/// 8-bit value of grey ramp step `step` (0..24): `8 + 10 * step`.
///
/// Steps at or above `GREY_COUNT` saturate at the last ramp entry (238);
/// without the clamp the `u8` arithmetic would overflow (and panic in a debug
/// build) for `step >= 25`.
fn grey_level(step: u8) -> u8 {
    GREY_BASE + GREY_STEP * step.min(GREY_COUNT - 1)
}

/// Grey ramp step for a colour's luminance — the basis of both the grayscale
/// quantized colour and its palette index.
fn grey_step_of(rgb: [f32; 3]) -> u8 {
    let byte = (luminance(rgb) * 255.0).round() as u32;
    nearest_grey_step(byte * 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLACK: [f32; 3] = [0.0, 0.0, 0.0];
    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
    const RED: [f32; 3] = [1.0, 0.0, 0.0];
    const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
    const BLUE: [f32; 3] = [0.0, 0.0, 1.0];

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-6)
    }

    /// The 240 palette entries `rgb_to_ansi256` is allowed to return (16..=255),
    /// as 8-bit triples, in index order.
    fn ansi256_search_space() -> Vec<[u8; 3]> {
        (CUBE_FIRST as u16..=255)
            .map(|i| to_bytes(ansi256_to_rgb(i as u8)))
            .collect()
    }

    /// Exhaustive nearest-neighbour over the search space, first minimum wins.
    /// The reference implementation the analytic search must agree with.
    fn brute_force_in(palette: &[[u8; 3]], rgb: [f32; 3]) -> u8 {
        let c = to_bytes(rgb);
        let mut best = CUBE_FIRST;
        let mut best_dist = u32::MAX;
        for (offset, entry) in palette.iter().enumerate() {
            let dist = dist2(c, *entry);
            if dist < best_dist {
                best_dist = dist;
                best = CUBE_FIRST + offset as u8;
            }
        }
        best
    }

    fn brute_force_ansi256(rgb: [f32; 3]) -> u8 {
        brute_force_in(&ansi256_search_space(), rgb)
    }

    #[test]
    fn cube_levels_are_the_xterm_ones() {
        // The exact specification values — not an even ramp.
        assert_eq!(CUBE_LEVELS, [0, 95, 135, 175, 215, 255]);
        // Spot-check known xterm entries: 16 #000000, 231 #ffffff, 196 #ff0000,
        // 46 #00ff00, 21 #0000ff, 226 #ffff00, 51 #00ffff, 201 #ff00ff,
        // 100 #878700 (135, 135, 0).
        assert_eq!(to_bytes(ansi256_to_rgb(16)), [0, 0, 0]);
        assert_eq!(to_bytes(ansi256_to_rgb(231)), [255, 255, 255]);
        assert_eq!(to_bytes(ansi256_to_rgb(196)), [255, 0, 0]);
        assert_eq!(to_bytes(ansi256_to_rgb(46)), [0, 255, 0]);
        assert_eq!(to_bytes(ansi256_to_rgb(21)), [0, 0, 255]);
        assert_eq!(to_bytes(ansi256_to_rgb(226)), [255, 255, 0]);
        assert_eq!(to_bytes(ansi256_to_rgb(51)), [0, 255, 255]);
        assert_eq!(to_bytes(ansi256_to_rgb(201)), [255, 0, 255]);
        assert_eq!(to_bytes(ansi256_to_rgb(100)), [135, 135, 0]);
        // Every cube component must come from the level table.
        for index in 16u16..=231 {
            let bytes = to_bytes(ansi256_to_rgb(index as u8));
            for channel in bytes {
                assert!(
                    CUBE_LEVELS.contains(&channel),
                    "index {index} has non-cube component {channel}"
                );
            }
        }
    }

    #[test]
    fn grey_ramp_is_8_plus_10i() {
        for i in 0u8..24 {
            let expected = 8 + 10 * i;
            assert_eq!(grey_level(i), expected);
            let bytes = to_bytes(ansi256_to_rgb(GREY_FIRST + i));
            assert_eq!(bytes, [expected, expected, expected], "grey step {i}");
        }
        // Ends of the ramp: 232 -> 8, 255 -> 238.
        assert_eq!(to_bytes(ansi256_to_rgb(232)), [8, 8, 8]);
        assert_eq!(to_bytes(ansi256_to_rgb(255)), [238, 238, 238]);
    }

    #[test]
    fn ansi256_palette_round_trips_exactly() {
        for index in CUBE_FIRST as u16..=255 {
            let index = index as u8;
            let rgb = ansi256_to_rgb(index);
            assert_eq!(
                rgb_to_ansi256(rgb),
                index,
                "round-trip failed for index {index} ({rgb:?})"
            );
        }
    }

    #[test]
    fn ansi16_palette_round_trips_exactly() {
        for index in 0u8..16 {
            assert_eq!(rgb_to_ansi16(ansi16_to_rgb(index)), index);
        }
        // Out-of-range indices wrap instead of panicking.
        assert_eq!(ansi16_to_rgb(16), ansi16_to_rgb(0));
        assert_eq!(ansi16_to_rgb(255), ansi16_to_rgb(15));
    }

    #[test]
    fn truecolor_is_identity() {
        for rgb in [BLACK, WHITE, RED, GREEN, BLUE, [0.25, 0.5, 0.75]] {
            assert_eq!(ColorMode::TrueColor.quantize(rgb), rgb);
            assert_eq!(ColorMode::TrueColor.palette_index(rgb), None);
        }
    }

    #[test]
    fn primaries_in_ansi256() {
        assert_eq!(ColorMode::Ansi256.palette_index(BLACK), Some(16));
        assert_eq!(ColorMode::Ansi256.palette_index(WHITE), Some(231));
        assert_eq!(ColorMode::Ansi256.palette_index(RED), Some(196));
        assert_eq!(ColorMode::Ansi256.palette_index(GREEN), Some(46));
        assert_eq!(ColorMode::Ansi256.palette_index(BLUE), Some(21));
        // Pure primaries are exact cube entries, so quantize is lossless.
        for rgb in [BLACK, WHITE, RED, GREEN, BLUE] {
            assert!(close(ColorMode::Ansi256.quantize(rgb), rgb), "{rgb:?}");
        }
    }

    #[test]
    fn primaries_in_ansi16() {
        assert_eq!(ColorMode::Ansi16.palette_index(BLACK), Some(0));
        assert_eq!(ColorMode::Ansi16.palette_index(WHITE), Some(15));
        assert_eq!(ColorMode::Ansi16.palette_index(RED), Some(9));
        assert_eq!(ColorMode::Ansi16.palette_index(GREEN), Some(10));
        assert_eq!(ColorMode::Ansi16.palette_index(BLUE), Some(12));
        for rgb in [BLACK, WHITE, RED, GREEN, BLUE] {
            assert!(close(ColorMode::Ansi16.quantize(rgb), rgb), "{rgb:?}");
        }
    }

    #[test]
    fn ansi16_indices_stay_in_range() {
        for r in 0..=8 {
            for g in 0..=8 {
                for b in 0..=8 {
                    let rgb = [r as f32 / 8.0, g as f32 / 8.0, b as f32 / 8.0];
                    let index = rgb_to_ansi16(rgb);
                    assert!(index <= 15, "index {index} out of range for {rgb:?}");
                    assert_eq!(ColorMode::Ansi16.palette_index(rgb), Some(index));
                }
            }
        }
        // Garbage input must not escape the range either.
        for rgb in [
            [f32::NAN, 2.0, -1.0],
            [f32::INFINITY, f32::NEG_INFINITY, 0.5],
        ] {
            assert!(rgb_to_ansi16(rgb) <= 15);
        }
    }

    #[test]
    fn primaries_in_grayscale() {
        // luminance * 255 -> nearest ramp entry (8 + 10i):
        //   black 0.000 ->   0.0 -> 8   (step 0,  index 232)
        //   white 1.000 -> 255.0 -> 238 (step 23, index 255)
        //   red   0.299 ->  76.2 -> 78  (step 7,  index 239)
        //   green 0.587 -> 149.7 -> 148 (step 14, index 246)
        //   blue  0.114 ->  29.1 -> 28  (step 2,  index 234)
        let cases = [
            (BLACK, 232u8, 8u8),
            (WHITE, 255, 238),
            (RED, 239, 78),
            (GREEN, 246, 148),
            (BLUE, 234, 28),
        ];
        for (rgb, index, level) in cases {
            assert_eq!(ColorMode::Grayscale.palette_index(rgb), Some(index), "{rgb:?}");
            let out = ColorMode::Grayscale.quantize(rgb);
            assert!(close(out, from_bytes([level, level, level])), "{rgb:?} -> {out:?}");
        }
    }

    #[test]
    fn grayscale_output_is_neutral() {
        for r in 0..=6 {
            for g in 0..=6 {
                for b in 0..=6 {
                    let rgb = [r as f32 / 6.0, g as f32 / 6.0, b as f32 / 6.0];
                    let out = ColorMode::Grayscale.quantize(rgb);
                    assert_eq!(out[0], out[1], "not neutral: {out:?}");
                    assert_eq!(out[1], out[2], "not neutral: {out:?}");
                    let index = ColorMode::Grayscale.palette_index(rgb).unwrap();
                    assert!((GREY_FIRST..=255).contains(&index), "index {index}");
                    // The quantized grey must be the colour of that index.
                    assert!(close(out, ansi256_to_rgb(index)), "{out:?} vs index {index}");
                }
            }
        }
    }

    #[test]
    fn monochrome_is_a_single_colour() {
        for rgb in [BLACK, WHITE, RED, GREEN, BLUE, [0.3, 0.6, 0.9]] {
            assert_eq!(ColorMode::Monochrome.quantize(rgb), MONOCHROME_FOREGROUND);
            assert_eq!(ColorMode::Monochrome.palette_index(rgb), None);
        }
    }

    #[test]
    fn luminance_uses_bt601_coefficients() {
        assert!((luminance(RED) - 0.299).abs() < 1e-6);
        assert!((luminance(GREEN) - 0.587).abs() < 1e-6);
        assert!((luminance(BLUE) - 0.114).abs() < 1e-6);
        assert_eq!(luminance(BLACK), 0.0);
        assert_eq!(luminance(WHITE), 1.0);
        assert!((luminance([0.5, 0.5, 0.5]) - 0.5).abs() < 1e-6);
        // 0.299*0.2 + 0.587*0.4 + 0.114*0.6 = 0.0598 + 0.2348 + 0.0684 = 0.363
        assert!((luminance([0.2, 0.4, 0.6]) - 0.363).abs() < 1e-6);
    }

    #[test]
    fn grey_ramp_beats_the_cube_for_near_greys() {
        // 128 is an exact ramp entry (8 + 10*12) but 7 away from cube level 135,
        // so mid grey must map to grey index 244, not to cube index
        // 16 + 36*2 + 6*2 + 2 = 102.
        assert_eq!(rgb_to_ansi256([0.5, 0.5, 0.5]), 244);
        assert_eq!(to_bytes(ansi256_to_rgb(244)), [128, 128, 128]);
        assert!(close(
            ColorMode::Ansi256.quantize([0.5, 0.5, 0.5]),
            from_bytes([128, 128, 128])
        ));
    }

    #[test]
    fn ties_resolve_to_the_lower_index() {
        // 115 is exactly between cube levels 95 and 135: the lower level wins,
        // giving 16 + 36*1 = 52 (grey is far away for a saturated red).
        let tie_red = from_bytes([115, 0, 0]);
        assert_eq!(rgb_to_ansi256(tie_red), 52);
        assert_eq!(brute_force_ansi256(tie_red), 52);
        // (13,13,13) sits exactly between ramp entries 8 and 18: step 0 wins.
        let tie_grey = from_bytes([13, 13, 13]);
        assert_eq!(rgb_to_ansi256(tie_grey), 232);
        assert_eq!(brute_force_ansi256(tie_grey), 232);
        // 160 is exactly between system grey (128, index 8) and silver
        // (192, index 7): the lower index wins.
        assert_eq!(rgb_to_ansi16(from_bytes([160, 160, 160])), 7);
    }

    #[test]
    fn analytic_search_matches_brute_force() {
        // Axis includes the exact tie points 115/155/195/235 plus a spread of
        // ordinary values, so both the cube and the grey ramp win somewhere.
        const AXIS: [u8; 20] = [
            0, 17, 34, 51, 68, 85, 102, 115, 119, 136, 153, 155, 170, 187, 195, 204, 221, 235,
            238, 255,
        ];
        let palette = ansi256_search_space();
        for r in AXIS {
            for g in AXIS {
                for b in AXIS {
                    let rgb = from_bytes([r, g, b]);
                    assert_eq!(
                        rgb_to_ansi256(rgb),
                        brute_force_in(&palette, rgb),
                        "mismatch at ({r}, {g}, {b})"
                    );
                }
            }
        }
    }

    #[test]
    fn non_finite_and_out_of_range_input_is_safe() {
        let inputs = [
            [f32::NAN, f32::NAN, f32::NAN],
            [f32::INFINITY, 0.0, 0.0],
            [f32::NEG_INFINITY, 1.0, f32::NAN],
            [-1.0, -0.5, -0.0],
            [2.0, 10.0, 1e30],
            [f32::MAX, f32::MIN, 0.5],
        ];
        for mode in ColorMode::ALL {
            for rgb in inputs {
                let out = mode.quantize(rgb);
                for c in out {
                    assert!(c.is_finite(), "{mode:?} produced non-finite {out:?}");
                    assert!((0.0..=1.0).contains(&c), "{mode:?} produced {out:?}");
                }
                // Must not panic, and any index must be a real palette slot.
                let _ = mode.palette_index(rgb);
                assert!(luminance(rgb).is_finite());
            }
        }
        // NaN is treated as 0.0, infinities clamp to the range ends.
        assert_eq!(ColorMode::TrueColor.quantize([f32::NAN; 3]), BLACK);
        assert_eq!(ColorMode::TrueColor.quantize([f32::INFINITY; 3]), WHITE);
        assert_eq!(ColorMode::TrueColor.quantize([-5.0, 0.5, 7.0]), [0.0, 0.5, 1.0]);
        assert_eq!(rgb_to_ansi256([2.0, 2.0, 2.0]), 231);
        assert_eq!(rgb_to_ansi256([f32::NAN, f32::NAN, f32::NAN]), 16);
    }

    #[test]
    fn quantize_buffer_matches_per_pixel_quantize() {
        let source: Vec<[f32; 3]> = (0..64)
            .map(|i| {
                let t = i as f32 / 63.0;
                [t, 1.0 - t, (t * 3.0).fract()]
            })
            .collect();
        for mode in ColorMode::ALL {
            let mut buffer = source.clone();
            quantize_buffer(mode, &mut buffer);
            assert_eq!(buffer.len(), source.len());
            for (i, pixel) in buffer.iter().enumerate() {
                assert_eq!(*pixel, mode.quantize(source[i]), "{mode:?} pixel {i}");
            }
        }
        // Empty buffers are fine.
        let mut empty: Vec<[f32; 3]> = Vec::new();
        quantize_buffer(ColorMode::Ansi256, &mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn ansi256_low_indices_are_the_system_colours() {
        for index in 0u8..16 {
            assert_eq!(ansi256_to_rgb(index), ansi16_to_rgb(index), "index {index}");
        }
        // The duplicates that make the system colours unusable as quantization
        // targets: black == cube 16, white == cube 231, grey == ramp 244.
        assert_eq!(to_bytes(ansi16_to_rgb(0)), to_bytes(ansi256_to_rgb(16)));
        assert_eq!(to_bytes(ansi16_to_rgb(15)), to_bytes(ansi256_to_rgb(231)));
        assert_eq!(to_bytes(ansi16_to_rgb(8)), to_bytes(ansi256_to_rgb(244)));
    }

    #[test]
    fn ansi256_indices_never_hit_the_system_colours() {
        for r in 0..=5 {
            for g in 0..=5 {
                for b in 0..=5 {
                    let rgb = [r as f32 / 5.0, g as f32 / 5.0, b as f32 / 5.0];
                    let index = rgb_to_ansi256(rgb);
                    assert!(index >= CUBE_FIRST, "index {index} for {rgb:?}");
                    assert_eq!(ColorMode::Ansi256.palette_index(rgb), Some(index));
                }
            }
        }
        // Garbage input must stay inside the searched range too.
        for rgb in [
            [f32::NAN, 2.0, -1.0],
            [f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            [-3.0, 0.5, 9.0],
        ] {
            assert!(rgb_to_ansi256(rgb) >= CUBE_FIRST);
        }
    }

    #[test]
    fn non_palette_colour_maps_to_the_nearest_cube_entry() {
        // [0.25, 0.75, 0.5] are exact binary fractions, so the byte conversion
        // is not borderline: 0.25*255 = 63.75 -> 64, 0.75*255 = 191.25 -> 191,
        // 0.5*255 = 127.5 -> 128.
        //   cube: 64 -> 95 (r = 1, d = 31), 191 -> 175 (g = 3, d = 16),
        //         128 -> 135 (b = 2, d = 7); index 16 + 36 + 18 + 2 = 72,
        //         d^2 = 961 + 256 + 49 = 1266
        //   grey: sum 383, nearest ramp entry 128 (step 12),
        //         d^2 = 64^2 + 63^2 + 0 = 4096 + 3969 = 8065
        // so the cube wins.
        let rgb = [0.25, 0.75, 0.5];
        assert_eq!(to_bytes(rgb), [64, 191, 128]);
        assert_eq!(rgb_to_ansi256(rgb), 72);
        assert_eq!(brute_force_ansi256(rgb), 72);
        assert!(close(
            ColorMode::Ansi256.quantize(rgb),
            from_bytes([95, 175, 135])
        ));
    }

    #[test]
    fn palette_index_and_quantize_agree() {
        let samples = [
            BLACK,
            WHITE,
            RED,
            GREEN,
            BLUE,
            [0.25, 0.75, 0.5],
            [0.125, 0.125, 0.125],
            [1.0, 0.5, 0.0],
        ];
        for rgb in samples {
            for mode in ColorMode::ALL {
                match mode.palette_index(rgb) {
                    // The quantized colour must be exactly the colour of the
                    // index a terminal backend would emit.
                    Some(index) => {
                        let expected = match mode {
                            ColorMode::Ansi16 => ansi16_to_rgb(index),
                            _ => ansi256_to_rgb(index),
                        };
                        assert!(close(mode.quantize(rgb), expected), "{mode:?} {rgb:?}");
                    }
                    None => assert!(matches!(
                        mode,
                        ColorMode::TrueColor | ColorMode::Monochrome
                    )),
                }
            }
        }
    }

    #[test]
    fn grey_level_saturates_instead_of_overflowing() {
        assert_eq!(grey_level(0), GREY_BASE);
        assert_eq!(grey_level(GREY_COUNT - 1), 238);
        // Out-of-range steps clamp to the last ramp entry; `8 + 10 * 25` would
        // otherwise overflow the u8 arithmetic and panic in a debug build.
        assert_eq!(grey_level(GREY_COUNT), 238);
        assert_eq!(grey_level(255), 238);
    }

    #[test]
    fn default_mode_is_truecolor() {
        assert_eq!(ColorMode::default(), ColorMode::TrueColor);
        assert_eq!(ColorMode::ALL.len(), 5);
    }
}
