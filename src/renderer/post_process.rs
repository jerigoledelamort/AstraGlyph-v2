// CPU post-processing over the low-res scene buffer (ROADMAP Phase 3.2:
// bloom, SSAO, gamma correction, chromatic aberration).
//
// Why CPU: the ASCII pipeline already reads the scene render target back to
// system memory every frame (see renderer/ascii_pass.rs), and the buffer is one
// pixel per glyph cell — about 120x68. At that size these effects cost
// microseconds, need no extra render passes or WGSL, and become plain functions
// over a buffer, which means they can actually be unit-tested.
//
// Everything works in f32 RGB. Values above 1.0 are allowed between stages
// (bloom legitimately overshoots) and are only clamped when converting back to
// bytes in `to_rgba8`.

use crate::engine::core::{EngineError, Result};

/// A mutable f32 RGB image, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameBuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 3]>,
}

impl FrameBuffer {
    /// A black buffer of the given size.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0.0; 3]; (width as usize) * (height as usize)],
        }
    }

    /// Convert the RGBA8 readback into f32 RGB (alpha is dropped — the ASCII
    /// stage only ever uses colour and luminance).
    pub fn from_rgba8(pixels: &[[u8; 4]], width: u32, height: u32) -> Result<Self> {
        let expected = (width as usize) * (height as usize);
        if pixels.len() != expected {
            return Err(EngineError::InvalidState(format!(
                "post-process: expected {expected} pixels for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels: pixels
                .iter()
                .map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                    ]
                })
                .collect(),
        })
    }

    /// Convert back to RGBA8, clamping out-of-range values (bloom pushes past
    /// 1.0) and forcing alpha to opaque.
    pub fn to_rgba8(&self) -> Vec<[u8; 4]> {
        self.pixels
            .iter()
            .map(|p| {
                [
                    encode_channel(p[0]),
                    encode_channel(p[1]),
                    encode_channel(p[2]),
                    255,
                ]
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Pixel at exact coordinates, or `None` when out of range.
    pub fn get(&self, x: u32, y: u32) -> Option<[f32; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.pixels.get((y as usize) * (self.width as usize) + (x as usize)).copied()
    }

    /// Pixel with edge clamping — the sampling primitive the blur kernels use.
    /// Returns black for an empty buffer, which keeps kernels branch-free.
    pub fn sample_clamped(&self, x: i32, y: i32) -> [f32; 3] {
        if self.width == 0 || self.height == 0 {
            return [0.0; 3];
        }
        let cx = x.clamp(0, self.width as i32 - 1) as usize;
        let cy = y.clamp(0, self.height as i32 - 1) as usize;
        self.pixels[cy * (self.width as usize) + cx]
    }

    fn set(&mut self, x: u32, y: u32, value: [f32; 3]) {
        if x < self.width && y < self.height {
            let idx = (y as usize) * (self.width as usize) + (x as usize);
            self.pixels[idx] = value;
        }
    }
}

/// Non-linear window-space depth in 0..1 as read from a `Depth32Float` target
/// (1.0 = far plane, i.e. nothing was drawn).
#[derive(Clone, Debug)]
pub struct DepthBuffer {
    pub width: u32,
    pub height: u32,
    pub depth: Vec<f32>,
}

impl DepthBuffer {
    /// An all-far buffer of the given size.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: vec![1.0; (width as usize) * (height as usize)],
        }
    }

    /// Wrap a readback slice, checking its length against the dimensions.
    pub fn from_slice(depth: &[f32], width: u32, height: u32) -> Result<Self> {
        let expected = (width as usize) * (height as usize);
        if depth.len() != expected {
            return Err(EngineError::InvalidState(format!(
                "post-process: expected {expected} depth samples for {width}x{height}, got {}",
                depth.len()
            )));
        }
        Ok(Self { width, height, depth: depth.to_vec() })
    }

    /// Edge-clamped sample. Non-finite stored values are reported as far (1.0)
    /// so they cannot poison the occlusion maths.
    pub fn sample_clamped(&self, x: i32, y: i32) -> f32 {
        if self.width == 0 || self.height == 0 {
            return 1.0;
        }
        let cx = x.clamp(0, self.width as i32 - 1) as usize;
        let cy = y.clamp(0, self.height as i32 - 1) as usize;
        let d = self.depth[cy * (self.width as usize) + cx];
        if d.is_finite() { d } else { 1.0 }
    }
}

/// Which effects are enabled and how strongly.
#[derive(Clone, Copy, Debug)]
pub struct PostProcessSettings {
    /// Output gamma. 1.0 disables the correction entirely.
    pub gamma: f32,
    /// Luminance above which a pixel contributes to bloom.
    pub bloom_threshold: f32,
    /// Bloom mix strength. 0.0 disables bloom.
    pub bloom_intensity: f32,
    /// Bloom blur radius in cells.
    pub bloom_radius: u32,
    /// Horizontal per-channel offset in cells. 0.0 disables aberration.
    pub aberration_strength: f32,
    /// SSAO sampling radius in cells. 0 disables SSAO.
    pub ssao_radius: u32,
    /// How much occlusion darkens a pixel. 0.0 disables SSAO.
    pub ssao_strength: f32,
}

impl Default for PostProcessSettings {
    /// Everything off. The ASCII pipeline quantizes colour to a handful of
    /// glyphs anyway, so effects are opt-in per scene rather than silently
    /// altering the baseline image — and a default of "off" makes any visual
    /// change traceable to an explicit setting.
    fn default() -> Self {
        Self::none()
    }
}

impl PostProcessSettings {
    /// Every effect disabled.
    pub const fn none() -> Self {
        Self {
            gamma: 1.0,
            bloom_threshold: 1.0,
            bloom_intensity: 0.0,
            bloom_radius: 0,
            aberration_strength: 0.0,
            ssao_radius: 0,
            ssao_strength: 0.0,
        }
    }

    /// A visible-but-tasteful preset for the demo: gentle AO, a soft bloom and
    /// a hint of CRT fringing.
    pub const fn demo() -> Self {
        Self {
            gamma: 1.1,
            bloom_threshold: 0.65,
            bloom_intensity: 0.5,
            bloom_radius: 2,
            aberration_strength: 0.6,
            ssao_radius: 2,
            ssao_strength: 0.5,
        }
    }

    /// Whether any effect would actually modify the buffer.
    pub fn any_enabled(&self) -> bool {
        self.gamma_enabled()
            || self.bloom_enabled()
            || self.aberration_enabled()
            || self.ssao_enabled()
    }

    /// The same settings with screen-space AO removed.
    ///
    /// Used when ray-traced occlusion is active. The two are the same
    /// measurement computed twice, and the traced one is strictly better
    /// informed: its rays see geometry that is off-screen or hidden behind a
    /// nearer surface, which is exactly what a depth buffer cannot report.
    /// Running both darkens every crease twice, and the result reads as a
    /// lighting bug rather than as two effects stacking.
    pub const fn without_ssao(mut self) -> Self {
        self.ssao_radius = 0;
        self.ssao_strength = 0.0;
        self
    }

    /// Whether screen-space AO would run. Public because the choice between it
    /// and the traced version is made by the caller that knows which lighting
    /// path is active.
    pub fn ssao_active(&self) -> bool {
        self.ssao_enabled()
    }

    fn gamma_enabled(&self) -> bool {
        self.gamma.is_finite() && self.gamma > 0.0 && (self.gamma - 1.0).abs() > f32::EPSILON
    }

    fn bloom_enabled(&self) -> bool {
        self.bloom_intensity.is_finite()
            && self.bloom_intensity > 0.0
            && self.bloom_radius > 0
            && self.bloom_threshold.is_finite()
    }

    fn aberration_enabled(&self) -> bool {
        self.aberration_strength.is_finite() && self.aberration_strength.abs() > 0.0
    }

    fn ssao_enabled(&self) -> bool {
        self.ssao_radius > 0 && self.ssao_strength.is_finite() && self.ssao_strength > 0.0
    }
}

#[cfg(test)]
mod traced_interaction_tests {
    use super::*;

    /// `without_ssao` must silence screen-space AO and leave every other effect
    /// alone — the traced path replaces occlusion, not bloom or gamma.
    #[test]
    fn without_ssao_disables_only_occlusion() {
        let demo = PostProcessSettings::demo();
        assert!(demo.ssao_active(), "the demo preset should include SSAO");
        let traced = demo.without_ssao();
        assert!(!traced.ssao_active());
        assert_eq!(traced.gamma, demo.gamma);
        assert_eq!(traced.bloom_intensity, demo.bloom_intensity);
        assert_eq!(traced.bloom_radius, demo.bloom_radius);
        assert_eq!(traced.bloom_threshold, demo.bloom_threshold);
        assert_eq!(traced.aberration_strength, demo.aberration_strength);
        assert!(
            traced.any_enabled(),
            "removing SSAO must not disable the whole post stack"
        );
    }

    #[test]
    fn without_ssao_is_idempotent_and_safe_on_none() {
        let none = PostProcessSettings::none().without_ssao();
        assert!(!none.any_enabled());
        let twice = PostProcessSettings::demo().without_ssao().without_ssao();
        assert!(!twice.ssao_active());
    }
}

/// Perceived luminance, matching the coefficients used elsewhere in the
/// pipeline (ITU-R BT.601).
fn luminance(rgb: [f32; 3]) -> f32 {
    0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
}

fn encode_channel(v: f32) -> u8 {
    if !v.is_finite() {
        return 0;
    }
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Apply `gamma` as an output transform: `c^(1/gamma)`.
///
/// A gamma of exactly 1.0 (or a non-positive / non-finite value, which would
/// otherwise produce NaN across the whole frame) leaves the buffer untouched.
pub fn gamma_correct(fb: &mut FrameBuffer, gamma: f32) {
    if !gamma.is_finite() || gamma <= 0.0 || (gamma - 1.0).abs() <= f32::EPSILON {
        return;
    }
    let inv = 1.0 / gamma;
    for p in &mut fb.pixels {
        for c in p.iter_mut() {
            // Negative inputs have no meaningful power; clamp before powf so the
            // result can never be NaN.
            *c = c.max(0.0).powf(inv);
        }
    }
}

/// Bright-pass, blur, add. `threshold` selects the blooming pixels by
/// luminance, `radius` is the blur extent in cells and `intensity` scales what
/// gets added back.
///
/// The blur is a separable box blur (horizontal then vertical pass), which is
/// O(width*height*radius) rather than O(radius^2) and is visually smooth enough
/// once the result is quantized to glyphs.
pub fn bloom(fb: &mut FrameBuffer, threshold: f32, intensity: f32, radius: u32) {
    if fb.is_empty() || radius == 0 {
        return;
    }
    if !intensity.is_finite() || intensity <= 0.0 || !threshold.is_finite() {
        return;
    }

    // Bright pass.
    let mut bright = FrameBuffer {
        width: fb.width,
        height: fb.height,
        pixels: fb
            .pixels
            .iter()
            .map(|p| {
                if luminance(*p) > threshold {
                    *p
                } else {
                    [0.0; 3]
                }
            })
            .collect(),
    };

    // Nothing bright enough: adding a blurred all-black buffer would be a no-op,
    // so skip the blur work entirely.
    if bright.pixels.iter().all(|p| p[0] <= 0.0 && p[1] <= 0.0 && p[2] <= 0.0) {
        return;
    }

    box_blur_separable(&mut bright, radius);

    for (dst, add) in fb.pixels.iter_mut().zip(bright.pixels.iter()) {
        for c in 0..3 {
            dst[c] += add[c] * intensity;
        }
    }
}

/// Two-pass separable box blur with edge clamping.
fn box_blur_separable(fb: &mut FrameBuffer, radius: u32) {
    if fb.is_empty() || radius == 0 {
        return;
    }
    let r = radius as i32;
    let norm = 1.0 / (2 * r + 1) as f32;

    // Horizontal.
    let src = fb.clone();
    for y in 0..fb.height as i32 {
        for x in 0..fb.width as i32 {
            let mut acc = [0.0f32; 3];
            for dx in -r..=r {
                let s = src.sample_clamped(x + dx, y);
                for c in 0..3 {
                    acc[c] += s[c];
                }
            }
            fb.set(x as u32, y as u32, [acc[0] * norm, acc[1] * norm, acc[2] * norm]);
        }
    }

    // Vertical.
    let src = fb.clone();
    for y in 0..fb.height as i32 {
        for x in 0..fb.width as i32 {
            let mut acc = [0.0f32; 3];
            for dy in -r..=r {
                let s = src.sample_clamped(x, y + dy);
                for c in 0..3 {
                    acc[c] += s[c];
                }
            }
            fb.set(x as u32, y as u32, [acc[0] * norm, acc[1] * norm, acc[2] * norm]);
        }
    }
}

/// Retro CRT fringing: sample red from `strength` cells to the left and blue
/// from `strength` cells to the right, leaving green in place.
///
/// Sampling is edge-clamped, never wrapped — a wrap would smear the right edge
/// of the frame onto the left, which is far more visible than the effect itself.
pub fn chromatic_aberration(fb: &mut FrameBuffer, strength: f32) {
    if fb.is_empty() || !strength.is_finite() || strength == 0.0 {
        return;
    }
    // Round to whole cells: at this resolution one cell IS the smallest visible
    // step, and sub-cell interpolation would just blur the frame.
    let offset = strength.round() as i32;
    if offset == 0 {
        return;
    }

    let src = fb.clone();
    for y in 0..fb.height as i32 {
        for x in 0..fb.width as i32 {
            let r = src.sample_clamped(x - offset, y)[0];
            let g = src.sample_clamped(x, y)[1];
            let b = src.sample_clamped(x + offset, y)[2];
            fb.set(x as u32, y as u32, [r, g, b]);
        }
    }
}

/// Screen-space ambient occlusion approximated from depth alone.
///
/// Heuristic, stated plainly: for each pixel, compare its depth against the
/// neighbours within `radius`. Every neighbour that is *nearer* than the centre
/// by more than a small bias counts as an occluder; the occlusion ratio then
/// scales the pixel down by up to `strength`. Pixels on the far plane (nothing
/// drawn) are skipped so the background never darkens.
///
/// This has no normals and no hemisphere sampling, so it is an approximation,
/// not physically-based AO. At one sample per glyph cell that is the right
/// trade: it darkens creases and contact regions, which is all that survives
/// quantization to ASCII anyway.
pub fn ssao(fb: &mut FrameBuffer, depth: &DepthBuffer, radius: u32, strength: f32) {
    if fb.is_empty() || radius == 0 || !strength.is_finite() || strength <= 0.0 {
        return;
    }
    // A mismatched depth buffer means the readback lagged a resize; darkening
    // by the wrong geometry looks worse than skipping a frame of AO.
    if depth.width != fb.width || depth.height != fb.height {
        return;
    }

    const BIAS: f32 = 0.0005;
    let r = radius as i32;
    let strength = strength.clamp(0.0, 1.0);

    let mut factors = Vec::with_capacity(fb.pixels.len());
    for y in 0..fb.height as i32 {
        for x in 0..fb.width as i32 {
            let center = depth.sample_clamped(x, y);
            // Far plane / empty space: leave it alone.
            if center >= 1.0 {
                factors.push(1.0);
                continue;
            }

            let mut occluders = 0u32;
            let mut samples = 0u32;
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    samples += 1;
                    if depth.sample_clamped(x + dx, y + dy) < center - BIAS {
                        occluders += 1;
                    }
                }
            }

            let ratio = if samples == 0 {
                0.0
            } else {
                occluders as f32 / samples as f32
            };
            factors.push(1.0 - ratio * strength);
        }
    }

    for (p, f) in fb.pixels.iter_mut().zip(factors.iter()) {
        for c in p.iter_mut() {
            *c *= *f;
        }
    }
}

/// Run the enabled effects in a fixed order:
///
/// 1. **SSAO** — occlusion is a property of the geometry, so it must darken the
///    surface before anything reads its brightness.
/// 2. **Bloom** — bright-pass runs after AO, otherwise light would bleed out of
///    areas that AO is about to darken.
/// 3. **Chromatic aberration** — a lens artefact, applied to the composed image.
/// 4. **Gamma** — an output transform, so it goes last by definition.
///
/// `depth` may be `None`, in which case SSAO is skipped.
pub fn apply_all(fb: &mut FrameBuffer, depth: Option<&DepthBuffer>, s: &PostProcessSettings) {
    if fb.is_empty() {
        return;
    }
    if let (Some(depth), true) = (depth, s.ssao_enabled()) {
        ssao(fb, depth, s.ssao_radius, s.ssao_strength);
    }
    if s.bloom_enabled() {
        bloom(fb, s.bloom_threshold, s.bloom_intensity, s.bloom_radius);
    }
    if s.aberration_enabled() {
        chromatic_aberration(fb, s.aberration_strength);
    }
    if s.gamma_enabled() {
        gamma_correct(fb, s.gamma);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fb_from(width: u32, height: u32, values: &[[f32; 3]]) -> FrameBuffer {
        assert_eq!(values.len(), (width * height) as usize);
        FrameBuffer { width, height, pixels: values.to_vec() }
    }

    fn grey(v: f32) -> [f32; 3] {
        [v, v, v]
    }

    // --- FrameBuffer basics ---

    #[test]
    fn from_rgba8_length_mismatch_is_an_error() {
        let px = vec![[0u8, 0, 0, 255]; 3];
        assert!(FrameBuffer::from_rgba8(&px, 2, 2).is_err());
        assert!(FrameBuffer::from_rgba8(&px, 3, 1).is_ok());
    }

    #[test]
    fn rgba8_round_trip_is_within_one_step() {
        let original: Vec<[u8; 4]> = (0..16u8)
            .map(|i| [i * 17, 255 - i * 17, i * 3, 255])
            .collect();
        let fb = FrameBuffer::from_rgba8(&original, 4, 4).unwrap();
        let back = fb.to_rgba8();
        assert_eq!(back.len(), original.len());
        for (a, b) in original.iter().zip(back.iter()) {
            for c in 0..3 {
                assert!(
                    (a[c] as i32 - b[c] as i32).abs() <= 1,
                    "channel drifted: {a:?} -> {b:?}"
                );
            }
            assert_eq!(b[3], 255);
        }
    }

    #[test]
    fn to_rgba8_clamps_overshoot_and_non_finite() {
        let fb = fb_from(3, 1, &[[2.0, -1.0, 0.5], [f32::NAN, 1.0, 0.0], grey(f32::INFINITY)]);
        let out = fb.to_rgba8();
        assert_eq!(out[0], [255, 0, 128, 255]);
        assert_eq!(out[1], [0, 255, 0, 255]);
        assert_eq!(out[2], [0, 0, 0, 255]);
    }

    #[test]
    fn get_and_sample_clamped_agree_inside_and_clamp_outside() {
        let fb = fb_from(2, 2, &[grey(0.0), grey(0.25), grey(0.5), grey(1.0)]);
        assert_eq!(fb.get(1, 1), Some(grey(1.0)));
        assert_eq!(fb.get(2, 0), None);
        assert_eq!(fb.sample_clamped(-5, -5), grey(0.0));
        assert_eq!(fb.sample_clamped(99, 99), grey(1.0));
    }

    #[test]
    fn empty_buffer_samples_black_without_panicking() {
        let fb = FrameBuffer::new(0, 0);
        assert!(fb.is_empty());
        assert_eq!(fb.sample_clamped(0, 0), [0.0; 3]);
        assert_eq!(fb.get(0, 0), None);
        assert!(fb.to_rgba8().is_empty());
    }

    // --- Gamma ---

    #[test]
    fn gamma_of_one_leaves_the_buffer_bit_identical() {
        let original = fb_from(2, 1, &[grey(0.25), [0.1, 0.7, 0.9]]);
        let mut fb = original.clone();
        gamma_correct(&mut fb, 1.0);
        assert_eq!(fb, original);
    }

    #[test]
    fn gamma_two_maps_a_quarter_to_a_half() {
        // 0.25^(1/2) == 0.5 exactly.
        let mut fb = fb_from(1, 1, &[grey(0.25)]);
        gamma_correct(&mut fb, 2.0);
        for c in fb.pixels[0] {
            assert!((c - 0.5).abs() < 1e-6, "got {c}");
        }
    }

    #[test]
    fn invalid_gamma_is_a_no_op() {
        let original = fb_from(1, 1, &[grey(0.5)]);
        for bad in [0.0, -2.0, f32::NAN, f32::INFINITY] {
            let mut fb = original.clone();
            gamma_correct(&mut fb, bad);
            assert_eq!(fb, original, "gamma {bad} should be ignored");
        }
    }

    #[test]
    fn gamma_never_produces_nan_from_negative_input() {
        let mut fb = fb_from(1, 1, &[[-0.5, 0.0, 0.25]]);
        gamma_correct(&mut fb, 2.2);
        assert!(fb.pixels[0].iter().all(|c| c.is_finite()));
    }

    // --- Bloom ---

    #[test]
    fn bloom_is_a_no_op_below_the_threshold() {
        let original = fb_from(3, 3, &[grey(0.1); 9]);
        let mut fb = original.clone();
        bloom(&mut fb, 0.9, 1.0, 2);
        assert_eq!(fb, original, "nothing exceeds the threshold, nothing may change");
    }

    #[test]
    fn bloom_is_a_no_op_when_disabled() {
        let original = fb_from(3, 3, &[grey(1.0); 9]);
        for (threshold, intensity, radius) in [(0.0, 0.0, 2), (0.0, 1.0, 0)] {
            let mut fb = original.clone();
            bloom(&mut fb, threshold, intensity, radius);
            assert_eq!(fb, original);
        }
    }

    #[test]
    fn bloom_spreads_a_bright_centre_into_its_neighbours() {
        let mut pixels = vec![grey(0.0); 9];
        pixels[4] = grey(1.0); // centre of a 3x3
        let mut fb = fb_from(3, 3, &pixels);
        bloom(&mut fb, 0.5, 1.0, 1);

        // The dark corner must have gained light from the centre...
        assert!(fb.pixels[0][0] > 0.0, "bloom did not reach the corner");
        // ...and the centre must still be the brightest pixel.
        let centre = fb.pixels[4][0];
        assert!(fb.pixels.iter().all(|p| p[0] <= centre + 1e-6));
    }

    #[test]
    fn bloom_preserves_dimensions() {
        let mut fb = fb_from(4, 2, &[grey(1.0); 8]);
        bloom(&mut fb, 0.1, 0.5, 3);
        assert_eq!((fb.width, fb.height), (4, 2));
        assert_eq!(fb.len(), 8);
    }

    #[test]
    fn box_blur_of_a_flat_buffer_is_the_same_flat_buffer() {
        let original = fb_from(4, 4, &[grey(0.4); 16]);
        let mut fb = original.clone();
        box_blur_separable(&mut fb, 2);
        for p in &fb.pixels {
            for c in *p {
                assert!((c - 0.4).abs() < 1e-5, "edge clamping should preserve a flat field, got {c}");
            }
        }
    }

    // --- Chromatic aberration ---

    #[test]
    fn aberration_shifts_red_left_and_blue_right() {
        // A single white pixel in the middle of a 5x1 strip.
        let mut pixels = vec![[0.0f32; 3]; 5];
        pixels[2] = [1.0, 1.0, 1.0];
        let mut fb = fb_from(5, 1, &pixels);
        chromatic_aberration(&mut fb, 1.0);

        // Red is sampled from x-1, so the white pixel's red shows up at x=3.
        assert_eq!(fb.pixels[3][0], 1.0, "red should move right by one (sampled from the left)");
        // Blue is sampled from x+1, so it shows up at x=1.
        assert_eq!(fb.pixels[1][2], 1.0, "blue should move left by one (sampled from the right)");
        // Green stays put.
        assert_eq!(fb.pixels[2][1], 1.0);
    }

    #[test]
    fn aberration_clamps_instead_of_wrapping() {
        // Bright pixel at the right edge; with wrapping its red would appear at x=0.
        let mut pixels = vec![[0.0f32; 3]; 4];
        pixels[3] = [1.0, 0.0, 0.0];
        let mut fb = fb_from(4, 1, &pixels);
        chromatic_aberration(&mut fb, 2.0);
        assert_eq!(fb.pixels[0][0], 0.0, "colour must not wrap around the edge");
    }

    #[test]
    fn aberration_is_a_no_op_when_disabled_or_sub_cell() {
        let original = fb_from(4, 1, &[grey(0.5), grey(0.2), grey(0.9), grey(0.1)]);
        for s in [0.0, 0.2, -0.4, f32::NAN] {
            let mut fb = original.clone();
            chromatic_aberration(&mut fb, s);
            assert_eq!(fb, original, "strength {s} should not change anything");
        }
    }

    // --- SSAO ---

    #[test]
    fn ssao_darkens_a_pixel_whose_neighbours_are_all_nearer() {
        let mut fb = fb_from(3, 3, &[grey(1.0); 9]);
        // Centre sits in a depth valley: everything around it is much nearer.
        let mut depth = vec![0.1f32; 9];
        depth[4] = 0.9;
        let depth = DepthBuffer::from_slice(&depth, 3, 3).unwrap();

        ssao(&mut fb, &depth, 1, 1.0);
        assert!(fb.pixels[4][0] < 0.2, "fully occluded centre should be dark, got {}", fb.pixels[4][0]);
    }

    #[test]
    fn ssao_leaves_a_flat_surface_untouched() {
        let original = fb_from(3, 3, &[grey(0.8); 9]);
        let mut fb = original.clone();
        let depth = DepthBuffer::from_slice(&[0.5f32; 9], 3, 3).unwrap();
        ssao(&mut fb, &depth, 1, 1.0);
        assert_eq!(fb, original, "no depth variation means no occlusion");
    }

    #[test]
    fn ssao_skips_the_far_plane() {
        let original = fb_from(2, 1, &[grey(0.5), grey(0.5)]);
        let mut fb = original.clone();
        // Left pixel is empty space (far), right one is near.
        let depth = DepthBuffer::from_slice(&[1.0, 0.2], 2, 1).unwrap();
        ssao(&mut fb, &depth, 1, 1.0);
        assert_eq!(fb.pixels[0], original.pixels[0], "background must not darken");
    }

    #[test]
    fn ssao_is_a_no_op_when_disabled_or_mismatched() {
        let original = fb_from(2, 2, &[grey(1.0); 4]);
        let depth = DepthBuffer::from_slice(&[0.5, 0.1, 0.1, 0.1], 2, 2).unwrap();

        for (radius, strength) in [(0u32, 1.0f32), (1, 0.0), (1, f32::NAN)] {
            let mut fb = original.clone();
            ssao(&mut fb, &depth, radius, strength);
            assert_eq!(fb, original, "radius {radius} strength {strength} should be inert");
        }

        // A depth buffer of the wrong size is ignored rather than misapplied.
        let wrong = DepthBuffer::new(4, 4);
        let mut fb = original.clone();
        ssao(&mut fb, &wrong, 2, 1.0);
        assert_eq!(fb, original);
    }

    #[test]
    fn depth_buffer_length_is_checked_and_non_finite_reads_as_far() {
        assert!(DepthBuffer::from_slice(&[0.0; 3], 2, 2).is_err());
        let d = DepthBuffer::from_slice(&[f32::NAN, 0.5, 0.5, 0.5], 2, 2).unwrap();
        assert_eq!(d.sample_clamped(0, 0), 1.0);
        assert_eq!(DepthBuffer::new(0, 0).sample_clamped(0, 0), 1.0);
    }

    // --- apply_all ---

    #[test]
    fn apply_all_with_no_settings_changes_nothing() {
        let original = fb_from(3, 3, &[grey(0.6); 9]);
        let mut fb = original.clone();
        apply_all(&mut fb, None, &PostProcessSettings::none());
        assert_eq!(fb, original);
        assert!(!PostProcessSettings::none().any_enabled());
        assert!(PostProcessSettings::demo().any_enabled());
    }

    #[test]
    fn apply_all_ordering_is_observable() {
        // SSAO darkens before bloom's bright pass, so running the pair together
        // must not be brighter than bloom alone on the same input.
        let pixels = vec![grey(1.0); 9];
        let mut depth = vec![0.9f32; 9];
        depth[4] = 0.95; // centre slightly further: its neighbours occlude it
        let depth = DepthBuffer::from_slice(&depth, 3, 3).unwrap();

        let mut with_ssao = fb_from(3, 3, &pixels);
        apply_all(
            &mut with_ssao,
            Some(&depth),
            &PostProcessSettings {
                bloom_threshold: 0.5,
                bloom_intensity: 1.0,
                bloom_radius: 1,
                ssao_radius: 1,
                ssao_strength: 1.0,
                ..PostProcessSettings::none()
            },
        );

        let mut bloom_only = fb_from(3, 3, &pixels);
        apply_all(
            &mut bloom_only,
            None,
            &PostProcessSettings {
                bloom_threshold: 0.5,
                bloom_intensity: 1.0,
                bloom_radius: 1,
                ..PostProcessSettings::none()
            },
        );

        assert!(
            with_ssao.pixels[4][0] < bloom_only.pixels[4][0],
            "AO must be applied before the bright pass"
        );
    }

    #[test]
    fn apply_all_never_produces_non_finite_values_on_hostile_settings() {
        let hostile = PostProcessSettings {
            gamma: f32::NAN,
            bloom_threshold: f32::NEG_INFINITY,
            bloom_intensity: f32::INFINITY,
            bloom_radius: 999,
            aberration_strength: f32::NAN,
            ssao_radius: 999,
            ssao_strength: -5.0,
        };
        let depth = DepthBuffer::from_slice(&[0.5, 0.1, 0.9, 0.2], 2, 2).unwrap();
        let mut fb = fb_from(2, 2, &[grey(0.5), grey(1.0), [-1.0, 2.0, 0.0], grey(0.25)]);
        apply_all(&mut fb, Some(&depth), &hostile);
        for p in &fb.pixels {
            for c in *p {
                assert!(c.is_finite(), "non-finite pixel escaped: {p:?}");
            }
        }
    }

    #[test]
    fn effects_survive_degenerate_buffer_sizes() {
        for (w, h) in [(0u32, 0u32), (1, 1), (1, 5), (5, 1)] {
            let mut fb = FrameBuffer::new(w, h);
            for p in fb.pixels.iter_mut() {
                *p = grey(0.9);
            }
            let depth = DepthBuffer::new(w, h);
            // Radii deliberately larger than the buffer.
            apply_all(
                &mut fb,
                Some(&depth),
                &PostProcessSettings {
                    gamma: 2.2,
                    bloom_threshold: 0.1,
                    bloom_intensity: 1.0,
                    bloom_radius: 10,
                    aberration_strength: 7.0,
                    ssao_radius: 10,
                    ssao_strength: 1.0,
                },
            );
            assert_eq!((fb.width, fb.height), (w, h));
            assert_eq!(fb.len(), (w * h) as usize);
            assert!(fb.pixels.iter().flatten().all(|c| c.is_finite()));
        }
    }
}
