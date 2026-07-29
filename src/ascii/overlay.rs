// 2D ASCII overlay layer (ROADMAP 3.3) — a character buffer drawn on top of
// the 3D scene. Pure CPU logic: no GPU, no wgpu types, no external crates.
//
// Where it sits in the pipeline:
//   scene pass -> low-res RGBA readback -> per-cell (glyph_index, rgb)
//   -> Overlay::composite_over(..)  <-- this module
//   -> InstanceData for the composite pass
// The overlay has exactly the same dimensions as the ASCII cell grid (one cell
// per low-res scene pixel), so compositing is a straight per-cell overwrite and
// needs no scaling or sampling.
//
// Design notes:
// - Transparency model: every cell carries an `opaque` flag. `opaque == false`
//   means "untouched, let the scene through"; `opaque == true` means "this cell
//   is UI, draw it instead of the scene". There is no alpha blending — ASCII
//   cells are discrete, a cell is either UI or scene. That keeps compositing
//   exact and testable.
// - Background fill for panels: OverlayCell has no separate background colour
//   channel (the composite pass draws one tinted glyph per cell, nothing else).
//   A "panel background" is therefore just an opaque cell whose glyph is a
//   shading block — `Overlay::background_cell` builds one from the densest
//   block glyph ('█', the last entry of ALL_CHARS today). Filling with
//   `SPACE_INDEX` instead gives a hard black hole that masks the scene without
//   adding ink. Both go through `fill_rect`.
// - Glyph mapping seam: the current atlas (ascii/glyph_atlas.rs) contains only
//   14 shading glyphs and NO letters, so text cannot use ASCII arithmetic such
//   as `c as u32 - 32`. Instead the char -> glyph-index policy is injected as a
//   `GlyphMapFn` stored on the Overlay (defaulting to `default_glyph_of`). When
//   a real font atlas lands (ROADMAP 3.1 "TTF font loading"), UI code keeps
//   working unchanged: only the injected function is swapped. Nothing in this
//   module assumes anything about glyph indices except that `SPACE_INDEX` is
//   blank.
// - Y grows downward (row 0 is the top row), matching the readback buffer and
//   `AsciiProcessor::pixels_to_instances`.
// - Every entry point is total: out-of-range coordinates, zero-sized overlays
//   and zero-sized rects are silently ignored, never a panic. UI code runs
//   every frame and must not be able to crash the renderer on a resize race.

use crate::ascii::glyph_atlas::{ALL_CHARS, BRIGHTNESS_RAMP, SPACE_INDEX};

/// A char -> glyph-index mapping policy.
///
/// A plain `fn` pointer (not a boxed closure) so `Overlay` stays `Copy`-cheap
/// to clone, `Debug`, and allocation-free.
pub type GlyphMapFn = fn(char) -> u32;

/// One cell of the scene layer as consumed by [`Overlay::composite_over`]:
/// `(glyph_index, linear rgb)`.
pub type SceneCell = (u32, [f32; 3]);

// Estimated ink coverage of unknown characters, used by `default_glyph_of` to
// pick a ramp glyph of comparable visual weight.
const DENSITY_NARROW: f32 = 0.3;
const DENSITY_LOWER: f32 = 0.55;
const DENSITY_UPPER: f32 = 0.75;
const DENSITY_PUNCT: f32 = 0.45;
const DENSITY_OTHER: f32 = 0.6;

/// A single overlay cell.
///
/// `opaque == false` is the transparent/untouched state produced by
/// [`Overlay::clear`]; the other fields are then meaningless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayCell {
    /// Index into the glyph atlas.
    pub glyph_index: u32,
    /// Linear RGB tint applied to the glyph.
    pub color: [f32; 3],
    /// `true` = this cell is UI and replaces the scene cell underneath.
    pub opaque: bool,
}

impl OverlayCell {
    /// A transparent cell: the scene shows through.
    pub fn transparent() -> Self {
        Self {
            glyph_index: SPACE_INDEX,
            color: [0.0, 0.0, 0.0],
            opaque: false,
        }
    }

    /// An opaque cell drawing `glyph_index` tinted with `color`.
    pub fn new(glyph_index: u32, color: [f32; 3]) -> Self {
        Self {
            glyph_index,
            color,
            opaque: true,
        }
    }
}

impl Default for OverlayCell {
    /// Same as [`OverlayCell::transparent`].
    fn default() -> Self {
        Self::transparent()
    }
}

/// Rough ink coverage (0.0 = blank, 1.0 = solid) of a character that the atlas
/// has no dedicated glyph for.
///
/// Deliberately coarse: it only has to order characters by visual weight so the
/// fallback text is readable as *text-shaped noise* rather than uniform blocks.
fn ink_density(c: char) -> f32 {
    if c.is_control() || c.is_whitespace() {
        return 0.0;
    }
    // Narrow / low-ink shapes first: these literals must win over the
    // digit/letter classes below.
    if matches!(
        c,
        '\'' | '"' | '`' | ',' | ';' | '!' | '|' | 'i' | 'l' | 'j' | 'I' | '1'
    ) {
        return DENSITY_NARROW;
    }
    if c.is_ascii_digit() || c.is_ascii_lowercase() {
        return DENSITY_LOWER;
    }
    if c.is_ascii_uppercase() {
        return DENSITY_UPPER;
    }
    if c.is_ascii_punctuation() {
        return DENSITY_PUNCT;
    }
    DENSITY_OTHER
}

/// Map a 0.0..=1.0 density onto the brightness ramp portion of the atlas.
///
/// Uses only `BRIGHTNESS_RAMP` (' ' '.' ':' '-' '=' '+' '*' '#' '%' '@'), which
/// is a prefix of `ALL_CHARS` with identical ordering, so a ramp position is
/// also an atlas index. The block glyphs at the tail of `ALL_CHARS` are
/// intentionally excluded: they are reserved for panel backgrounds, and solid
/// blocks would make fallback text unreadable. This is why the module does not
/// reuse `glyph_atlas::brightness_to_index` (which spans all 14 glyphs).
fn ramp_index(density: f32) -> u32 {
    // `saturating_sub` instead of `- 1`: keeps this function total even if
    // `BRIGHTNESS_RAMP` is ever emptied while the atlas is reworked (ROADMAP
    // 3.1) — an empty ramp collapses to index 0, which is `SPACE_INDEX`.
    // A NaN density also lands on 0: `clamp` passes NaN through and Rust's
    // float -> int cast saturates, so there is no panic and no UB.
    let last = BRIGHTNESS_RAMP.len().saturating_sub(1) as u32;
    let clamped = density.clamp(0.0, 1.0);
    ((clamped * last as f32).round() as u32).min(last)
}

/// Default char -> glyph-index policy.
///
/// 1. Characters the atlas actually contains (`ALL_CHARS`) map to their own
///    glyph — so `'#'`, `':'`, `'-'`, `'+'`, the blocks and `' '` are exact.
/// 2. Everything else (letters, digits, unknown Unicode) is approximated by
///    estimated ink density onto the brightness ramp. Whitespace and control
///    characters — including `'\n'`, which [`Overlay::draw_text`] does *not*
///    treat specially — become `SPACE_INDEX`.
///
/// Replace it via [`Overlay::with_glyph_map`] / [`Overlay::set_glyph_map`] once
/// a font atlas with real letters exists.
pub fn default_glyph_of(c: char) -> u32 {
    if let Some(index) = ALL_CHARS.iter().position(|&atlas_char| atlas_char == c) {
        return index as u32;
    }
    ramp_index(ink_density(c))
}

/// A character layer the size of the ASCII cell grid, composited over the scene.
///
/// Typical per-frame use: [`clear`](Overlay::clear), draw HUD/menu/console,
/// then [`composite_over`](Overlay::composite_over) the scene cells.
#[derive(Clone, Debug)]
pub struct Overlay {
    cols: u32,
    rows: u32,
    /// Row-major, exactly `cols * rows` entries.
    cells: Vec<OverlayCell>,
    /// Number of currently opaque cells; keeps `is_empty` O(1).
    opaque_count: usize,
    glyph_of: GlyphMapFn,
}

impl Overlay {
    /// Create a fully transparent overlay using [`default_glyph_of`].
    ///
    /// `cols` or `rows` of 0 yields a valid, permanently empty overlay.
    pub fn new(cols: u32, rows: u32) -> Self {
        Self::with_glyph_map(cols, rows, default_glyph_of)
    }

    /// Create a fully transparent overlay with a custom char -> glyph policy.
    pub fn with_glyph_map(cols: u32, rows: u32, glyph_of: GlyphMapFn) -> Self {
        let count = (cols as usize).saturating_mul(rows as usize);
        Self {
            cols,
            rows,
            cells: vec![OverlayCell::transparent(); count],
            opaque_count: 0,
            glyph_of,
        }
    }

    /// Number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Replace the char -> glyph-index policy (e.g. after loading a font atlas).
    ///
    /// Only affects subsequent drawing; already-written cells keep their glyphs.
    pub fn set_glyph_map(&mut self, glyph_of: GlyphMapFn) {
        self.glyph_of = glyph_of;
    }

    /// Resolve a character through the active glyph policy.
    ///
    /// Exposed so UI code can build [`OverlayCell`]s for [`Overlay::fill_rect`]
    /// without duplicating the mapping.
    pub fn glyph_index_of(&self, c: char) -> u32 {
        (self.glyph_of)(c)
    }

    /// An opaque "panel background" cell: the densest block glyph available,
    /// tinted with `color`. Pass it to [`Overlay::fill_rect`].
    pub fn background_cell(&self, color: [f32; 3]) -> OverlayCell {
        OverlayCell::new(self.glyph_index_of('█'), color)
    }

    /// Reset every cell to transparent. Called once per frame before drawing.
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = OverlayCell::transparent();
        }
        self.opaque_count = 0;
    }

    /// `true` when no opaque cell exists, i.e. compositing would be a no-op.
    ///
    /// Writing a transparent cell does not make an overlay non-empty, and
    /// overwriting the last opaque cell with a transparent one makes it empty
    /// again — "empty" is defined by visible content, not by write history.
    pub fn is_empty(&self) -> bool {
        self.opaque_count == 0
    }

    /// Row-major index of `(col, row)`, or `None` if out of range.
    fn index(&self, col: u32, row: u32) -> Option<usize> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        Some(row as usize * self.cols as usize + col as usize)
    }

    /// Write one cell. Out-of-range coordinates are silently ignored.
    pub fn set_cell(&mut self, col: u32, row: u32, cell: OverlayCell) {
        let Some(index) = self.index(col, row) else {
            return;
        };
        let was_opaque = self.cells[index].opaque;
        self.cells[index] = cell;
        match (was_opaque, cell.opaque) {
            (false, true) => self.opaque_count += 1,
            (true, false) => self.opaque_count -= 1,
            _ => {}
        }
    }

    /// Read one cell, or `None` if out of range.
    pub fn cell(&self, col: u32, row: u32) -> Option<&OverlayCell> {
        self.index(col, row).map(|index| &self.cells[index])
    }

    /// All cells, row-major, exactly `cols * rows` entries.
    pub fn cells(&self) -> &[OverlayCell] {
        &self.cells
    }

    /// Resize the overlay (window resize). Contents are dropped: the overlay is
    /// fully redrawn every frame, so preserving them would only ever show a
    /// stale, misaligned UI.
    pub fn resize(&mut self, cols: u32, rows: u32) {
        let count = (cols as usize).saturating_mul(rows as usize);
        self.cols = cols;
        self.rows = rows;
        self.cells.clear();
        self.cells.resize(count, OverlayCell::transparent());
        self.opaque_count = 0;
    }

    /// Draw `text` starting at `(col, row)`, one cell per `char`.
    ///
    /// Clipping / edge behaviour (never panics, never wraps):
    /// - a `row` outside the overlay draws nothing;
    /// - drawing stops at the right edge — the remaining characters are
    ///   discarded, they do **not** continue on the next row;
    /// - `'\n'`, `'\t'` and other control characters are not interpreted: they
    ///   consume one cell and are rendered by the glyph policy (blank under
    ///   [`default_glyph_of`]). Callers that need multiple lines call
    ///   `draw_text` once per line.
    ///
    /// Characters are mapped through the injected policy, so this works both
    /// with today's 14-glyph shading atlas and with a future font atlas.
    pub fn draw_text(&mut self, col: u32, row: u32, text: &str, color: [f32; 3]) {
        if row >= self.rows || col >= self.cols {
            return;
        }
        for (offset, c) in text.chars().enumerate() {
            let Ok(offset) = u32::try_from(offset) else {
                return;
            };
            let Some(x) = col.checked_add(offset) else {
                return;
            };
            if x >= self.cols {
                return;
            }
            let glyph_index = self.glyph_index_of(c);
            self.set_cell(x, row, OverlayCell::new(glyph_index, color));
        }
    }

    /// Compute the clipped `[x0, x1) x [y0, y1)` region of a logical rect, or
    /// `None` if nothing of it is visible.
    ///
    /// Saturating arithmetic: `col + w` cannot overflow into a panic even for
    /// `w == u32::MAX`.
    fn clip_rect(&self, col: u32, row: u32, w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
        if w == 0 || h == 0 || col >= self.cols || row >= self.rows {
            return None;
        }
        let x1 = col.saturating_add(w).min(self.cols);
        let y1 = row.saturating_add(h).min(self.rows);
        if col >= x1 || row >= y1 {
            return None;
        }
        Some((col, row, x1, y1))
    }

    /// Fill a `w x h` rect whose top-left corner is `(col, row)` with `cell`.
    ///
    /// Clipped against the overlay bounds; a zero `w` or `h`, or a fully
    /// off-screen rect, draws nothing. Use with
    /// [`Overlay::background_cell`] for panel backgrounds, or with a
    /// `SPACE_INDEX` cell to mask the scene without adding ink.
    pub fn fill_rect(&mut self, col: u32, row: u32, w: u32, h: u32, cell: OverlayCell) {
        let Some((x0, y0, x1, y1)) = self.clip_rect(col, row, w, h) else {
            return;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                self.set_cell(x, y, cell);
            }
        }
    }

    /// Draw the 1-cell border of a `w x h` rect, leaving the interior untouched.
    ///
    /// The atlas has no line-drawing characters, so the border is built from the
    /// closest available shapes: `'+'` at the corners, `'-'` along the top and
    /// bottom edges, `':'` along the left and right edges. All three go through
    /// the glyph policy, so a future font atlas can supply real box-drawing
    /// glyphs by mapping those characters differently.
    ///
    /// Degenerate sizes behave sensibly: `1 x 1` is a single `'+'`, `1 x h` is a
    /// vertical line capped with `'+'`, `w x 1` a horizontal one. `w` or `h` of
    /// 0 draws nothing. The rect is clipped, so a border edge lying outside the
    /// overlay simply disappears.
    pub fn draw_box(&mut self, col: u32, row: u32, w: u32, h: u32, color: [f32; 3]) {
        let Some((x0, y0, x1, y1)) = self.clip_rect(col, row, w, h) else {
            return;
        };
        // Logical (unclipped) far edges, so clipping cannot turn an edge cell
        // into a corner.
        let last_col = col.saturating_add(w - 1);
        let last_row = row.saturating_add(h - 1);

        // Real box-drawing characters: each corner is a distinct glyph, so the
        // frame closes properly instead of showing the gaps that '+'/'-'/':'
        // substitutes leave. A glyph map without them falls back through
        // `glyph_index_of`, so this stays correct on a minimal atlas too.
        let horizontal = OverlayCell::new(self.glyph_index_of('─'), color);
        let vertical = OverlayCell::new(self.glyph_index_of('│'), color);
        let top_left = OverlayCell::new(self.glyph_index_of('┌'), color);
        let top_right = OverlayCell::new(self.glyph_index_of('┐'), color);
        let bottom_left = OverlayCell::new(self.glyph_index_of('└'), color);
        let bottom_right = OverlayCell::new(self.glyph_index_of('┘'), color);

        for y in y0..y1 {
            let on_top = y == row;
            let on_bottom = y == last_row;
            for x in x0..x1 {
                let on_left = x == col;
                let on_right = x == last_col;
                let cell = match (on_top, on_bottom, on_left, on_right) {
                    (true, _, true, _) => top_left,
                    (true, _, _, true) => top_right,
                    (_, true, true, _) => bottom_left,
                    (_, true, _, true) => bottom_right,
                    (true, _, _, _) | (_, true, _, _) => horizontal,
                    (_, _, true, _) | (_, _, _, true) => vertical,
                    // Interior: not part of the border, leave untouched.
                    _ => continue,
                };
                self.set_cell(x, y, cell);
            }
        }
    }

    /// Composite this overlay over the scene's per-cell `(glyph_index, rgb)`.
    ///
    /// Opaque overlay cells overwrite the corresponding scene entry; everything
    /// else is left exactly as it was.
    ///
    /// The scene slice is expected to be `cols * rows` entries in the same
    /// row-major order, but a mismatched length is tolerated rather than
    /// asserted: a window resize can land between the readback and the
    /// composite, and dropping the UI for one frame beats panicking. A shorter
    /// slice is composited up to its length; a longer slice keeps its tail.
    pub fn composite_over(&self, scene: &mut [SceneCell]) {
        if self.opaque_count == 0 {
            return;
        }
        for (overlay_cell, scene_cell) in self.cells.iter().zip(scene.iter_mut()) {
            if overlay_cell.opaque {
                *scene_cell = (overlay_cell.glyph_index, overlay_cell.color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
    const RED: [f32; 3] = [1.0, 0.0, 0.0];

    /// Glyph index used by the test policy below. Chosen so that no character
    /// the tests feed it maps here under `default_glyph_of` (the default only
    /// ever returns 12 for '▓'), making "was the injected map used?" decisive.
    const CUSTOM_GLYPH: u32 = 12;

    /// Test policy: every char maps to `CUSTOM_GLYPH`.
    fn constant_glyph_of(_c: char) -> u32 {
        CUSTOM_GLYPH
    }

    #[test]
    fn new_is_empty_and_correctly_sized() {
        let overlay = Overlay::new(6, 4);
        assert_eq!(overlay.cols(), 6);
        assert_eq!(overlay.rows(), 4);
        assert_eq!(overlay.cells().len(), 24);
        assert!(overlay.is_empty());
        assert!(overlay.cells().iter().all(|c| !c.opaque));
        // The default cell is the transparent one.
        assert_eq!(OverlayCell::default(), OverlayCell::transparent());
        assert_eq!(OverlayCell::default().glyph_index, SPACE_INDEX);
    }

    #[test]
    fn set_and_get_round_trip() {
        let mut overlay = Overlay::new(4, 3);
        let cell = OverlayCell::new(5, RED);
        overlay.set_cell(2, 1, cell);
        assert_eq!(overlay.cell(2, 1), Some(&cell));
        // Row-major index = row * cols + col = 1 * 4 + 2 = 6.
        assert_eq!(overlay.cells()[6], cell);
        assert!(!overlay.is_empty());
        // Neighbours untouched.
        assert!(!overlay.cell(1, 1).unwrap().opaque);
        assert!(!overlay.cell(2, 0).unwrap().opaque);
    }

    #[test]
    fn out_of_bounds_access_is_safe() {
        let mut overlay = Overlay::new(4, 3);
        overlay.set_cell(4, 0, OverlayCell::new(1, WHITE)); // col == cols
        overlay.set_cell(0, 3, OverlayCell::new(1, WHITE)); // row == rows
        overlay.set_cell(u32::MAX, u32::MAX, OverlayCell::new(1, WHITE));
        assert!(overlay.is_empty());
        assert!(overlay.cells().iter().all(|c| !c.opaque));

        assert!(overlay.cell(4, 0).is_none());
        assert!(overlay.cell(0, 3).is_none());
        assert!(overlay.cell(u32::MAX, 1).is_none());
        assert!(overlay.cell(3, 2).is_some());
    }

    #[test]
    fn clear_resets_is_empty() {
        let mut overlay = Overlay::new(3, 2);
        overlay.draw_text(0, 0, "ab", WHITE);
        assert!(!overlay.is_empty());
        overlay.clear();
        assert!(overlay.is_empty());
        assert_eq!(overlay.cells().len(), 6);
        assert!(overlay.cells().iter().all(|c| *c == OverlayCell::transparent()));
    }

    #[test]
    fn is_empty_counts_opaque_cells_only() {
        let mut overlay = Overlay::new(3, 3);
        overlay.set_cell(0, 0, OverlayCell::transparent());
        assert!(overlay.is_empty(), "writing a transparent cell is not content");

        overlay.set_cell(1, 1, OverlayCell::new(3, WHITE));
        assert!(!overlay.is_empty());
        // Overwriting the same cell twice must not double-count.
        overlay.set_cell(1, 1, OverlayCell::new(4, RED));
        overlay.set_cell(1, 1, OverlayCell::transparent());
        assert!(overlay.is_empty());
    }

    #[test]
    fn draw_text_lands_expected_cells_in_order() {
        let mut overlay = Overlay::new(8, 2);
        overlay.draw_text(1, 1, "#@:", RED);
        // '#', '@' and ':' all exist in the atlas -> exact indices 7, 9, 2.
        assert_eq!(overlay.cell(1, 1), Some(&OverlayCell::new(7, RED)));
        assert_eq!(overlay.cell(2, 1), Some(&OverlayCell::new(9, RED)));
        assert_eq!(overlay.cell(3, 1), Some(&OverlayCell::new(2, RED)));
        // Nothing before the start, nothing after the end, nothing on row 0.
        assert!(!overlay.cell(0, 1).unwrap().opaque);
        assert!(!overlay.cell(4, 1).unwrap().opaque);
        assert!(overlay.cells()[0..8].iter().all(|c| !c.opaque));
    }

    #[test]
    fn draw_text_clips_at_right_edge_without_wrapping() {
        let mut overlay = Overlay::new(4, 2);
        overlay.draw_text(2, 0, "####", WHITE);
        // Only cols 2 and 3 fit.
        assert!(overlay.cell(2, 0).unwrap().opaque);
        assert!(overlay.cell(3, 0).unwrap().opaque);
        // Row 1 must stay empty — no wrapping.
        for col in 0..4 {
            assert!(!overlay.cell(col, 1).unwrap().opaque, "wrapped into row 1");
        }
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 2);
    }

    #[test]
    fn draw_text_out_of_range_is_a_noop() {
        let mut overlay = Overlay::new(4, 2);
        overlay.draw_text(0, 2, "###", WHITE); // row == rows
        overlay.draw_text(0, u32::MAX, "###", WHITE);
        overlay.draw_text(4, 0, "###", WHITE); // col == cols
        overlay.draw_text(u32::MAX, 0, "###", WHITE); // would overflow col + i
        assert!(overlay.is_empty());
    }

    #[test]
    fn draw_text_on_empty_string_draws_nothing() {
        let mut overlay = Overlay::new(4, 2);
        overlay.draw_text(0, 0, "", WHITE);
        assert!(overlay.is_empty());
    }

    #[test]
    fn injected_glyph_map_is_used() {
        let mut overlay = Overlay::with_glyph_map(4, 1, constant_glyph_of);
        overlay.draw_text(0, 0, "A ", WHITE);
        // Under the default policy these would be 7 ('A' by density) and 0 (' ').
        assert_ne!(default_glyph_of('A'), CUSTOM_GLYPH);
        assert_ne!(default_glyph_of(' '), CUSTOM_GLYPH);
        assert_eq!(overlay.cell(0, 0).unwrap().glyph_index, CUSTOM_GLYPH);
        assert_eq!(overlay.cell(1, 0).unwrap().glyph_index, CUSTOM_GLYPH);
        // The seam is used by every glyph-producing entry point, not just text.
        assert_eq!(overlay.glyph_index_of(' '), CUSTOM_GLYPH);
        assert_eq!(overlay.background_cell(WHITE).glyph_index, CUSTOM_GLYPH);
        overlay.draw_box(3, 0, 1, 1, WHITE);
        assert_eq!(overlay.cell(3, 0).unwrap().glyph_index, CUSTOM_GLYPH);

        // Swapping the policy at runtime affects later draws only.
        overlay.set_glyph_map(default_glyph_of);
        overlay.draw_text(2, 0, " ", WHITE);
        assert_eq!(overlay.cell(2, 0).unwrap().glyph_index, SPACE_INDEX);
        assert_eq!(overlay.cell(0, 0).unwrap().glyph_index, CUSTOM_GLYPH);
    }

    #[test]
    fn default_map_is_exact_for_atlas_chars() {
        for (index, &c) in ALL_CHARS.iter().enumerate() {
            assert_eq!(default_glyph_of(c), index as u32, "atlas char {c:?}");
        }
    }

    #[test]
    fn default_map_keeps_unknown_chars_inside_the_ramp() {
        let ramp_last = (BRIGHTNESS_RAMP.len() - 1) as u32;
        for c in ['A', 'z', '5', 'Q', 'i', '?', '(', 'ы', '☃'] {
            let index = default_glyph_of(c);
            assert!(index > 0, "visible char {c:?} mapped to blank");
            assert!(index <= ramp_last, "char {c:?} escaped the ramp: {index}");
        }
        // Blank-ish characters collapse to the space glyph.
        for c in ['\n', '\t', '\r', '\u{0}'] {
            assert_eq!(default_glyph_of(c), SPACE_INDEX, "char {c:?}");
        }
        // Heavier shapes get heavier glyphs: 'i' < 'a' < 'A'.
        assert!(default_glyph_of('i') < default_glyph_of('a'));
        assert!(default_glyph_of('a') < default_glyph_of('A'));
        // Derived by hand from the density table and ramp_index():
        // 0.3*9 = 2.7 -> 3, 0.55*9 = 4.95 -> 5, 0.75*9 = 6.75 -> 7.
        assert_eq!(default_glyph_of('i'), 3);
        assert_eq!(default_glyph_of('a'), 5);
        assert_eq!(default_glyph_of('A'), 7);
        assert_eq!(default_glyph_of('?'), 4); // 0.45*9 = 4.05 -> 4
        assert_eq!(default_glyph_of('☃'), 5); // 0.6*9 = 5.4 -> 5
    }

    #[test]
    fn brightness_ramp_is_a_prefix_of_all_chars() {
        // ramp_index() returns atlas indices only because of this invariant.
        assert!(BRIGHTNESS_RAMP.len() <= ALL_CHARS.len());
        for (i, &c) in BRIGHTNESS_RAMP.iter().enumerate() {
            assert_eq!(ALL_CHARS[i], c);
        }
        assert_eq!(ramp_index(0.0), 0);
        assert_eq!(ramp_index(1.0), (BRIGHTNESS_RAMP.len() - 1) as u32);
        assert_eq!(ramp_index(-5.0), 0);
        assert_eq!(ramp_index(9.0), (BRIGHTNESS_RAMP.len() - 1) as u32);
        // Non-finite densities must not panic and must stay inside the ramp.
        assert_eq!(ramp_index(f32::NAN), 0);
        assert_eq!(ramp_index(f32::NEG_INFINITY), 0);
        assert_eq!(ramp_index(f32::INFINITY), (BRIGHTNESS_RAMP.len() - 1) as u32);
    }

    #[test]
    fn background_cell_uses_the_densest_block() {
        let overlay = Overlay::new(2, 2);
        let cell = overlay.background_cell(RED);
        assert!(cell.opaque);
        assert_eq!(cell.color, RED);
        // '█' is the last entry of ALL_CHARS.
        assert_eq!(cell.glyph_index, (ALL_CHARS.len() - 1) as u32);
    }

    #[test]
    fn fill_rect_writes_exactly_the_region() {
        let mut overlay = Overlay::new(5, 4);
        overlay.fill_rect(1, 1, 3, 2, OverlayCell::new(11, WHITE));
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 6);
        for row in 1..3 {
            for col in 1..4 {
                assert!(overlay.cell(col, row).unwrap().opaque, "({col},{row})");
            }
        }
        assert!(!overlay.cell(0, 1).unwrap().opaque);
        assert!(!overlay.cell(4, 1).unwrap().opaque);
        assert!(!overlay.cell(1, 0).unwrap().opaque);
        assert!(!overlay.cell(1, 3).unwrap().opaque);
    }

    #[test]
    fn fill_rect_clips_at_every_edge() {
        let cell = OverlayCell::new(13, WHITE);

        // Right + bottom overflow.
        let mut overlay = Overlay::new(4, 4);
        overlay.fill_rect(2, 2, 10, 10, cell);
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 4);
        assert!(overlay.cell(3, 3).unwrap().opaque);

        // Starting exactly on the far edge: nothing visible.
        let mut overlay = Overlay::new(4, 4);
        overlay.fill_rect(4, 0, 2, 2, cell);
        overlay.fill_rect(0, 4, 2, 2, cell);
        assert!(overlay.is_empty());

        // Saturating arithmetic: no overflow panic in debug builds.
        let mut overlay = Overlay::new(4, 4);
        overlay.fill_rect(3, 3, u32::MAX, u32::MAX, cell);
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 1);
        assert!(overlay.cell(3, 3).unwrap().opaque);

        // Whole overlay.
        let mut overlay = Overlay::new(4, 4);
        overlay.fill_rect(0, 0, 4, 4, cell);
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 16);
    }

    #[test]
    fn zero_sized_rect_draws_nothing() {
        let mut overlay = Overlay::new(4, 4);
        overlay.fill_rect(1, 1, 0, 3, OverlayCell::new(13, WHITE));
        overlay.fill_rect(1, 1, 3, 0, OverlayCell::new(13, WHITE));
        overlay.draw_box(1, 1, 0, 3, WHITE);
        overlay.draw_box(1, 1, 3, 0, WHITE);
        assert!(overlay.is_empty());
    }

    #[test]
    fn draw_box_draws_border_only() {
        let mut overlay = Overlay::new(6, 5);
        overlay.draw_box(0, 0, 4, 3, RED);

        let horizontal = default_glyph_of('─');
        let vertical = default_glyph_of('│');

        // Corners: each is its own glyph now, so the frame closes cleanly instead
        // of showing the gaps a shared '+' left.
        for (col, row, corner_char) in [
            (0u32, 0u32, '┌'),
            (3, 0, '┐'),
            (0, 2, '└'),
            (3, 2, '┘'),
        ] {
            let cell = overlay.cell(col, row).unwrap();
            assert!(cell.opaque, "corner ({col},{row}) missing");
            assert_eq!(cell.glyph_index, default_glyph_of(corner_char), "corner ({col},{row})");
            assert_eq!(cell.color, RED);
        }
        // Horizontal edges.
        for col in 1..3 {
            assert_eq!(overlay.cell(col, 0).unwrap().glyph_index, horizontal);
            assert_eq!(overlay.cell(col, 2).unwrap().glyph_index, horizontal);
        }
        // Vertical edges.
        assert_eq!(overlay.cell(0, 1).unwrap().glyph_index, vertical);
        assert_eq!(overlay.cell(3, 1).unwrap().glyph_index, vertical);
        // Interior stays transparent.
        for col in 1..3 {
            assert_eq!(
                *overlay.cell(col, 1).unwrap(),
                OverlayCell::transparent(),
                "interior ({col},1) was painted"
            );
        }
        // Border of a 4x3 box = 2*4 + 2*(3-2) = 10 cells.
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 10);
        // Outside the box.
        assert!(!overlay.cell(4, 0).unwrap().opaque);
        assert!(!overlay.cell(0, 3).unwrap().opaque);
    }

    #[test]
    fn draw_box_degenerate_sizes() {
        let horizontal = default_glyph_of('─');
        let vertical = default_glyph_of('│');
        let top_left = default_glyph_of('┌');
        let top_right = default_glyph_of('┐');
        let bottom_left = default_glyph_of('└');

        // 1x1 -> a single cell that is both top and left, so it takes the
        // top-left corner.
        let mut overlay = Overlay::new(4, 4);
        overlay.draw_box(1, 1, 1, 1, WHITE);
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 1);
        assert_eq!(overlay.cell(1, 1).unwrap().glyph_index, top_left);

        // 1x3 -> vertical line capped with corners.
        let mut overlay = Overlay::new(4, 4);
        overlay.draw_box(0, 0, 1, 3, WHITE);
        assert_eq!(overlay.cell(0, 0).unwrap().glyph_index, top_left);
        assert_eq!(overlay.cell(0, 1).unwrap().glyph_index, vertical);
        assert_eq!(overlay.cell(0, 2).unwrap().glyph_index, bottom_left);
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 3);

        // 3x1 -> horizontal line capped with corners.
        let mut overlay = Overlay::new(4, 4);
        overlay.draw_box(0, 0, 3, 1, WHITE);
        assert_eq!(overlay.cell(0, 0).unwrap().glyph_index, top_left);
        assert_eq!(overlay.cell(1, 0).unwrap().glyph_index, horizontal);
        assert_eq!(overlay.cell(2, 0).unwrap().glyph_index, top_right);
        assert_eq!(overlay.cells().iter().filter(|c| c.opaque).count(), 3);
    }

    #[test]
    fn draw_box_clipping_keeps_edge_roles() {
        // The box extends past the right/bottom edge: the clipped-away far edge
        // must NOT turn visible cells into corners.
        let mut overlay = Overlay::new(3, 3);
        overlay.draw_box(0, 0, 10, 10, WHITE);
        let horizontal = default_glyph_of('─');
        let vertical = default_glyph_of('│');
        assert_eq!(overlay.cell(0, 0).unwrap().glyph_index, default_glyph_of('┌'));
        assert_eq!(overlay.cell(1, 0).unwrap().glyph_index, horizontal);
        assert_eq!(overlay.cell(2, 0).unwrap().glyph_index, horizontal);
        assert_eq!(overlay.cell(0, 1).unwrap().glyph_index, vertical);
        assert_eq!(overlay.cell(0, 2).unwrap().glyph_index, vertical);
        // Interior of the logical box, still interior after clipping.
        assert!(!overlay.cell(2, 2).unwrap().opaque);
        // Saturating: huge origin + huge size must not panic.
        overlay.draw_box(u32::MAX - 1, 0, u32::MAX, 2, WHITE);
    }

    #[test]
    fn composite_over_replaces_only_opaque_cells() {
        let mut overlay = Overlay::new(3, 2);
        overlay.set_cell(1, 0, OverlayCell::new(9, RED));
        overlay.set_cell(2, 1, OverlayCell::new(0, WHITE));

        let mut scene: Vec<SceneCell> = (0..6).map(|i| (i as u32, [0.5, 0.5, 0.5])).collect();
        overlay.composite_over(&mut scene);

        assert_eq!(scene[0], (0, [0.5, 0.5, 0.5]));
        assert_eq!(scene[1], (9, RED));
        assert_eq!(scene[2], (2, [0.5, 0.5, 0.5]));
        assert_eq!(scene[3], (3, [0.5, 0.5, 0.5]));
        assert_eq!(scene[4], (4, [0.5, 0.5, 0.5]));
        assert_eq!(scene[5], (0, WHITE));
    }

    #[test]
    fn composite_over_of_an_empty_overlay_changes_nothing() {
        let overlay = Overlay::new(2, 2);
        let original: Vec<SceneCell> = (0..4).map(|i| (i as u32, [1.0, 0.0, 0.0])).collect();
        let mut scene = original.clone();
        overlay.composite_over(&mut scene);
        assert_eq!(scene, original);
    }

    #[test]
    fn composite_over_tolerates_mismatched_scene_length() {
        let mut overlay = Overlay::new(3, 2);
        for row in 0..2 {
            for col in 0..3 {
                overlay.set_cell(col, row, OverlayCell::new(13, WHITE));
            }
        }

        // Too short: composited up to its length, no panic.
        let mut short: Vec<SceneCell> = vec![(1, [0.0, 0.0, 0.0]); 2];
        overlay.composite_over(&mut short);
        assert_eq!(short.len(), 2);
        assert!(short.iter().all(|c| *c == (13, WHITE)));

        // Too long: the tail beyond cols*rows is untouched.
        let mut long: Vec<SceneCell> = vec![(1, [0.0, 0.0, 0.0]); 9];
        overlay.composite_over(&mut long);
        assert_eq!(long.len(), 9);
        assert!(long[0..6].iter().all(|c| *c == (13, WHITE)));
        assert!(long[6..9].iter().all(|c| *c == (1, [0.0, 0.0, 0.0])));

        // Empty scene slice.
        overlay.composite_over(&mut []);
    }

    #[test]
    fn resize_to_bigger_and_smaller_stays_consistent() {
        let mut overlay = Overlay::new(4, 3);
        overlay.fill_rect(0, 0, 4, 3, OverlayCell::new(13, WHITE));
        assert!(!overlay.is_empty());

        overlay.resize(6, 5);
        assert_eq!(overlay.cols(), 6);
        assert_eq!(overlay.rows(), 5);
        assert_eq!(overlay.cells().len(), 30);
        assert!(overlay.is_empty(), "resize must drop contents");
        assert!(overlay.cell(5, 4).is_some());

        // The new far corner is addressable and lands at the right index.
        overlay.set_cell(5, 4, OverlayCell::new(1, RED));
        assert_eq!(overlay.cells()[29].glyph_index, 1);

        overlay.resize(2, 2);
        assert_eq!(overlay.cells().len(), 4);
        assert!(overlay.is_empty());
        assert!(overlay.cell(2, 0).is_none());
        overlay.set_cell(1, 1, OverlayCell::new(2, WHITE));
        assert_eq!(overlay.cells()[3].glyph_index, 2);
        assert!(!overlay.is_empty());

        // Resize to zero and back.
        overlay.resize(0, 0);
        assert_eq!(overlay.cells().len(), 0);
        assert!(overlay.is_empty());
        overlay.resize(3, 1);
        assert_eq!(overlay.cells().len(), 3);
    }

    #[test]
    fn zero_sized_overlay_is_usable() {
        for (cols, rows) in [(0u32, 0u32), (0, 5), (5, 0)] {
            let mut overlay = Overlay::new(cols, rows);
            assert_eq!(overlay.cells().len(), 0);
            assert!(overlay.is_empty());
            assert!(overlay.cell(0, 0).is_none());
            overlay.set_cell(0, 0, OverlayCell::new(3, WHITE));
            overlay.draw_text(0, 0, "hello", WHITE);
            overlay.fill_rect(0, 0, 4, 4, OverlayCell::new(13, WHITE));
            overlay.draw_box(0, 0, 4, 4, WHITE);
            overlay.clear();
            assert!(overlay.is_empty());
            let mut scene: Vec<SceneCell> = vec![(1, [0.0, 0.0, 0.0]); 3];
            overlay.composite_over(&mut scene);
            assert!(scene.iter().all(|c| *c == (1, [0.0, 0.0, 0.0])));
        }
    }
}
