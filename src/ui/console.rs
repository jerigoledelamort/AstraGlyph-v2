// Debug console / command line (ROADMAP 3.3) — editable input line, bounded
// scrollback and a bottom-anchored panel drawn into an `Overlay`.
//
// Design decisions:
// - No input library and no winit: the app converts whatever the platform gives
//   it into `insert_char` (a text character), abstract `ConsoleAction`s (edit /
//   cursor / history / clear) and the `scroll_up` / `scroll_down` calls. That
//   keeps this module pure logic plus drawing, so every rule below is
//   unit-testable without a window or a GPU.
// - No command execution here. `Submit` hands the line back to the caller, which
//   runs it and reports the result through `print` / `print_error`. A console
//   that owned a command table would have to know about the renderer, the scene
//   and gameplay — exactly the god object the architecture rules forbid. The
//   echo/history/clear-input bookkeeping still happens here, because that is
//   editor behaviour, not command behaviour.
// - The input line is stored as a `String` but every index the API exposes or
//   accepts is a CHARACTER index. Byte offsets are derived with `char_indices`
//   at the point of use, so a multi-byte character ('ы', an emoji) can never be
//   split by a slice; this module never indexes a `str` with a raw number.
// - Scrollback is a `Vec` bounded by `capacity`: pushing past it drops the
//   OLDEST lines. `Vec` (not `VecDeque`) because `lines()` hands out one
//   contiguous slice, and the front-drop cost is a memmove of a few hundred
//   small structs at most, once per printed line.
// - Long text is CLIPPED, never wrapped. Scrollback keeps a strict one line =
//   one row mapping, which makes the scroll window plain arithmetic; the prompt
//   row instead scrolls HORIZONTALLY so the cursor is always visible.
// - Every entry point is total: actions on an empty line, a cursor at either
//   end, an overlay far too small for the panel, or a zero-sized overlay are all
//   defined and panic-free. UI code runs every frame and must not be able to
//   crash the renderer.

use crate::ascii::font5x7::text_width;
use crate::ascii::overlay::{Overlay, OverlayCell};

/// Default number of scrollback lines kept by [`Console::new`].
pub const DEFAULT_SCROLLBACK: usize = 256;

/// Text drawn in front of the input line, and in front of echoed commands.
pub const PROMPT: &str = "> ";

/// Character marking the cursor position on the prompt row.
///
/// It replaces the character underneath it — the atlas has no inverse-video or
/// blinking support, so overwriting one cell is the only unambiguous marker.
pub const CURSOR_CHAR: char = '_';

/// Panel background tint (used with [`Overlay::background_cell`]).
pub const BACKGROUND_COLOR: [f32; 3] = [0.02, 0.03, 0.05];
/// Colour of echoed commands ([`LineKind::Input`]).
pub const INPUT_COLOR: [f32; 3] = [0.65, 0.85, 1.0];
/// Colour of normal output ([`LineKind::Output`]).
pub const OUTPUT_COLOR: [f32; 3] = [0.85, 0.90, 0.85];
/// Colour of error output ([`LineKind::Error`]).
pub const ERROR_COLOR: [f32; 3] = [1.0, 0.35, 0.30];
/// Colour of the prompt marker and of the cursor.
pub const PROMPT_COLOR: [f32; 3] = [0.95, 0.95, 0.55];

/// Smallest panel height (in rows) the console asks for: one scrollback row plus
/// the prompt row, plus one row of context.
const MIN_PANEL_ROWS: u32 = 3;

/// Fraction of the overlay height the panel occupies: `rows / PANEL_DIVISOR`.
const PANEL_DIVISOR: u32 = 3;

/// An abstract editing action, decoupled from any key or platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleAction {
    /// Accept the current line: echo it, push it to history, clear the input.
    Submit,
    /// Delete the character before the cursor.
    Backspace,
    /// Delete the character under the cursor.
    Delete,
    /// Move the cursor one character left.
    Left,
    /// Move the cursor one character right.
    Right,
    /// Move the cursor to the start of the line.
    Home,
    /// Move the cursor to the end of the line.
    End,
    /// Recall an older history entry.
    HistoryPrev,
    /// Recall a newer history entry, or the stashed in-progress line.
    HistoryNext,
    /// Wipe the scrollback (input and history are kept).
    Clear,
}

/// What a scrollback line is, which decides its colour and prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// A command echoed back when it was submitted. Drawn with the [`PROMPT`].
    Input,
    /// Normal command output.
    Output,
    /// An error reported by the command.
    Error,
}

impl LineKind {
    /// Colour this kind is drawn with.
    pub fn color(self) -> [f32; 3] {
        match self {
            LineKind::Input => INPUT_COLOR,
            LineKind::Output => OUTPUT_COLOR,
            LineKind::Error => ERROR_COLOR,
        }
    }
}

/// One line of scrollback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsoleLine {
    /// How the line should be presented.
    pub kind: LineKind,
    /// The text, without any prompt prefix.
    pub text: String,
}

impl ConsoleLine {
    /// Build a line of the given kind.
    pub fn new(kind: LineKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
        }
    }
}

/// Byte offset of character number `char_index`, or the string length when the
/// index is at/past the end.
///
/// The one place this module converts a character index into a byte offset; the
/// result is always a `char` boundary, so `String::insert` / `String::remove`
/// cannot panic on it.
fn byte_offset(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(offset, _)| offset)
        .unwrap_or(text.len())
}

/// A debug console: an edit line, command history and bounded scrollback.
///
/// The caller drives it: feed characters with [`Console::insert_char`] and
/// actions with [`Console::handle`], execute whatever `Submit` returns, then
/// report the result with [`Console::print`] / [`Console::print_error`].
#[derive(Clone, Debug)]
pub struct Console {
    open: bool,
    /// The line being edited. Never contains control characters.
    input: String,
    /// Cursor position in CHARACTERS, always `0..=input.chars().count()`.
    cursor: usize,
    /// Scrollback, oldest first, at most `capacity` entries.
    lines: Vec<ConsoleLine>,
    /// Scrollback bound; at least 1.
    capacity: usize,
    /// Submitted commands, oldest first.
    history: Vec<String>,
    /// Index into `history` while browsing it, `None` while editing own text.
    history_pos: Option<usize>,
    /// The in-progress line put aside when history browsing started.
    stash: Option<String>,
    /// How many lines the view is scrolled up from the bottom.
    scroll: usize,
}

impl Console {
    /// A closed, empty console with [`DEFAULT_SCROLLBACK`] lines of scrollback.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_SCROLLBACK)
    }

    /// A closed, empty console keeping at most `scrollback` lines.
    ///
    /// `0` is clamped to 1: a capacity of zero would silently discard every
    /// printed line, which reads as a broken console rather than as a setting.
    pub fn with_capacity(scrollback: usize) -> Self {
        Self {
            open: false,
            input: String::new(),
            cursor: 0,
            lines: Vec::new(),
            capacity: scrollback.max(1),
            history: Vec::new(),
            history_pos: None,
            stash: None,
            scroll: 0,
        }
    }

    /// Is the console visible? [`Console::draw`] is a no-op while it is not.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Show the console. Input, history and scrollback are preserved, so
    /// reopening resumes exactly where the user left off.
    pub fn open(&mut self) {
        self.open = true;
    }

    /// Hide the console without touching its state.
    pub fn close(&mut self) {
        self.open = false;
    }

    /// Flip [`Console::is_open`].
    pub fn toggle(&mut self) {
        self.open = !self.open;
    }

    /// Number of characters in the input line.
    fn input_len(&self) -> usize {
        self.input.chars().count()
    }

    /// The line currently being edited.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Cursor position in CHARACTERS (not bytes), `0..=input().chars().count()`.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Insert a character at the cursor and advance it.
    ///
    /// Control characters (including `'\n'`, `'\t'` and `'\r'`) are ignored:
    /// they are actions, not text, and would corrupt a single-line editor.
    /// Any other character is accepted verbatim, including ones the current
    /// glyph atlas cannot draw — the text is data, rendering is the overlay's
    /// glyph policy's problem.
    ///
    /// Editing detaches from history browsing (see [`Console::handle`]).
    pub fn insert_char(&mut self, c: char) {
        if c.is_control() {
            return;
        }
        let offset = byte_offset(&self.input, self.cursor);
        self.input.insert(offset, c);
        self.cursor += 1;
        self.detach_from_history();
    }

    /// Apply an action.
    ///
    /// Only [`ConsoleAction::Submit`] ever returns a value: the trimmed command
    /// line, or `None` when the line is empty or whitespace only (in which case
    /// nothing is echoed and nothing is pushed to history, but the input is
    /// still cleared). The caller executes the returned string.
    ///
    /// History browsing: `HistoryPrev` stashes the in-progress line before
    /// recalling the newest entry and then walks towards older ones, stopping at
    /// the oldest. `HistoryNext` walks back towards newer entries and, past the
    /// newest one, restores the stashed line — losing what you were typing is
    /// the classic console annoyance. Editing a recalled line detaches from
    /// history: the edited text becomes the working line and the stash is
    /// dropped, so a later `HistoryPrev` stashes *that* instead.
    pub fn handle(&mut self, action: ConsoleAction) -> Option<String> {
        match action {
            ConsoleAction::Submit => return self.submit(),
            ConsoleAction::Backspace => self.backspace(),
            ConsoleAction::Delete => self.delete(),
            ConsoleAction::Left => self.cursor = self.cursor.saturating_sub(1),
            ConsoleAction::Right => {
                self.cursor = self.cursor.saturating_add(1).min(self.input_len())
            }
            ConsoleAction::Home => self.cursor = 0,
            ConsoleAction::End => self.cursor = self.input_len(),
            ConsoleAction::HistoryPrev => self.history_prev(),
            ConsoleAction::HistoryNext => self.history_next(),
            ConsoleAction::Clear => self.clear_lines(),
        }
        None
    }

    /// Stop treating the input line as a recalled history entry.
    fn detach_from_history(&mut self) {
        self.history_pos = None;
        self.stash = None;
    }

    /// Delete the character before the cursor; a no-op at the start of the line.
    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let offset = byte_offset(&self.input, self.cursor - 1);
        self.input.remove(offset);
        self.cursor -= 1;
        self.detach_from_history();
    }

    /// Delete the character under the cursor; a no-op at the end of the line.
    fn delete(&mut self) {
        let offset = byte_offset(&self.input, self.cursor);
        if offset >= self.input.len() {
            return;
        }
        self.input.remove(offset);
        self.detach_from_history();
    }

    /// Replace the input line with `text`, cursor at the end.
    fn set_input(&mut self, text: String) {
        self.input = text;
        self.cursor = self.input_len();
    }

    /// Recall an older history entry.
    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let index = match self.history_pos {
            // Entering history: keep what the user was typing.
            None => {
                self.stash = Some(self.input.clone());
                self.history.len() - 1
            }
            // Already at the oldest entry: stay there.
            Some(0) => return,
            Some(pos) => pos - 1,
        };
        self.history_pos = Some(index);
        let entry = self.history[index].clone();
        self.set_input(entry);
    }

    /// Recall a newer history entry, or restore the stashed in-progress line.
    fn history_next(&mut self) {
        let Some(pos) = self.history_pos else {
            return;
        };
        if pos + 1 < self.history.len() {
            self.history_pos = Some(pos + 1);
            let entry = self.history[pos + 1].clone();
            self.set_input(entry);
        } else {
            // Past the newest entry: back to what the user was typing.
            self.history_pos = None;
            let stashed = self.stash.take().unwrap_or_default();
            self.set_input(stashed);
        }
    }

    /// Echo, remember and clear the current line. See [`Console::handle`].
    fn submit(&mut self) -> Option<String> {
        let command = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        self.detach_from_history();
        if command.is_empty() {
            return None;
        }
        self.push_line(ConsoleLine::new(LineKind::Input, command.clone()));
        // Consecutive duplicates add nothing but noise when walking history.
        if self.history.last() != Some(&command) {
            self.history.push(command.clone());
        }
        Some(command)
    }

    /// Append a line, dropping the oldest ones once `capacity` is exceeded, and
    /// jump the view back to the bottom.
    fn push_line(&mut self, line: ConsoleLine) {
        self.lines.push(line);
        if self.lines.len() > self.capacity {
            let excess = self.lines.len() - self.capacity;
            self.lines.drain(0..excess);
        }
        self.scroll = 0;
    }

    /// Append a normal output line. New output scrolls the view to the bottom.
    pub fn print(&mut self, text: impl Into<String>) {
        self.push_line(ConsoleLine::new(LineKind::Output, text));
    }

    /// Append an error line. New output scrolls the view to the bottom.
    pub fn print_error(&mut self, text: impl Into<String>) {
        self.push_line(ConsoleLine::new(LineKind::Error, text));
    }

    /// Scrollback, oldest first.
    pub fn lines(&self) -> &[ConsoleLine] {
        &self.lines
    }

    /// Submitted commands, oldest first.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Maximum scrollback capacity (never 0).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drop the scrollback. Input line and history are kept.
    fn clear_lines(&mut self) {
        self.lines.clear();
        self.scroll = 0;
    }

    /// How many lines the view is scrolled up from the bottom (0 = bottom).
    pub fn scroll_offset(&self) -> usize {
        self.scroll
    }

    /// Scroll towards older lines, clamped so at least one line stays in view.
    pub fn scroll_up(&mut self, amount: usize) {
        let max = self.lines.len().saturating_sub(1);
        self.scroll = self.scroll.saturating_add(amount).min(max);
    }

    /// Scroll towards newer lines, clamped at the bottom.
    pub fn scroll_down(&mut self, amount: usize) {
        self.scroll = self.scroll.saturating_sub(amount);
    }

    /// Draw the console panel, bottom-anchored over roughly the lower third of
    /// the overlay: an opaque background, the visible tail of the scrollback,
    /// and the prompt row at the very bottom.
    ///
    /// A no-op while closed. Layout is fully defensive: the panel is at most as
    /// tall as the overlay, so a 1-row overlay gets the prompt row only and a
    /// zero-sized overlay draws nothing. Scrollback lines are clipped at the
    /// right edge (never wrapped, so one line always occupies one row); the
    /// prompt row scrolls horizontally instead, keeping the cursor visible.
    ///
    /// The background comes from [`Overlay::background_cell`], i.e. the densest
    /// block glyph the overlay's glyph policy can supply. A policy without one
    /// (the combined atlas maps `'█'` to the blank glyph) yields an opaque blank
    /// cell instead — still a solid mask, so the scene never bleeds through the
    /// panel, it just renders as black rather than tinted.
    pub fn draw(&self, overlay: &mut Overlay) {
        if !self.open {
            return;
        }
        let (cols, rows) = (overlay.cols(), overlay.rows());
        if cols == 0 || rows == 0 {
            return;
        }

        let height = (rows / PANEL_DIVISOR).max(MIN_PANEL_ROWS).min(rows);
        let top = rows - height;
        let prompt_row = rows - 1;

        let background = overlay.background_cell(BACKGROUND_COLOR);
        overlay.fill_rect(0, top, cols, height, background);

        // Everything above the prompt row is scrollback; on a 1-row panel there
        // is none.
        self.draw_scrollback(overlay, top, height - 1);
        self.draw_prompt(overlay, prompt_row);
    }

    /// Draw the visible tail of the scrollback into `visible_rows` rows ending
    /// just above the prompt row.
    ///
    /// The text is bottom-aligned: the newest line always sits directly above
    /// the prompt, and a scrollback shorter than the panel leaves the gap at the
    /// top instead of floating above an empty strip.
    fn draw_scrollback(&self, overlay: &mut Overlay, top: u32, visible_rows: u32) {
        if visible_rows == 0 {
            return;
        }
        let visible = visible_rows as usize;
        let total = self.lines.len();
        // Clamp again here: `scroll_up` cannot know the panel height, so a
        // small panel may hold a scroll offset that would leave it blank.
        let scroll = self.scroll.min(total.saturating_sub(visible));
        let end = total - scroll;
        let start = end.saturating_sub(visible);
        // Empty rows go at the top of the panel.
        let first_row = top + (visible - (end - start)) as u32;
        let prompt_width = text_width(PROMPT) as u32;

        for (offset, line) in self.lines[start..end].iter().enumerate() {
            let Ok(offset) = u32::try_from(offset) else {
                return;
            };
            let row = first_row + offset;
            let color = line.kind.color();
            match line.kind {
                // Echoed commands carry the prompt, so they read as input even
                // in a monochrome colour mode.
                LineKind::Input => {
                    overlay.draw_text(0, row, PROMPT, color);
                    overlay.draw_text(prompt_width, row, &line.text, color);
                }
                _ => overlay.draw_text(0, row, &line.text, color),
            }
        }
    }

    /// Draw `PROMPT`, the visible window of the input line, and the cursor.
    fn draw_prompt(&self, overlay: &mut Overlay, row: u32) {
        overlay.draw_text(0, row, PROMPT, PROMPT_COLOR);

        let prompt_width = text_width(PROMPT) as u32;
        let available = overlay.cols().saturating_sub(prompt_width);
        if available == 0 {
            // Panel narrower than the prompt itself: the clipped prompt is all
            // that fits, and there is nowhere to put the cursor.
            return;
        }
        let available = available as usize;

        // Horizontal scroll: show the window that ends at the cursor, so the
        // cursor cell is always inside the panel.
        let start = self.cursor.saturating_sub(available - 1);
        let visible: String = self.input.chars().skip(start).take(available).collect();
        overlay.draw_text(prompt_width, row, &visible, INPUT_COLOR);

        let cursor_col = prompt_width + (self.cursor - start) as u32;
        let glyph = overlay.glyph_index_of(CURSOR_CHAR);
        overlay.set_cell(cursor_col, row, OverlayCell::new(glyph, PROMPT_COLOR));
    }
}

impl Default for Console {
    /// Same as [`Console::new`].
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Overlay whose glyph policy is the real text mapping, so a glyph index
    /// identifies exactly one character and cell assertions are meaningful
    /// (the default shading policy collapses most letters onto one ramp glyph).
    fn overlay(cols: u32, rows: u32) -> Overlay {
        Overlay::with_glyph_map(cols, rows, crate::ascii::overlay_glyph_of)
    }

    /// Type `text` into the console, character by character.
    fn type_text(console: &mut Console, text: &str) {
        for c in text.chars() {
            console.insert_char(c);
        }
    }

    /// Read back a row as a string, using the glyph mapping in reverse.
    fn row_text(overlay: &Overlay, row: u32) -> String {
        (0..overlay.cols())
            .map(|col| {
                let glyph = overlay.cell(col, row).map(|c| c.glyph_index).unwrap_or(0);
                (' '..='~')
                    .find(|&c| crate::ascii::overlay_glyph_of(c) == glyph)
                    .unwrap_or(' ')
            })
            .collect()
    }

    #[test]
    fn new_console_is_closed_and_empty() {
        let console = Console::default();
        assert!(!console.is_open());
        assert_eq!(console.input(), "");
        assert_eq!(console.cursor(), 0);
        assert!(console.lines().is_empty());
        assert!(console.history().is_empty());
        assert_eq!(console.scroll_offset(), 0);
        assert_eq!(console.capacity(), DEFAULT_SCROLLBACK);
    }

    #[test]
    fn open_close_toggle_preserve_state() {
        let mut console = Console::new();
        type_text(&mut console, "spawn");
        console.open();
        assert!(console.is_open());
        console.toggle();
        assert!(!console.is_open());
        console.toggle();
        assert!(console.is_open());
        console.close();
        assert!(!console.is_open());
        // Closing must not throw away the draft.
        assert_eq!(console.input(), "spawn");
        assert_eq!(console.cursor(), 5);
    }

    #[test]
    fn insert_and_submit_round_trip() {
        let mut console = Console::new();
        type_text(&mut console, "help me");
        assert_eq!(console.input(), "help me");
        assert_eq!(console.cursor(), 7);

        assert_eq!(
            console.handle(ConsoleAction::Submit),
            Some("help me".to_string())
        );
        // Input cleared, command echoed and remembered.
        assert_eq!(console.input(), "");
        assert_eq!(console.cursor(), 0);
        assert_eq!(
            console.lines().to_vec(),
            vec![ConsoleLine::new(LineKind::Input, "help me")]
        );
        assert_eq!(console.history().to_vec(), vec!["help me".to_string()]);
    }

    #[test]
    fn submit_trims_and_ignores_blank_lines() {
        let mut console = Console::new();
        assert_eq!(console.handle(ConsoleAction::Submit), None);
        type_text(&mut console, "   ");
        assert_eq!(console.handle(ConsoleAction::Submit), None);
        // Blank submits clear the line but leave no trace.
        assert_eq!(console.input(), "");
        assert_eq!(console.cursor(), 0);
        assert!(console.lines().is_empty());
        assert!(console.history().is_empty());

        type_text(&mut console, "  fps  ");
        assert_eq!(console.handle(ConsoleAction::Submit), Some("fps".to_string()));
        assert_eq!(console.lines()[0].text, "fps");
        assert_eq!(console.history().to_vec(), vec!["fps".to_string()]);
    }

    #[test]
    fn insert_char_ignores_control_characters() {
        let mut console = Console::new();
        for c in ['\n', '\r', '\t', '\u{0}', '\u{7f}', '\u{1b}'] {
            console.insert_char(c);
        }
        assert_eq!(console.input(), "");
        assert_eq!(console.cursor(), 0);

        type_text(&mut console, "a\nb");
        assert_eq!(console.input(), "ab");
        assert_eq!(console.cursor(), 2);
    }

    #[test]
    fn insert_happens_at_the_cursor() {
        let mut console = Console::new();
        type_text(&mut console, "ac");
        console.handle(ConsoleAction::Left);
        console.insert_char('b');
        assert_eq!(console.input(), "abc");
        assert_eq!(console.cursor(), 2);
        console.handle(ConsoleAction::Home);
        console.insert_char('.');
        assert_eq!(console.input(), ".abc");
        assert_eq!(console.cursor(), 1);
    }

    #[test]
    fn backspace_at_both_ends_and_in_the_middle() {
        let mut console = Console::new();
        // Empty line: safe no-op.
        console.handle(ConsoleAction::Backspace);
        assert_eq!(console.input(), "");
        assert_eq!(console.cursor(), 0);

        type_text(&mut console, "abcd");
        // At the end.
        console.handle(ConsoleAction::Backspace);
        assert_eq!(console.input(), "abc");
        assert_eq!(console.cursor(), 3);
        // In the middle.
        console.handle(ConsoleAction::Left);
        console.handle(ConsoleAction::Backspace);
        assert_eq!(console.input(), "ac");
        assert_eq!(console.cursor(), 1);
        // At the start: nothing to delete.
        console.handle(ConsoleAction::Home);
        console.handle(ConsoleAction::Backspace);
        assert_eq!(console.input(), "ac");
        assert_eq!(console.cursor(), 0);
    }

    #[test]
    fn delete_at_both_ends_and_in_the_middle() {
        let mut console = Console::new();
        // Empty line: safe no-op.
        console.handle(ConsoleAction::Delete);
        assert_eq!(console.input(), "");

        type_text(&mut console, "abcd");
        // At the end: nothing under the cursor.
        console.handle(ConsoleAction::Delete);
        assert_eq!(console.input(), "abcd");
        assert_eq!(console.cursor(), 4);
        // At the start.
        console.handle(ConsoleAction::Home);
        console.handle(ConsoleAction::Delete);
        assert_eq!(console.input(), "bcd");
        assert_eq!(console.cursor(), 0);
        // In the middle.
        console.handle(ConsoleAction::Right);
        console.handle(ConsoleAction::Delete);
        assert_eq!(console.input(), "bd");
        assert_eq!(console.cursor(), 1);
    }

    #[test]
    fn cursor_movement_is_bounded() {
        let mut console = Console::new();
        // Empty line: every move stays at 0.
        for action in [
            ConsoleAction::Left,
            ConsoleAction::Right,
            ConsoleAction::Home,
            ConsoleAction::End,
        ] {
            console.handle(action);
            assert_eq!(console.cursor(), 0);
        }

        type_text(&mut console, "abc");
        for _ in 0..5 {
            console.handle(ConsoleAction::Right);
        }
        assert_eq!(console.cursor(), 3, "cursor ran past the end");
        for _ in 0..5 {
            console.handle(ConsoleAction::Left);
        }
        assert_eq!(console.cursor(), 0, "cursor ran before the start");
        console.handle(ConsoleAction::End);
        assert_eq!(console.cursor(), 3);
        console.handle(ConsoleAction::Home);
        assert_eq!(console.cursor(), 0);
    }

    #[test]
    fn multibyte_input_keeps_characters_intact() {
        let mut console = Console::new();
        type_text(&mut console, "ыф");
        console.insert_char('🚀');
        assert_eq!(console.input(), "ыф🚀");
        // Three characters, eight bytes (2 + 2 + 4): the cursor counts
        // characters, so it must not follow the byte length.
        assert_eq!(console.cursor(), 3);
        assert_eq!(console.input().len(), 8);

        // Insert in the middle of multi-byte text.
        console.handle(ConsoleAction::Home);
        console.handle(ConsoleAction::Right);
        console.insert_char('ю');
        assert_eq!(console.input(), "ыюф🚀");
        assert_eq!(console.cursor(), 2);

        // Every move lands on a character boundary.
        console.handle(ConsoleAction::End);
        assert_eq!(console.cursor(), 4);
        console.handle(ConsoleAction::Left);
        assert_eq!(console.cursor(), 3);
    }

    #[test]
    fn multibyte_backspace_and_delete_remove_whole_characters() {
        let mut console = Console::new();
        type_text(&mut console, "ыф🚀д");

        console.handle(ConsoleAction::Backspace);
        assert_eq!(console.input(), "ыф🚀");
        assert_eq!(console.cursor(), 3);

        // Backspace over the 4-byte emoji.
        console.handle(ConsoleAction::Backspace);
        assert_eq!(console.input(), "ыф");
        assert_eq!(console.cursor(), 2);

        // Delete from the front, and in the middle.
        type_text(&mut console, "🚀ю");
        assert_eq!(console.input(), "ыф🚀ю");
        console.handle(ConsoleAction::Home);
        console.handle(ConsoleAction::Delete);
        assert_eq!(console.input(), "ф🚀ю");
        assert_eq!(console.cursor(), 0);
        console.handle(ConsoleAction::Right);
        console.handle(ConsoleAction::Delete);
        assert_eq!(console.input(), "фю");
        assert_eq!(console.cursor(), 1);

        // Delete at the end of multi-byte text is a no-op, not a panic.
        console.handle(ConsoleAction::End);
        console.handle(ConsoleAction::Delete);
        assert_eq!(console.input(), "фю");
        assert_eq!(console.cursor(), 2);

        // Submitting multi-byte text round-trips.
        assert_eq!(console.handle(ConsoleAction::Submit), Some("фю".to_string()));
        assert_eq!(console.lines()[0].text, "фю");
    }

    #[test]
    fn history_walks_in_both_directions() {
        let mut console = Console::new();
        for command in ["one", "two", "three"] {
            type_text(&mut console, command);
            console.handle(ConsoleAction::Submit);
        }
        assert_eq!(console.history().len(), 3);

        // Up: newest first, then older, stopping at the oldest.
        console.handle(ConsoleAction::HistoryPrev);
        assert_eq!(console.input(), "three");
        assert_eq!(console.cursor(), 5, "recall puts the cursor at the end");
        console.handle(ConsoleAction::HistoryPrev);
        assert_eq!(console.input(), "two");
        console.handle(ConsoleAction::HistoryPrev);
        assert_eq!(console.input(), "one");
        console.handle(ConsoleAction::HistoryPrev);
        assert_eq!(console.input(), "one", "walked past the oldest entry");

        // Down again.
        console.handle(ConsoleAction::HistoryNext);
        assert_eq!(console.input(), "two");
        console.handle(ConsoleAction::HistoryNext);
        assert_eq!(console.input(), "three");
        // Past the newest: back to the (empty) line being typed.
        console.handle(ConsoleAction::HistoryNext);
        assert_eq!(console.input(), "");
        assert_eq!(console.cursor(), 0);
        // Already detached: another Next changes nothing.
        console.handle(ConsoleAction::HistoryNext);
        assert_eq!(console.input(), "");

        // Recalled commands can be submitted again.
        console.handle(ConsoleAction::HistoryPrev);
        assert_eq!(console.handle(ConsoleAction::Submit), Some("three".to_string()));
    }

    #[test]
    fn history_next_restores_the_stashed_line() {
        let mut console = Console::new();
        type_text(&mut console, "spawn");
        console.handle(ConsoleAction::Submit);
        type_text(&mut console, "half typ");

        console.handle(ConsoleAction::HistoryPrev);
        assert_eq!(console.input(), "spawn");
        console.handle(ConsoleAction::HistoryNext);
        assert_eq!(console.input(), "half typ", "in-progress line was lost");
        assert_eq!(console.cursor(), 8);
    }

    #[test]
    fn history_on_an_empty_history_is_a_noop() {
        let mut console = Console::new();
        type_text(&mut console, "draft");
        console.handle(ConsoleAction::HistoryPrev);
        console.handle(ConsoleAction::HistoryNext);
        assert_eq!(console.input(), "draft");
        assert_eq!(console.cursor(), 5);
    }

    #[test]
    fn editing_a_recalled_line_detaches_from_history() {
        let mut console = Console::new();
        for command in ["alpha", "beta"] {
            type_text(&mut console, command);
            console.handle(ConsoleAction::Submit);
        }
        type_text(&mut console, "draft");

        console.handle(ConsoleAction::HistoryPrev); // "beta"
        console.insert_char('!'); // now editing my own line
        assert_eq!(console.input(), "beta!");
        // Next no longer restores anything: we are no longer in history.
        console.handle(ConsoleAction::HistoryNext);
        assert_eq!(console.input(), "beta!");
        // And a fresh Prev stashes the edited line instead.
        console.handle(ConsoleAction::HistoryPrev);
        assert_eq!(console.input(), "beta");
        console.handle(ConsoleAction::HistoryNext);
        assert_eq!(console.input(), "beta!");
    }

    #[test]
    fn history_skips_consecutive_duplicates() {
        let mut console = Console::new();
        for _ in 0..3 {
            type_text(&mut console, "fps");
            console.handle(ConsoleAction::Submit);
        }
        type_text(&mut console, "grid");
        console.handle(ConsoleAction::Submit);
        type_text(&mut console, "fps");
        console.handle(ConsoleAction::Submit);
        assert_eq!(
            console.history().to_vec(),
            vec!["fps".to_string(), "grid".to_string(), "fps".to_string()]
        );
        // Every submit is still echoed, duplicate or not.
        assert_eq!(console.lines().len(), 5);
    }

    #[test]
    fn print_and_print_error_use_distinct_kinds() {
        let mut console = Console::new();
        console.print("ok");
        console.print_error("boom");
        console.print(String::from("owned"));
        assert_eq!(
            console.lines().to_vec(),
            vec![
                ConsoleLine::new(LineKind::Output, "ok"),
                ConsoleLine::new(LineKind::Error, "boom"),
                ConsoleLine::new(LineKind::Output, "owned"),
            ]
        );
        assert_eq!(console.lines()[0].kind.color(), OUTPUT_COLOR);
        assert_eq!(console.lines()[1].kind.color(), ERROR_COLOR);
        assert_ne!(LineKind::Input.color(), OUTPUT_COLOR);
        assert_ne!(LineKind::Input.color(), ERROR_COLOR);
    }

    #[test]
    fn scrollback_drops_the_oldest_lines() {
        let mut console = Console::with_capacity(3);
        for i in 0..5 {
            console.print(format!("line {i}"));
        }
        assert_eq!(console.lines().len(), 3);
        let texts: Vec<&str> = console.lines().iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, ["line 2", "line 3", "line 4"]);

        // Echoed commands are bounded by the same capacity.
        type_text(&mut console, "cmd");
        console.handle(ConsoleAction::Submit);
        assert_eq!(console.lines().len(), 3);
        assert_eq!(console.lines()[2].text, "cmd");
        assert_eq!(console.lines()[0].text, "line 3");
    }

    #[test]
    fn zero_capacity_still_keeps_one_line() {
        let mut console = Console::with_capacity(0);
        assert_eq!(console.capacity(), 1);
        console.print("a");
        console.print("b");
        assert_eq!(console.lines().len(), 1);
        assert_eq!(console.lines()[0].text, "b");
    }

    #[test]
    fn clear_wipes_the_scrollback_only() {
        let mut console = Console::new();
        type_text(&mut console, "cmd");
        console.handle(ConsoleAction::Submit);
        console.print("out");
        console.scroll_up(1);
        type_text(&mut console, "draft");

        console.handle(ConsoleAction::Clear);
        assert!(console.lines().is_empty());
        assert_eq!(console.scroll_offset(), 0);
        // History and the line being typed survive.
        assert_eq!(console.history().to_vec(), vec!["cmd".to_string()]);
        assert_eq!(console.input(), "draft");
    }

    #[test]
    fn scroll_offset_clamps_at_both_ends() {
        let mut console = Console::new();
        // Nothing to scroll yet.
        console.scroll_up(10);
        assert_eq!(console.scroll_offset(), 0);

        for i in 0..5 {
            console.print(format!("line {i}"));
        }
        console.scroll_up(2);
        assert_eq!(console.scroll_offset(), 2);
        console.scroll_up(usize::MAX);
        assert_eq!(console.scroll_offset(), 4, "at least one line stays in view");
        console.scroll_down(1);
        assert_eq!(console.scroll_offset(), 3);
        console.scroll_down(usize::MAX);
        assert_eq!(console.scroll_offset(), 0);

        // New output jumps back to the bottom.
        console.scroll_up(3);
        console.print("fresh");
        assert_eq!(console.scroll_offset(), 0);
    }

    #[test]
    fn draw_is_a_noop_while_closed() {
        let mut console = Console::new();
        console.print("hidden");
        type_text(&mut console, "cmd");
        let mut target = overlay(20, 9);
        console.draw(&mut target);
        assert!(target.is_empty(), "closed console painted the overlay");

        console.open();
        console.draw(&mut target);
        assert!(!target.is_empty());
    }

    #[test]
    fn draw_does_not_panic_on_a_tiny_overlay() {
        let mut console = Console::new();
        console.open();
        console.print("some fairly long output line");
        type_text(&mut console, "a rather long command line too");

        for (cols, rows) in [(0u32, 0u32), (1, 1), (1, 0), (0, 4), (2, 1), (3, 2), (4, 4)] {
            let mut target = overlay(cols, rows);
            console.draw(&mut target);
            // Nothing may be painted outside the buffer, and it must not grow.
            assert_eq!(target.cells().len(), (cols as usize) * (rows as usize));
        }

        // A 1x1 overlay gets the first prompt character and nothing else.
        let mut target = overlay(1, 1);
        console.draw(&mut target);
        assert_eq!(row_text(&target, 0), ">");
    }

    #[test]
    fn draw_panel_is_bottom_anchored_over_the_lower_third() {
        let mut console = Console::new();
        console.open();
        let mut target = overlay(10, 12);
        console.draw(&mut target);

        // 12 / 3 = 4 rows -> rows 8..12 painted, everything above untouched.
        for row in 0..8 {
            for col in 0..10 {
                assert!(
                    !target.cell(col, row).unwrap().opaque,
                    "painted above the panel at ({col},{row})"
                );
            }
        }
        for row in 8..12 {
            for col in 0..10 {
                assert!(
                    target.cell(col, row).unwrap().opaque,
                    "panel gap at ({col},{row})"
                );
            }
        }
    }

    #[test]
    fn draw_shows_the_newest_lines_above_the_prompt() {
        let mut console = Console::new();
        console.open();
        console.print("aaa");
        console.print_error("bbb");
        console.print("ccc");
        type_text(&mut console, "cmd");

        let mut target = overlay(8, 3);
        console.draw(&mut target);
        // Panel = 3 rows: two scrollback rows (the two newest lines) + prompt.
        assert_eq!(row_text(&target, 0), "bbb     ");
        assert_eq!(row_text(&target, 1), "ccc     ");
        assert_eq!(row_text(&target, 2), "> cmd_  ");
        // Kinds are colour-coded.
        assert_eq!(target.cell(0, 0).unwrap().color, ERROR_COLOR);
        assert_eq!(target.cell(0, 1).unwrap().color, OUTPUT_COLOR);
        assert_eq!(target.cell(5, 2).unwrap().color, PROMPT_COLOR);

        // Scrolling up moves the window towards older lines.
        console.scroll_up(1);
        let mut target = overlay(8, 3);
        console.draw(&mut target);
        assert_eq!(row_text(&target, 0), "aaa     ");
        assert_eq!(row_text(&target, 1), "bbb     ");
    }

    #[test]
    fn draw_prefixes_echoed_commands_with_the_prompt() {
        let mut console = Console::new();
        console.open();
        type_text(&mut console, "fps");
        console.handle(ConsoleAction::Submit);
        console.print("60");

        let mut target = overlay(10, 3);
        console.draw(&mut target);
        assert_eq!(row_text(&target, 0), "> fps     ");
        assert_eq!(row_text(&target, 1), "60        ");
        assert_eq!(target.cell(0, 0).unwrap().color, INPUT_COLOR);
    }

    #[test]
    fn draw_clips_long_scrollback_lines_without_wrapping() {
        let mut console = Console::new();
        console.open();
        console.print("0123456789ABCDEF");

        let mut target = overlay(6, 3);
        console.draw(&mut target);
        // Row 0 is empty (only one line of scrollback, and the gap goes to the
        // top), row 1 holds the clipped line, and nothing spilled onto the
        // prompt row — the 10 characters that did not fit are simply gone.
        assert_eq!(row_text(&target, 0), "      ");
        assert_eq!(row_text(&target, 1), "012345");
        assert_eq!(row_text(&target, 2), "> _   ");
        // Clipped, not wrapped: the panel is still exactly 3 rows of 6 cells.
        assert_eq!(target.cells().len(), 18);
    }

    #[test]
    fn draw_scrolls_the_prompt_to_keep_the_cursor_visible() {
        let mut console = Console::new();
        console.open();
        type_text(&mut console, "abcdefgh");

        // 8 columns: 2 for the prompt, 6 for the line, cursor at the end.
        let mut target = overlay(8, 3);
        console.draw(&mut target);
        assert_eq!(row_text(&target, 2), "> defgh_");

        // Home: the window snaps back to the start of the line.
        console.handle(ConsoleAction::Home);
        let mut target = overlay(8, 3);
        console.draw(&mut target);
        assert_eq!(row_text(&target, 2), "> _bcdef");

        // Multi-byte text must not break the window arithmetic. 'ы' is not in
        // the font, so it renders blank — the cursor position is what matters.
        let mut console = Console::new();
        console.open();
        type_text(&mut console, "ыыы");
        console.handle(ConsoleAction::Left);
        let mut target = overlay(8, 3);
        console.draw(&mut target);
        assert_eq!(row_text(&target, 2), ">   _   ");
    }
}
