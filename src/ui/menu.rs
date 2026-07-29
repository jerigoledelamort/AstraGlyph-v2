// Text menu system (ROADMAP Phase 3.3): a navigable panel of buttons, toggles,
// choices and submenus, drawn into the ASCII overlay.
//
// The menu never sees a key code. It consumes abstract `MenuAction`s and returns
// `MenuEvent`s, so the app layer owns the bindings and this module stays testable
// without a window — the same split the camera rig uses.
//
// Two details drive most of the design:
//
// * Separators are not selectable, so navigation has to step OVER them. Every
//   off-by-one bug in a menu lives here, which is why the search is written once
//   as a bounded scan and reused by both directions.
// * Submenus need a path stack, and leaving one must restore the parent's
//   previous selection. Resetting to the first item on every `Back` is the most
//   common papercut in text menus.

use crate::ascii::font5x7;
use crate::ascii::overlay::{Overlay, OverlayCell};

/// Abstract navigation vocabulary. The app maps real keys onto these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuAction {
    Up,
    Down,
    Activate,
    Back,
}

/// One row of a menu.
#[derive(Clone, Debug, PartialEq)]
pub enum MenuItem {
    /// Fires [`MenuEvent::Activated`] with its id.
    Button { label: String, id: String },
    /// Flips on activation and reports the NEW state.
    Toggle { label: String, id: String, on: bool },
    /// Cycles through `options`, reporting the new index.
    Choice {
        label: String,
        id: String,
        options: Vec<String>,
        selected: usize,
    },
    /// Descends into a nested menu.
    Submenu { label: String, items: Vec<MenuItem> },
    /// Visual divider — never selectable.
    Separator,
}

impl MenuItem {
    /// Convenience constructor for a button.
    pub fn button(label: impl Into<String>, id: impl Into<String>) -> Self {
        Self::Button { label: label.into(), id: id.into() }
    }

    /// Convenience constructor for a toggle.
    pub fn toggle(label: impl Into<String>, id: impl Into<String>, on: bool) -> Self {
        Self::Toggle { label: label.into(), id: id.into(), on }
    }

    /// Convenience constructor for a choice. `selected` is clamped into range so
    /// a caller cannot build an item that indexes out of bounds.
    pub fn choice(
        label: impl Into<String>,
        id: impl Into<String>,
        options: Vec<String>,
        selected: usize,
    ) -> Self {
        let bounded = if options.is_empty() {
            0
        } else {
            selected.min(options.len() - 1)
        };
        Self::Choice {
            label: label.into(),
            id: id.into(),
            options,
            selected: bounded,
        }
    }

    /// Convenience constructor for a submenu.
    pub fn submenu(label: impl Into<String>, items: Vec<MenuItem>) -> Self {
        Self::Submenu { label: label.into(), items }
    }

    /// Whether navigation can land on this item.
    pub fn is_selectable(&self) -> bool {
        !matches!(self, MenuItem::Separator)
    }

    /// The item's id, if it has one (separators and submenus do not).
    pub fn id(&self) -> Option<&str> {
        match self {
            MenuItem::Button { id, .. }
            | MenuItem::Toggle { id, .. }
            | MenuItem::Choice { id, .. } => Some(id),
            _ => None,
        }
    }

    /// Text shown for this row, including any state indicator.
    fn display(&self) -> String {
        match self {
            MenuItem::Button { label, .. } => label.clone(),
            MenuItem::Toggle { label, on, .. } => {
                format!("{label}  [{}]", if *on { 'x' } else { ' ' })
            }
            MenuItem::Choice { label, options, selected, .. } => match options.get(*selected) {
                Some(option) => format!("{label}  < {option} >"),
                None => format!("{label}  < - >"),
            },
            // The arrow signals "there is more behind this row".
            MenuItem::Submenu { label, .. } => format!("{label}  \u{25BA}"),
            MenuItem::Separator => String::new(),
        }
    }
}

/// What a [`MenuAction`] produced.
#[derive(Clone, Debug, PartialEq)]
pub enum MenuEvent {
    Activated(String),
    Toggled(String, bool),
    Chose(String, usize),
    Closed,
}

/// One entry on the submenu path: which index was selected at that level.
#[derive(Clone, Copy, Debug)]
struct Frame {
    /// Index of the submenu item that was entered.
    entered: usize,
    /// Selection to restore when coming back up.
    restore: usize,
}

/// Maximum submenu nesting the menu will descend. A corrupted or cyclic item
/// tree therefore cannot make navigation recurse without bound.
const MAX_DEPTH: usize = 16;

/// A navigable text menu.
#[derive(Clone, Debug)]
pub struct Menu {
    title: String,
    items: Vec<MenuItem>,
    /// Path of entered submenus, outermost first.
    path: Vec<Frame>,
    /// Selection within the current level.
    selected: usize,
    open: bool,
    /// First visible row when the current level does not fit the panel.
    scroll: usize,
}

impl Menu {
    /// Build a menu. The initial selection lands on the first selectable item.
    pub fn new(title: impl Into<String>, items: Vec<MenuItem>) -> Self {
        let mut menu = Self {
            title: title.into(),
            items,
            path: Vec::new(),
            selected: 0,
            open: false,
            scroll: 0,
        };
        menu.selected = menu.first_selectable().unwrap_or(0);
        menu
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn open(&mut self) {
        self.open = true;
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    /// Nesting level; 0 at the top.
    pub fn depth(&self) -> usize {
        self.path.len()
    }

    /// Items of the level currently displayed.
    pub fn current_items(&self) -> &[MenuItem] {
        let mut items: &[MenuItem] = &self.items;
        for frame in &self.path {
            match items.get(frame.entered) {
                Some(MenuItem::Submenu { items: nested, .. }) => items = nested,
                // The path can only be built by descending into real submenus, so
                // this is unreachable in practice; returning the level found so
                // far is still better than panicking.
                _ => break,
            }
        }
        items
    }

    /// Selected index within the current level, or `None` when the level has no
    /// selectable items at all.
    pub fn selected_index(&self) -> Option<usize> {
        let items = self.current_items();
        match items.get(self.selected) {
            Some(item) if item.is_selectable() => Some(self.selected),
            _ => None,
        }
    }

    /// First visible row of the current level (non-zero only when scrolled).
    pub fn scroll_offset(&self) -> usize {
        self.scroll
    }

    /// Find an item by id anywhere in the tree, so the app can read a toggle or
    /// choice value back without tracking events.
    pub fn item_state(&self, id: &str) -> Option<&MenuItem> {
        fn search<'a>(items: &'a [MenuItem], id: &str, depth: usize) -> Option<&'a MenuItem> {
            if depth >= MAX_DEPTH {
                return None;
            }
            for item in items {
                if item.id() == Some(id) {
                    return Some(item);
                }
                if let MenuItem::Submenu { items: nested, .. } = item {
                    if let Some(found) = search(nested, id, depth + 1) {
                        return Some(found);
                    }
                }
            }
            None
        }
        search(&self.items, id, 0)
    }

    /// Set a toggle's state by id, anywhere in the tree.
    ///
    /// Needed so an opened menu reflects what is actually active: the engine owns
    /// the settings, the menu is only a view of them, and a menu that shows stale
    /// values is worse than no menu.
    pub fn set_toggle(&mut self, id: &str, value: bool) -> bool {
        Self::visit_mut(&mut self.items, id, 0, &mut |item| {
            if let MenuItem::Toggle { on, .. } = item {
                *on = value;
                true
            } else {
                false
            }
        })
    }

    /// Set a choice's selected index by id, clamped to its option count.
    pub fn set_choice(&mut self, id: &str, index: usize) -> bool {
        Self::visit_mut(&mut self.items, id, 0, &mut |item| {
            if let MenuItem::Choice { options, selected, .. } = item {
                if options.is_empty() {
                    return false;
                }
                *selected = index.min(options.len() - 1);
                true
            } else {
                false
            }
        })
    }

    /// Find the item with `id` and apply `f`; returns what `f` reported.
    fn visit_mut(
        items: &mut [MenuItem],
        id: &str,
        depth: usize,
        f: &mut impl FnMut(&mut MenuItem) -> bool,
    ) -> bool {
        if depth >= MAX_DEPTH {
            return false;
        }
        for item in items.iter_mut() {
            if item.id() == Some(id) {
                return f(item);
            }
            if let MenuItem::Submenu { items: nested, .. } = item {
                if Self::visit_mut(nested, id, depth + 1, f) {
                    return true;
                }
            }
        }
        false
    }

    /// Apply a navigation action.
    pub fn handle(&mut self, action: MenuAction) -> Option<MenuEvent> {
        if !self.open {
            return None;
        }
        match action {
            MenuAction::Up => {
                self.step(-1);
                None
            }
            MenuAction::Down => {
                self.step(1);
                None
            }
            MenuAction::Activate => self.activate(),
            MenuAction::Back => self.back(),
        }
    }

    /// Move the selection by `delta` over selectable items, wrapping around.
    fn step(&mut self, delta: isize) {
        let len = self.current_items().len();
        if len == 0 {
            return;
        }
        // Bounded scan: at most one full lap, so a level of nothing but
        // separators terminates instead of spinning.
        let mut index = self.selected;
        for _ in 0..len {
            index = if delta < 0 {
                if index == 0 { len - 1 } else { index - 1 }
            } else {
                (index + 1) % len
            };
            if self
                .current_items()
                .get(index)
                .is_some_and(|item| item.is_selectable())
            {
                self.selected = index;
                return;
            }
        }
        // No selectable item anywhere in this level: leave the selection alone.
    }

    fn first_selectable(&self) -> Option<usize> {
        self.current_items()
            .iter()
            .position(|item| item.is_selectable())
    }

    fn activate(&mut self) -> Option<MenuEvent> {
        let index = self.selected_index()?;

        // Descending needs the immutable borrow released first.
        let is_submenu = matches!(self.current_items().get(index), Some(MenuItem::Submenu { .. }));
        if is_submenu {
            if self.path.len() >= MAX_DEPTH {
                return None;
            }
            self.path.push(Frame { entered: index, restore: self.selected });
            self.selected = self.first_selectable().unwrap_or(0);
            self.scroll = 0;
            return None;
        }

        // Mutating an item at the current level means reaching through the path.
        // `self.items` and `self.path` are disjoint fields, so borrowing one
        // mutably and the other immutably is fine.
        Self::apply_at(&mut self.items, &self.path, index, |item| match item {
            MenuItem::Button { id, .. } => Some(MenuEvent::Activated(id.clone())),
            MenuItem::Toggle { id, on, .. } => {
                *on = !*on;
                Some(MenuEvent::Toggled(id.clone(), *on))
            }
            MenuItem::Choice { id, options, selected, .. } => {
                if options.is_empty() {
                    return None;
                }
                *selected = (*selected + 1) % options.len();
                Some(MenuEvent::Chose(id.clone(), *selected))
            }
            MenuItem::Submenu { .. } | MenuItem::Separator => None,
        })
        .flatten()
    }

    fn back(&mut self) -> Option<MenuEvent> {
        match self.path.pop() {
            Some(frame) => {
                // Restore where the user was, not the top of the list.
                self.selected = frame.restore;
                self.scroll = 0;
                None
            }
            None => {
                self.open = false;
                Some(MenuEvent::Closed)
            }
        }
    }

    /// Run `f` on the item at `index` within the level addressed by `path`.
    ///
    /// Recursive rather than an iterative `&mut` descent: walking a nested
    /// structure by reassigning a `&mut` in a loop trips the borrow checker
    /// unless every branch moves the borrow out, and the workarounds all involve
    /// an `unreachable!()`. Recursion hands the borrow down naturally, and the
    /// depth is bounded by `MAX_DEPTH` because that is what limits `path`.
    fn apply_at<R>(
        items: &mut [MenuItem],
        path: &[Frame],
        index: usize,
        f: impl FnOnce(&mut MenuItem) -> R,
    ) -> Option<R> {
        match path.split_first() {
            None => items.get_mut(index).map(f),
            Some((frame, rest)) => match items.get_mut(frame.entered) {
                Some(MenuItem::Submenu { items: nested, .. }) => {
                    Self::apply_at(nested, rest, index, f)
                }
                _ => None,
            },
        }
    }

    // --- Drawing ---

    /// Panel width and height in cells for the current level, before clamping.
    fn desired_size(&self) -> (u32, u32) {
        let items = self.current_items();
        let widest_item = items
            .iter()
            .map(|item| font5x7::text_width(&item.display()))
            .max()
            .unwrap_or(0);
        let content = widest_item.max(font5x7::text_width(&self.title));
        // 1 border + 1 pad on each side, plus room for the selection marker.
        let width = content as u32 + 6;
        // Border, title, separator row, items, border.
        let height = items.len() as u32 + 5;
        (width, height)
    }

    /// Draw the menu centred in `overlay`. A no-op while closed.
    ///
    /// The panel is clamped to the overlay, and when the level has more items
    /// than fit, the view scrolls to keep the selection visible. Both matter
    /// because the glyph grid is small and a menu can legitimately be taller
    /// than it.
    pub fn draw(&self, overlay: &mut Overlay) {
        if !self.open || overlay.cols() == 0 || overlay.rows() == 0 {
            return;
        }

        const FRAME: [f32; 3] = [0.55, 0.75, 0.95];
        const TITLE: [f32; 3] = [1.0, 0.95, 0.6];
        const ITEM: [f32; 3] = [0.85, 0.9, 0.85];
        const SELECTED: [f32; 3] = [0.15, 0.15, 0.2];
        const SELECTED_BG: [f32; 3] = [0.85, 0.85, 0.4];
        const PANEL_BG: [f32; 3] = [0.04, 0.05, 0.08];

        let (want_w, want_h) = self.desired_size();
        let width = want_w.min(overlay.cols()).max(1);
        let height = want_h.min(overlay.rows()).max(1);
        let col = (overlay.cols() - width) / 2;
        let row = (overlay.rows() - height) / 2;

        overlay.fill_rect(col, row, width, height, overlay.background_cell(PANEL_BG));
        overlay.draw_box(col, row, width, height, FRAME);

        // Title, centred inside the frame.
        let inner_w = width.saturating_sub(2);
        if inner_w > 0 && height > 1 {
            let title_w = font5x7::text_width(&self.title).min(inner_w as usize) as u32;
            let title_col = col + 1 + (inner_w - title_w) / 2;
            overlay.draw_text(title_col, row + 1, &self.title, TITLE);
        }

        // Rows available for items: inside the frame, below title + divider.
        let first_item_row = row + 3;
        let last_row = row + height - 1;
        if first_item_row >= last_row {
            return; // panel too short for any item
        }
        let visible = (last_row - first_item_row) as usize;

        // Divider under the title.
        if height > 4 {
            for x in (col + 1)..(col + width - 1) {
                overlay.set_cell(x, row + 2, OverlayCell::new(overlay.glyph_index_of('─'), FRAME));
            }
        }

        let items = self.current_items();
        let start = self.visible_start(items.len(), visible);

        for (offset, item) in items.iter().skip(start).take(visible).enumerate() {
            let index = start + offset;
            let y = first_item_row + offset as u32;

            if matches!(item, MenuItem::Separator) {
                for x in (col + 1)..(col + width - 1) {
                    overlay.set_cell(x, y, OverlayCell::new(overlay.glyph_index_of('─'), FRAME));
                }
                continue;
            }

            let is_selected = self.selected_index() == Some(index);
            if is_selected {
                // Highlight the whole row so the selection is unmistakable at
                // this glyph size.
                overlay.fill_rect(col + 1, y, width - 2, 1, overlay.background_cell(SELECTED_BG));
            }
            let color = if is_selected { SELECTED } else { ITEM };
            let marker = if is_selected { '\u{25BA}' } else { ' ' };
            overlay.set_cell(col + 1, y, OverlayCell::new(overlay.glyph_index_of(marker), color));
            overlay.draw_text(col + 3, y, &item.display(), color);
        }

        // Scroll indicators, so a clipped list does not look like the whole list.
        if start > 0 {
            overlay.set_cell(
                col + width - 2,
                first_item_row,
                OverlayCell::new(overlay.glyph_index_of('\u{25B2}'), FRAME),
            );
        }
        if start + visible < items.len() {
            overlay.set_cell(
                col + width - 2,
                last_row - 1,
                OverlayCell::new(overlay.glyph_index_of('\u{25BC}'), FRAME),
            );
        }
    }

    /// First item index to display so the selection stays visible.
    fn visible_start(&self, len: usize, visible: usize) -> usize {
        if visible == 0 || len <= visible {
            return 0;
        }
        let selected = self.selected.min(len.saturating_sub(1));
        // Keep the selection inside the window, scrolling the minimum needed.
        let max_start = len - visible;
        if selected < self.scroll {
            selected.min(max_start)
        } else if selected >= self.scroll + visible {
            (selected + 1 - visible).min(max_start)
        } else {
            self.scroll.min(max_start)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Menu {
        Menu::new(
            "SETTINGS",
            vec![
                MenuItem::button("Resume", "resume"),
                MenuItem::Separator,
                MenuItem::toggle("Post FX", "post", false),
                MenuItem::choice(
                    "Palette",
                    "palette",
                    vec!["True".into(), "256".into(), "16".into()],
                    0,
                ),
                MenuItem::submenu(
                    "Graphics",
                    vec![
                        MenuItem::toggle("Bloom", "bloom", true),
                        MenuItem::button("Reset", "reset"),
                    ],
                ),
                MenuItem::Separator,
                MenuItem::button("Quit", "quit"),
            ],
        )
    }

    #[test]
    fn starts_closed_and_on_the_first_selectable_item() {
        let menu = sample();
        assert!(!menu.is_open());
        assert_eq!(menu.selected_index(), Some(0));
        assert_eq!(menu.depth(), 0);
        assert_eq!(menu.title(), "SETTINGS");
    }

    #[test]
    fn actions_are_ignored_while_closed() {
        let mut menu = sample();
        assert_eq!(menu.handle(MenuAction::Down), None);
        assert_eq!(menu.selected_index(), Some(0), "selection must not move");
        assert_eq!(menu.handle(MenuAction::Activate), None);
    }

    #[test]
    fn navigation_skips_separators_in_both_directions() {
        let mut menu = sample();
        menu.open();
        // Index 1 is a separator, so Down from 0 must land on 2.
        menu.handle(MenuAction::Down);
        assert_eq!(menu.selected_index(), Some(2));
        // And Up must come back over it.
        menu.handle(MenuAction::Up);
        assert_eq!(menu.selected_index(), Some(0));
    }

    #[test]
    fn navigation_wraps_around_both_ends() {
        let mut menu = sample();
        menu.open();
        // Up from the first item wraps to the last selectable one (index 6).
        menu.handle(MenuAction::Up);
        assert_eq!(menu.selected_index(), Some(6));
        // Down from there wraps back to the first.
        menu.handle(MenuAction::Down);
        assert_eq!(menu.selected_index(), Some(0));
    }

    #[test]
    fn a_level_of_only_separators_has_no_selection_and_terminates() {
        let mut menu = Menu::new("EMPTY", vec![MenuItem::Separator, MenuItem::Separator]);
        menu.open();
        assert_eq!(menu.selected_index(), None);
        // Must return rather than scanning forever.
        menu.handle(MenuAction::Down);
        menu.handle(MenuAction::Up);
        assert_eq!(menu.selected_index(), None);
        assert_eq!(menu.handle(MenuAction::Activate), None);
    }

    #[test]
    fn empty_menu_is_safe() {
        let mut menu = Menu::new("NOTHING", Vec::new());
        menu.open();
        assert_eq!(menu.selected_index(), None);
        menu.handle(MenuAction::Down);
        assert_eq!(menu.handle(MenuAction::Activate), None);
        let mut overlay = Overlay::new(20, 10);
        menu.draw(&mut overlay);
    }

    #[test]
    fn button_activation_reports_its_id() {
        let mut menu = sample();
        menu.open();
        assert_eq!(
            menu.handle(MenuAction::Activate),
            Some(MenuEvent::Activated("resume".into()))
        );
    }

    #[test]
    fn toggle_flips_and_reports_the_new_state() {
        let mut menu = sample();
        menu.open();
        menu.handle(MenuAction::Down); // -> Post FX
        assert_eq!(
            menu.handle(MenuAction::Activate),
            Some(MenuEvent::Toggled("post".into(), true))
        );
        assert_eq!(
            menu.handle(MenuAction::Activate),
            Some(MenuEvent::Toggled("post".into(), false))
        );
        // And the item itself carries the state.
        assert_eq!(
            menu.item_state("post"),
            Some(&MenuItem::Toggle { label: "Post FX".into(), id: "post".into(), on: false })
        );
    }

    #[test]
    fn choice_cycles_and_wraps() {
        let mut menu = sample();
        menu.open();
        menu.handle(MenuAction::Down);
        menu.handle(MenuAction::Down); // -> Palette
        assert_eq!(menu.handle(MenuAction::Activate), Some(MenuEvent::Chose("palette".into(), 1)));
        assert_eq!(menu.handle(MenuAction::Activate), Some(MenuEvent::Chose("palette".into(), 2)));
        assert_eq!(
            menu.handle(MenuAction::Activate),
            Some(MenuEvent::Chose("palette".into(), 0)),
            "must wrap back to the first option"
        );
    }

    #[test]
    fn choice_with_no_options_is_inert() {
        let mut menu = Menu::new("C", vec![MenuItem::choice("Empty", "e", Vec::new(), 5)]);
        menu.open();
        assert_eq!(menu.handle(MenuAction::Activate), None);
    }

    #[test]
    fn choice_constructor_clamps_the_initial_index() {
        let item = MenuItem::choice("X", "x", vec!["a".into(), "b".into()], 99);
        assert_eq!(item, MenuItem::Choice {
            label: "X".into(),
            id: "x".into(),
            options: vec!["a".into(), "b".into()],
            selected: 1,
        });
    }

    #[test]
    fn submenu_descends_and_back_restores_the_parent_selection() {
        let mut menu = sample();
        menu.open();
        // Walk down to Graphics (index 4).
        menu.handle(MenuAction::Down); // 2
        menu.handle(MenuAction::Down); // 3
        menu.handle(MenuAction::Down); // 4
        assert_eq!(menu.selected_index(), Some(4));

        assert_eq!(menu.handle(MenuAction::Activate), None, "descending emits no event");
        assert_eq!(menu.depth(), 1);
        assert_eq!(menu.current_items().len(), 2);
        assert_eq!(menu.selected_index(), Some(0));

        // A toggle inside the submenu works on the nested item.
        assert_eq!(
            menu.handle(MenuAction::Activate),
            Some(MenuEvent::Toggled("bloom".into(), false))
        );

        // Back returns to the parent, restoring the row we came from.
        assert_eq!(menu.handle(MenuAction::Back), None);
        assert_eq!(menu.depth(), 0);
        assert_eq!(menu.selected_index(), Some(4), "must not reset to the first item");
    }

    #[test]
    fn back_at_the_top_level_closes_the_menu() {
        let mut menu = sample();
        menu.open();
        assert_eq!(menu.handle(MenuAction::Back), Some(MenuEvent::Closed));
        assert!(!menu.is_open());
    }

    #[test]
    fn item_state_finds_nested_ids_and_rejects_unknown_ones() {
        let menu = sample();
        assert!(menu.item_state("bloom").is_some(), "nested id must be found");
        assert!(menu.item_state("quit").is_some());
        assert_eq!(menu.item_state("nope"), None);
        // Submenus and separators have no id.
        assert_eq!(menu.item_state("Graphics"), None);
    }

    #[test]
    fn setters_write_state_by_id_including_nested_items() {
        let mut menu = sample();
        assert!(menu.set_toggle("post", true));
        assert!(matches!(menu.item_state("post"), Some(MenuItem::Toggle { on: true, .. })));

        assert!(menu.set_choice("palette", 2));
        assert!(matches!(
            menu.item_state("palette"),
            Some(MenuItem::Choice { selected: 2, .. })
        ));
        // Out-of-range index clamps rather than panicking later during draw.
        assert!(menu.set_choice("palette", 99));
        assert!(matches!(
            menu.item_state("palette"),
            Some(MenuItem::Choice { selected: 2, .. })
        ));

        // Nested item.
        assert!(menu.set_toggle("bloom", false));
        assert!(matches!(menu.item_state("bloom"), Some(MenuItem::Toggle { on: false, .. })));

        // Unknown id, and an id whose item is the wrong kind.
        assert!(!menu.set_toggle("nope", true));
        assert!(!menu.set_toggle("palette", true), "a Choice is not a Toggle");
        assert!(!menu.set_choice("post", 1), "a Toggle is not a Choice");
    }

    #[test]
    fn draw_is_a_no_op_while_closed() {
        let menu = sample();
        let mut overlay = Overlay::new(60, 30);
        menu.draw(&mut overlay);
        assert!(overlay.is_empty(), "closed menu must not paint anything");
    }

    #[test]
    fn draw_paints_a_centred_panel_when_open() {
        let mut menu = sample();
        menu.open();
        let mut overlay = Overlay::new(60, 30);
        menu.draw(&mut overlay);
        assert!(!overlay.is_empty());

        // The centre of the overlay must be inside the panel.
        let centre = overlay.cell(30, 15).unwrap();
        assert!(centre.opaque, "panel should cover the overlay centre");
        // The far corner must be outside it.
        assert!(!overlay.cell(0, 0).unwrap().opaque, "panel must not fill the screen");
    }

    #[test]
    fn draw_survives_an_overlay_smaller_than_the_panel() {
        let mut menu = sample();
        menu.open();
        for (cols, rows) in [(1u32, 1u32), (3, 2), (8, 4), (12, 6)] {
            let mut overlay = Overlay::new(cols, rows);
            menu.draw(&mut overlay); // must not panic
        }
        // A zero-sized overlay is also legal.
        let mut overlay = Overlay::new(0, 0);
        menu.draw(&mut overlay);
    }

    #[test]
    fn scrolling_keeps_the_selection_visible() {
        let items: Vec<MenuItem> = (0..30)
            .map(|i| MenuItem::button(format!("Item {i}"), format!("id{i}")))
            .collect();
        let mut menu = Menu::new("LONG", items);
        menu.open();

        // Only a handful of rows fit in a short overlay.
        let visible = 5usize;

        // Selection near the start needs no scrolling.
        assert_eq!(menu.visible_start(30, visible), 0);

        // Move well past the window and the start must follow.
        for _ in 0..10 {
            menu.handle(MenuAction::Down);
        }
        assert_eq!(menu.selected_index(), Some(10));
        let start = menu.visible_start(30, visible);
        assert!(start <= 10 && 10 < start + visible, "selection {} outside [{start}, {})", 10, start + visible);

        // And the window never runs past the end of the list.
        for _ in 0..25 {
            menu.handle(MenuAction::Down);
        }
        let start = menu.visible_start(30, visible);
        assert!(start + visible <= 30, "window overshot the list: {start} + {visible}");
    }

    #[test]
    fn drawing_a_long_menu_stays_inside_the_overlay() {
        let items: Vec<MenuItem> = (0..40)
            .map(|i| MenuItem::button(format!("Entry number {i}"), format!("id{i}")))
            .collect();
        let mut menu = Menu::new("VERY LONG MENU TITLE", items);
        menu.open();
        let mut overlay = Overlay::new(30, 12);
        menu.draw(&mut overlay);
        // Every painted cell must be inside the overlay by construction; the real
        // check is that drawing a panel larger than the surface does not panic and
        // still produces output.
        assert!(!overlay.is_empty());
    }

    #[test]
    fn submenu_nesting_is_bounded() {
        // Build a chain deeper than MAX_DEPTH and confirm descending stops.
        let mut deepest = vec![MenuItem::button("Leaf", "leaf")];
        for i in 0..(MAX_DEPTH + 4) {
            deepest = vec![MenuItem::submenu(format!("Level {i}"), deepest)];
        }
        let mut menu = Menu::new("DEEP", deepest);
        menu.open();
        for _ in 0..(MAX_DEPTH + 10) {
            menu.handle(MenuAction::Activate);
        }
        assert!(menu.depth() <= MAX_DEPTH, "depth {} exceeded the bound", menu.depth());
    }
}
