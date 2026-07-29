// Input state: tracks keyboard and mouse for first-person camera control.

use std::collections::HashSet;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

/// Tracks the current input state (keys, mouse buttons, mouse motion).
#[derive(Default)]
pub struct InputState {
    /// Set of currently-pressed key codes.
    pressed_keys: HashSet<KeyCode>,
    /// Set of currently-pressed mouse buttons.
    pressed_mouse: HashSet<MouseButton>,
    /// Accumulated mouse motion delta since last frame.
    mouse_delta: (f64, f64),
    /// Mouse wheel delta.
    mouse_wheel: f32,
    /// Characters typed since the last frame, in order.
    ///
    /// Kept separate from `pressed_keys` because text entry needs the LOGICAL
    /// character (respecting layout, shift and dead keys), while movement needs
    /// the physical key position. The console consumes this; the camera does not.
    typed: String,
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a key state change.
    pub fn key_event(&mut self, key: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_keys.insert(key);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&key);
            }
        }
    }

    /// Record a mouse button state change.
    pub fn mouse_button_event(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_mouse.insert(button);
            }
            ElementState::Released => {
                self.pressed_mouse.remove(&button);
            }
        }
    }

    /// Accumulate mouse motion.
    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Record mouse wheel scroll.
    pub fn mouse_wheel(&mut self, delta: f32) {
        self.mouse_wheel += delta;
    }

    /// Record text produced by a key press. Control characters are dropped here
    /// so consumers never have to filter them.
    pub fn text_input(&mut self, text: &str) {
        for c in text.chars() {
            if !c.is_control() {
                self.typed.push(c);
            }
        }
    }

    /// Consume the characters typed since the last call.
    pub fn take_typed(&mut self) -> String {
        std::mem::take(&mut self.typed)
    }

    /// Drop any pending text without consuming it as input — used when focus
    /// moves so stale keystrokes do not appear in a newly opened field.
    pub fn clear_typed(&mut self) {
        self.typed.clear();
    }

    /// Check if a key is currently pressed.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    /// Check if a mouse button is currently pressed.
    pub fn is_mouse_pressed(&self, button: MouseButton) -> bool {
        self.pressed_mouse.contains(&button)
    }

    /// Consume and return the accumulated mouse delta.
    pub fn take_mouse_delta(&mut self) -> (f64, f64) {
        let delta = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        delta
    }

    /// Consume and return the accumulated mouse wheel delta.
    pub fn take_mouse_wheel(&mut self) -> f32 {
        let w = self.mouse_wheel;
        self.mouse_wheel = 0.0;
        w
    }

    /// Is left mouse button held? (used for camera look).
    pub fn is_look_active(&self) -> bool {
        self.pressed_mouse.contains(&MouseButton::Left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_press_release() {
        let mut input = InputState::new();
        assert!(!input.is_key_pressed(KeyCode::KeyW));
        input.key_event(KeyCode::KeyW, ElementState::Pressed);
        assert!(input.is_key_pressed(KeyCode::KeyW));
        input.key_event(KeyCode::KeyW, ElementState::Released);
        assert!(!input.is_key_pressed(KeyCode::KeyW));
    }

    #[test]
    fn mouse_delta_accumulation() {
        let mut input = InputState::new();
        input.mouse_motion(10.0, 5.0);
        input.mouse_motion(3.0, 2.0);
        let delta = input.take_mouse_delta();
        assert_eq!(delta, (13.0, 7.0));
        // After taking, should be reset.
        let delta2 = input.take_mouse_delta();
        assert_eq!(delta2, (0.0, 0.0));
    }

    #[test]
    fn typed_text_accumulates_and_drops_control_characters() {
        let mut input = InputState::new();
        assert_eq!(input.take_typed(), "");
        input.text_input("he");
        input.text_input("llo");
        // Control characters (Enter, Tab, Backspace) must not reach the console as
        // text — they arrive as key codes and mean actions, not content.
        input.text_input("\r\n\t\u{8}");
        input.text_input("!");
        assert_eq!(input.take_typed(), "hello!");
        assert_eq!(input.take_typed(), "", "taking must drain the buffer");
    }

    #[test]
    fn clear_typed_discards_pending_text() {
        let mut input = InputState::new();
        input.text_input("stale");
        input.clear_typed();
        assert_eq!(input.take_typed(), "");
    }

    #[test]
    fn mouse_button_look() {
        let mut input = InputState::new();
        assert!(!input.is_look_active());
        input.mouse_button_event(MouseButton::Left, ElementState::Pressed);
        assert!(input.is_look_active());
        input.mouse_button_event(MouseButton::Left, ElementState::Released);
        assert!(!input.is_look_active());
    }
}