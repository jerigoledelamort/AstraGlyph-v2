// AstraGlyph — entry point.
// ASCII rendering engine: 3D scenes rendered as colored ASCII art.

// Most of what this silences is API surface exercised by unit tests but not (yet) by
// the engine itself: `Vec2`, the geometry module's shape-shape helpers, mixer and
// interpreter accessors. Auditing with it off produced 120 warnings, of which exactly
// one was a real finding — the PNG decoder was never called at runtime, so it is now
// invoked at startup on whatever is in `assets/textures` (see `AppState::load_texture`).
//
// The rest are genuinely-tested code paths kept for the callers they will have, and
// turning the lint on would mean sprinkling per-item allows to no benefit. Worth
// re-auditing the same way when a phase lands, since a blanket allow is exactly how a
// module ends up existing without ever being run.
#![allow(dead_code)]

mod app;
mod ascii;
mod assets;
mod audio;
mod demo;
mod engine;
mod graphics;
mod physics;
mod renderer;
mod scripting;
mod scene;
mod ui;

use engine::core::Result;
use engine::platform;
use std::cell::RefCell;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

/// Main application implementing winit 0.30 ApplicationHandler.
struct MainApp {
    title: String,
    width: u32,
    height: u32,
    state: RefCell<Option<app::state::AppState>>,
    window: Option<Window>,
}

impl ApplicationHandler for MainApp {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = platform::create_window(elwt, &self.title, self.width, self.height)
            .expect("Failed to create window");
        let size = window.inner_size();
        let state = app::state::AppState::new(&window, size.width, size.height)
            .expect("Failed to initialize AppState");
        self.window = Some(window);
        *self.state.borrow_mut() = Some(state);
    }

    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                elwt.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(s) = self.state.borrow_mut().as_mut() {
                    s.resize(physical_size.width, physical_size.height);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        text,
                        ..
                    },
                ..
            } => {
                let mut consumed_escape = false;
                if let Some(s) = self.state.borrow_mut().as_mut() {
                    s.input_mut().key_event(code, key_state);
                    // Forward the logical text so the console can be typed into.
                    // Only on press, and only when a UI layer wants it — otherwise
                    // movement keys would queue up as stray characters.
                    if key_state == ElementState::Pressed && s.ui_wants_text() {
                        if let Some(text) = text.as_ref() {
                            s.input_mut().text_input(text);
                        }
                    }
                    if code == KeyCode::Escape && key_state == ElementState::Pressed {
                        // Escape closes an open menu/console first; only a second
                        // press with nothing open quits.
                        consumed_escape = s.handle_escape();
                    }
                }
                if code == KeyCode::Escape && key_state == ElementState::Pressed && !consumed_escape
                {
                    elwt.exit();
                }
            }
            WindowEvent::MouseInput {
                state: mouse_state,
                button,
                ..
            } => {
                if let Some(s) = self.state.borrow_mut().as_mut() {
                    s.input_mut().mouse_button_event(button, mouse_state);
                }
            }
            WindowEvent::Focused(false) => {
                // Releases that happen while another window has focus never reach
                // us, so anything held right now would stay stuck down forever.
                if let Some(s) = self.state.borrow_mut().as_mut() {
                    s.input_mut().clear_all();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Line deltas come from a wheel, pixel deltas from a trackpad;
                // normalize both to "notches" so camera zoom feels the same.
                let notches = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 120.0,
                };
                if let Some(s) = self.state.borrow_mut().as_mut() {
                    s.input_mut().mouse_wheel(notches);
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _elwt: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if let Some(s) = self.state.borrow_mut().as_mut() {
                s.input_mut().mouse_motion(delta.0, delta.1);
            }
        }
    }

    fn about_to_wait(&mut self, elwt: &ActiveEventLoop) {
        if let Some(s) = self.state.borrow_mut().as_mut() {
            if let Err(e) = s.render() {
                eprintln!("Render error: {e}");
                elwt.exit();
            }
            // The menu's Quit entry and the console's `quit` command request
            // shutdown through the state rather than reaching for the event loop.
            if s.should_exit() {
                elwt.exit();
            }
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()
        .map_err(|e| engine::core::EngineError::Platform(e.to_string()))?;

    let mut app = MainApp {
        title: "AstraGlyph".to_string(),
        width: 1280,
        height: 720,
        state: RefCell::new(None),
        window: None,
    };

    event_loop.run_app(&mut app).expect("event loop failed");

    Ok(())
}
