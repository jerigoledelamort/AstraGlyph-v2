// AstraGlyph — entry point.
// ASCII rendering engine: 3D scenes rendered as colored ASCII art.

#![allow(dead_code)]

mod app;
mod ascii;
mod demo;
mod engine;
mod graphics;
mod renderer;
mod scene;

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
                        ..
                    },
                ..
            } => {
                if let Some(s) = self.state.borrow_mut().as_mut() {
                    s.input_mut().key_event(code, key_state);
                }
                if code == KeyCode::Escape && key_state == ElementState::Pressed {
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
