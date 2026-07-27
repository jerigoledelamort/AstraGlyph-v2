// AstraGlyph — entry point.
// ASCII rendering engine: 3D scenes rendered as colored ASCII art.

mod app;
mod ascii;
mod demo;
mod engine;
mod graphics;
mod renderer;
mod scene;

use engine::core::Result;
use winit::event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

fn main() -> Result<()> {
    let platform = engine::platform::Platform::new("AstraGlyph", 1280, 720)?;

    use std::cell::RefCell;
    let state = RefCell::new(None::<app::state::AppState>);

    platform.run(move |elwt, event, window| {
        match event {
            Event::NewEvents(winit::event::StartCause::Init) => {
                let size = window.inner_size();
                match app::state::AppState::new(window, size.width, size.height) {
                    Ok(s) => *state.borrow_mut() = Some(s),
                    Err(e) => {
                        eprintln!("Failed to initialize: {e}");
                        elwt.exit();
                    }
                }
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }

                WindowEvent::Resized(physical_size) => {
                    if let Some(s) = state.borrow_mut().as_mut() {
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
                    if let Some(s) = state.borrow_mut().as_mut() {
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
                    if let Some(s) = state.borrow_mut().as_mut() {
                        s.input_mut().mouse_button_event(button, mouse_state);
                    }
                }

                _ => {}
            },

            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if let Some(s) = state.borrow_mut().as_mut() {
                    s.input_mut().mouse_motion(delta.0, delta.1);
                }
            }

            Event::AboutToWait => {
                if let Some(s) = state.borrow_mut().as_mut() {
                    if let Err(e) = s.render() {
                        eprintln!("Render error: {e}");
                        elwt.exit();
                    }
                }
                window.request_redraw();
            }

            _ => {}
        }
    });

    Ok(())
}
