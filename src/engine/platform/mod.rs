// Platform abstraction layer — wraps winit event loop.
// Uses the winit 0.30 ApplicationHandler trait.

use crate::engine::core::Result;
use crate::engine::core::EngineError;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

/// Helper to create a window from an active event loop.
pub fn create_window(elwt: &ActiveEventLoop, title: &str, width: u32, height: u32) -> Result<Window> {
    elwt.create_window(
        WindowAttributes::default()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height)),
    )
    .map_err(|e| EngineError::Platform(e.to_string()))
}