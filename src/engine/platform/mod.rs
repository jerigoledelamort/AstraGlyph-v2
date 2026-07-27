// Platform abstraction layer — wraps winit window and event handling.

use crate::engine::core::Result;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};

/// Wraps a winit window and event loop.
pub struct Platform {
    event_loop: EventLoop<()>,
    window: Window,
}

impl Platform {
    /// Create a new window with the given title, width, and height.
    pub fn new(title: &str, width: u32, height: u32) -> Result<Self> {
        let event_loop = EventLoop::new()
            .map_err(|e| crate::engine::core::EngineError::Platform(e.to_string()))?;

        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title(title)
                    .with_inner_size(winit::dpi::LogicalSize::new(width, height)),
            )
            .map_err(|e| crate::engine::core::EngineError::Platform(e.to_string()))?;

        Ok(Self { event_loop, window })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Surface size in logical pixels.
    pub fn size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }

    /// Move the event loop to completion.
    /// `handler` is called on each event; the window is moved into the closure.
    pub fn run<F>(self, mut handler: F)
    where
        F: 'static + FnMut(&ActiveEventLoop, winit::event::Event<()>, &Window),
    {
        let window = self.window;
        self.event_loop
            .run(move |event, elwt| handler(elwt, event, &window))
            .expect("event loop failed")
    }
}