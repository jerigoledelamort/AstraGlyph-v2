// Graphics context: wgpu instance, adapter, device, queue, surface.

use crate::engine::core::{block_on, EngineError, Result};
use wgpu::{Adapter, Device, Instance, Queue, Surface};
use wgpu::SurfaceConfiguration;

/// Result of asking the surface for a frame.
pub enum FrameOutcome {
    /// A frame is ready to render into.
    Frame(wgpu::SurfaceTexture),
    /// No frame this time, but the situation is transient (the window was
    /// resized, minimized, occluded, or the swapchain went stale). The surface
    /// has been reconfigured where that helps; just skip this frame.
    Skip(&'static str),
}

/// Holds all wgpu context needed for rendering.
pub struct GraphicsContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    // Surface borrows the window; we extend the lifetime to 'static because
    // the window outlives the GraphicsContext (owned by AppState, which is
    // dropped before the window).
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub width: u32,
    pub height: u32,
}

impl GraphicsContext {
    /// Initialize wgpu context from a winit window.
    ///
    /// Uses the `block_on` utility to drive the async wgpu init futures synchronously.
    ///
    /// # Safety
    /// The caller must ensure the window outlives the GraphicsContext.
    pub fn new(
        window: &winit::window::Window,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let instance = Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let surface = instance
            .create_surface(window)
            .map_err(|e| EngineError::Graphics(e.to_string()))?;

        // SAFETY: The window is owned by Platform and outlives AppState.
        // We extend the surface lifetime to 'static to avoid complex lifetime
        // propagation through AppState.
        let surface: Surface<'static> = unsafe { std::mem::transmute(surface) };

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(|e| EngineError::Graphics(e.to_string()))?;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("AstraGlyph Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| EngineError::Graphics(e.to_string()))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            config,
            width,
            height,
        })
    }

    /// Reconfigure the surface after a window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.width = width;
            self.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Get the current frame texture for rendering.
    ///
    /// Every non-success state a desktop surface produces is recoverable: a
    /// resize or minimize invalidates the swapchain, and occlusion just means
    /// nothing is visible right now. These used to be returned as hard errors,
    /// which made the caller tear the application down on an ordinary window
    /// event — so they are reported as `Skip` and the surface is reconfigured
    /// instead.
    pub fn current_frame(&mut self) -> FrameOutcome {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => FrameOutcome::Frame(frame),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => FrameOutcome::Frame(frame),
            wgpu::CurrentSurfaceTexture::Timeout => FrameOutcome::Skip("timeout"),
            wgpu::CurrentSurfaceTexture::Occluded => FrameOutcome::Skip("occluded"),
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.reconfigure();
                FrameOutcome::Skip("outdated")
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure();
                FrameOutcome::Skip("lost")
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                self.reconfigure();
                FrameOutcome::Skip("validation")
            }
        }
    }

    /// Re-apply the current configuration to the surface.
    pub fn reconfigure(&mut self) {
        if self.config.width > 0 && self.config.height > 0 {
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Whether the surface currently has a drawable size. A minimized window
    /// reports 0x0, and rendering into that is invalid.
    pub fn is_renderable(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Surface size (width, height).
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}