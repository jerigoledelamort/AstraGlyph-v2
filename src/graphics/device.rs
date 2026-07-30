// Graphics context: wgpu instance, adapter, device, queue, surface.

use crate::engine::core::{block_on, EngineError, Result};
use crate::graphics::capabilities::{self, RayTracingStatus};
use wgpu::{Adapter, Device, Instance, Queue, Surface};
use wgpu::SurfaceConfiguration;

/// Default limits with the acceleration-structure entries raised off zero.
///
/// Kept as a free function next to the only call site because it is a property
/// of *this* engine's needs (one TLAS, one geometry group per BLAS), not a
/// general capability query.
fn acceleration_structure_limits(adapter: &wgpu::Limits) -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    limits.max_tlas_instance_count = capabilities::requested_limit(
        capabilities::REQUESTED_TLAS_INSTANCES,
        adapter.max_tlas_instance_count,
    );
    limits.max_blas_primitive_count = capabilities::requested_limit(
        capabilities::REQUESTED_BLAS_PRIMITIVES,
        adapter.max_blas_primitive_count,
    );
    limits.max_blas_geometry_count = capabilities::requested_limit(
        capabilities::REQUESTED_BLAS_GEOMETRIES,
        adapter.max_blas_geometry_count,
    );
    limits.max_acceleration_structures_per_shader_stage = capabilities::requested_limit(
        capabilities::REQUESTED_ACCELERATION_STRUCTURES,
        adapter.max_acceleration_structures_per_shader_stage,
    );
    limits
}

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
    /// Whether hardware ray query is active, and if not, why not. Decided once
    /// here so no consumer has to re-derive it (see `graphics::capabilities`).
    ray_tracing: RayTracingStatus,
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

        // Ray query is optional: ask for it only when the adapter advertises it
        // and the environment has not vetoed it. Requesting an unsupported
        // feature makes `request_device` fail outright, so the negotiation has
        // to happen before the request, not after.
        let ray_query = wgpu::Features::from(wgpu::FeaturesWGPU::EXPERIMENTAL_RAY_QUERY);
        let adapter_features = adapter.features();
        let adapter_supports = adapter_features.contains(ray_query);

        // Timestamp queries, for honest per-pass GPU timings (Phase 6.3). Optional
        // and requested only if present: asking for a feature the adapter lacks
        // fails the whole device request, which would trade a profiler for a
        // renderer.
        let timestamps = wgpu::Features::from(wgpu::FeaturesWebGPU::TIMESTAMP_QUERY);
        let want_timestamps = adapter_features.contains(timestamps);
        let vetoed = capabilities::ray_tracing_vetoed();
        let request_ray_query = capabilities::should_request_ray_query(adapter_supports, vetoed);

        let info = adapter.get_info();
        eprintln!(
            "gpu: {} ({:?}, {:?}) driver \"{}\"",
            info.name, info.device_type, info.backend, info.driver_info
        );

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("AstraGlyph Device"),
            required_features: {
                let mut features = wgpu::Features::empty();
                if request_ray_query {
                    features |= ray_query;
                }
                if want_timestamps {
                    features |= timestamps;
                }
                features
            },
            required_limits: if request_ray_query {
                acceleration_structure_limits(&adapter.limits())
            } else {
                wgpu::Limits::default()
            },
            // EXPERIMENTAL-prefixed features are gated behind this token; with
            // the default (disabled) value the RAY_QUERY request above would be
            // rejected regardless of hardware support.
            //
            // SAFETY: the token only acknowledges that wgpu's ray-tracing
            // implementation is work-in-progress and may contain soundness bugs.
            // The engine handles that by keeping the rasterised path intact and
            // switchable at runtime, so a misbehaving driver degrades to the
            // fallback rather than being the only option.
            experimental_features: if request_ray_query {
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            },
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| EngineError::Graphics(e.to_string()))?;

        // What the *device* granted, not what was asked for: a device may hand
        // back fewer features than requested, and treating that as success would
        // send ray queries at a device that cannot run them.
        // A zero instance budget is as disqualifying as a missing feature: the
        // TLAS could not hold a single object, so there would be nothing to trace
        // against even though the ray-query calls themselves would validate.
        let tlas_capacity = device.limits().max_tlas_instance_count;
        let granted = device.features().contains(ray_query) && tlas_capacity > 0;
        let ray_tracing = capabilities::classify(adapter_supports, vetoed, granted);
        eprintln!(
            "raytracing: {} (tlas capacity {tlas_capacity})",
            ray_tracing.describe()
        );
        eprintln!(
            "profiler: gpu timestamps {}",
            if device.features().contains(timestamps) {
                "available"
            } else {
                "unavailable, GPU times will be wall-clock estimates"
            }
        );

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
            // No vsync: the engine is a rendering testbed, so an uncapped frame
            // rate is the useful default — it makes the cost of a change visible
            // instead of hiding it behind the refresh interval.
            present_mode: wgpu::PresentMode::AutoNoVsync,
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
            ray_tracing,
        })
    }

    /// Whether hardware ray query is active, and if not, why not.
    pub fn ray_tracing(&self) -> RayTracingStatus {
        self.ray_tracing
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