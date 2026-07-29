// Application state: ties together graphics, scene, camera, renderer, and input.
// The main loop updates this state each frame.

use std::time::Instant;
use crate::engine::core::Result;
use crate::engine::math::{radians, Vec3};
use crate::graphics::GraphicsContext;
use crate::renderer::{AsciiProcessor, CompositePipeline, ScenePipeline};
use crate::scene::{Camera, MeshComponent, MaterialComponent, MaterialUniform, Scene};
use crate::demo::material_spheres;

use super::input::InputState;
use super::metrics::FrameMetrics;

/// First-person camera controller parameters.
struct CameraController {
    /// Movement speed (units per second).
    move_speed: f32,
    /// Mouse look sensitivity.
    look_sensitivity: f32,
    /// Current yaw (horizontal angle) in radians.
    yaw: f32,
    /// Current pitch (vertical angle) in radians.
    pitch: f32,
}

impl CameraController {
    fn new() -> Self {
        Self {
            move_speed: 15.0,
            look_sensitivity: 0.0025,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    /// Update camera position and orientation from input.
    fn update(&mut self, camera: &mut Camera, input: &mut InputState, dt: f32) {
        // Mouse look (only when LMB is held).
        if input.is_look_active() {
            let (dx, dy) = input.take_mouse_delta();
            self.yaw -= dx as f32 * self.look_sensitivity;
            self.pitch -= dy as f32 * self.look_sensitivity;
            // Clamp pitch to avoid flipping.
            self.pitch = self.pitch.clamp(-radians(89.0), radians(89.0));
        } else {
            // Consume delta to prevent accumulation.
            let _ = input.take_mouse_delta();
        }

        // Compute forward and right vectors from yaw/pitch.
        let forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.sin() * self.pitch.cos(),
        );
        let right = Vec3::new(
            self.yaw.sin(),
            0.0,
            self.yaw.cos(),
        );

        // WASD movement.
        let mut move_dir = Vec3::ZERO;
        if input.is_key_pressed(winit::keyboard::KeyCode::KeyW) {
            move_dir += forward;
        }
        if input.is_key_pressed(winit::keyboard::KeyCode::KeyS) {
            move_dir -= forward;
        }
        if input.is_key_pressed(winit::keyboard::KeyCode::KeyA) {
            move_dir -= right;
        }
        if input.is_key_pressed(winit::keyboard::KeyCode::KeyD) {
            move_dir += right;
        }
        // Vertical movement.
        if input.is_key_pressed(winit::keyboard::KeyCode::ShiftLeft) {
            move_dir += Vec3::UNIT_Y;
        }
        if input.is_key_pressed(winit::keyboard::KeyCode::ControlLeft) {
            move_dir -= Vec3::UNIT_Y;
        }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize() * self.move_speed * dt;
            camera.position += move_dir;
        }

        // Update camera target to look in the forward direction.
        camera.target = camera.position + forward;
    }
}

/// Main application state.
pub struct AppState {
    pub graphics: GraphicsContext,
    scene_pipeline: ScenePipeline,
    composite_pipeline: CompositePipeline,
    ascii_processor: AsciiProcessor,
    camera: Camera,
    camera_controller: CameraController,
    input: InputState,
    scene: Scene,
    last_frame: Instant,
    metrics: FrameMetrics,
    /// Grid resolution (matches the ASCII cell grid).
    grid_cols: u32,
    grid_rows: u32,
}

impl AppState {
    /// Initialize the application state from a winit window.
    pub fn new(
        window: &winit::window::Window,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let graphics = GraphicsContext::new(window, width, height)?;

        // ASCII grid resolution (low-res scene render target).
        let grid_cols = 120;
        let grid_rows = 68;

        let scene_format = wgpu::TextureFormat::Rgba8Unorm;
        let scene_pipeline = ScenePipeline::new(
            &graphics.device,
            grid_cols,
            grid_rows,
            scene_format,
        )?;

        let composite_pipeline = CompositePipeline::new(
            &graphics.device,
            graphics.config.format,
            grid_cols * grid_rows,
        )?;
        composite_pipeline.upload_atlas(&graphics.queue);

        let ascii_processor = AsciiProcessor::new(
            &graphics.device,
            grid_cols,
            grid_rows,
        );

        let (scene, camera) = material_spheres::build_scene();

        Ok(Self {
            graphics,
            scene_pipeline,
            composite_pipeline,
            ascii_processor,
            camera,
            camera_controller: CameraController::new(),
            input: InputState::new(),
            scene,
            last_frame: Instant::now(),
            metrics: FrameMetrics::new(),
            grid_cols,
            grid_rows,
        })
    }

    /// Handle a window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.graphics.resize(width, height);
        let aspect = width as f32 / height as f32;
        self.camera.set_aspect(aspect);
    }

    /// Get a mutable reference to the input state.
    pub fn input_mut(&mut self) -> &mut InputState {
        &mut self.input
    }

    /// Render one frame.
    pub fn render(&mut self) -> Result<()> {
        // Compute delta time.
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Begin metrics tracking.
        self.metrics.begin_frame();

        // Update camera from input.
        self.camera_controller.update(&mut self.camera, &mut self.input, dt);

        // Update scene uniforms.
        self.scene_pipeline.update_camera(&self.graphics.queue, &self.camera);
        let lights = material_spheres::lights();
        self.scene_pipeline.update_lights(&self.graphics.queue, &lights);

        // Collect all meshes with their material indices for batched rendering,
        // and the scene's world-space bounding box (needed to size the shadow
        // camera's frustum around whatever geometry is actually on screen).
        // Material index = position in the materials vec, matching the storage buffer.
        let mesh_entities = self.scene.entities_with::<MeshComponent>();
        let mut meshes: Vec<(_, _, u32)> = Vec::new();
        let mut materials: Vec<MaterialUniform> = Vec::new();
        let mut scene_min = Vec3::splat(f32::INFINITY);
        let mut scene_max = Vec3::splat(f32::NEG_INFINITY);
        for entity in mesh_entities {
            if let Some(mesh) = self.scene.get_component::<MeshComponent>(entity) {
                let mat_idx = if let Some(mat) = self.scene.get_component::<MaterialComponent>(entity) {
                    let idx = materials.len() as u32;
                    materials.push(MaterialUniform::from(mat));
                    idx
                } else {
                    0
                };
                for v in &mesh.vertices {
                    scene_min = Vec3::new(scene_min.x.min(v.position.x), scene_min.y.min(v.position.y), scene_min.z.min(v.position.z));
                    scene_max = Vec3::new(scene_max.x.max(v.position.x), scene_max.y.max(v.position.y), scene_max.z.max(v.position.z));
                }
                meshes.push((entity, mesh, mat_idx));
            }
        }

        // Upload materials to GPU.
        self.scene_pipeline.upload_materials(&self.graphics.queue, &materials);

        // Point the (simplified) shadow camera at light[0], sized to fit the scene.
        if let (Some(light0), true) = (lights.first(), scene_min.x.is_finite()) {
            let center = (scene_min + scene_max) * 0.5;
            let radius = (scene_max - scene_min).length() * 0.5;
            let light_vp = light0.shadow_view_proj(center, radius);
            self.scene_pipeline.update_shadow_camera(&self.graphics.queue, light_vp);
        }

        // Render all meshes in a single render pass (GPU phase 1: scene).
        let gpu_start = Instant::now();
        self.scene_pipeline.render_batched(
            &self.graphics.device,
            &self.graphics.queue,
            &meshes,
            &materials,
        )?;
        self.metrics.record_gpu_phase(gpu_start);

        // Read back the scene render target pixels (double-buffered, non-blocking —
        // may lag the current frame by ~1 frame, never stalls the CPU on the GPU).
        let pixels = self.ascii_processor.read_pixels(
            &self.graphics.device,
            &self.graphics.queue,
            &self.scene_pipeline.target_texture,
        );

        // Convert pixels to glyph instance data.
        let (sw, sh) = self.graphics.size();
        let instances = self.ascii_processor.pixels_to_instances(&pixels, sw, sh);
        let instance_count = instances.len() as u32;

        // Update instance buffer.
        self.composite_pipeline.update_instances(&self.graphics.queue, &instances);

        // Render glyphs to screen (GPU phase 2: composite).
        let frame = self.graphics.current_frame()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let gpu_start = Instant::now();
        self.composite_pipeline.render(
            &self.graphics.device,
            &self.graphics.queue,
            &view,
            instance_count,
        );
        self.metrics.record_gpu_phase(gpu_start);

        self.graphics.queue.present(frame);

        // Finalize metrics and log if ready.
        self.metrics.end_frame();

        Ok(())
    }

    pub fn fps(&self) -> f32 {
        self.metrics.fps()
    }
}