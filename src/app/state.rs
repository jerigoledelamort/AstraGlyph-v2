// Application state: ties together graphics, scene, camera, renderer, and input.
// The main loop updates this state each frame.

use std::time::Instant;
use crate::engine::core::Result;
use crate::engine::math::{radians, Vec3};
use crate::graphics::GraphicsContext;
use crate::renderer::{AsciiProcessor, CompositePipeline, LightUniform, ScenePipeline};
use crate::scene::{Camera, MeshComponent, Scene};
use crate::demo::cornell_box;

use super::input::InputState;

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
            move_speed: 300.0,
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
            self.yaw += dx as f32 * self.look_sensitivity;
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
        if input.is_key_pressed(winit::keyboard::KeyCode::Space) {
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
    frame_count: u64,
    fps_accum: f32,
    fps: f32,
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

        let (scene, camera) = cornell_box::build_scene();

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
            frame_count: 0,
            fps_accum: 0.0,
            fps: 0.0,
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

        // FPS counter.
        self.frame_count += 1;
        self.fps_accum += dt;
        if self.fps_accum >= 0.5 {
            self.fps = self.frame_count as f32 / self.fps_accum;
            self.frame_count = 0;
            self.fps_accum = 0.0;
        }

        // Update camera from input.
        self.camera_controller.update(&mut self.camera, &mut self.input, dt);

        // Update scene uniforms.
        self.scene_pipeline.update_view_proj(&self.graphics.queue, &self.camera);
        let light = LightUniform {
            direction: [cornell_box::default_light().x, cornell_box::default_light().y, cornell_box::default_light().z],
            ambient: 0.15,
            diffuse: 0.85,
        };
        self.scene_pipeline.update_light(&self.graphics.queue, &light);

        // Render all meshes in the scene to the offscreen target.
        let mesh_entities = self.scene.entities_with::<MeshComponent>();
        for entity in mesh_entities {
            if let Some(mesh) = self.scene.get_component::<MeshComponent>(entity) {
                self.scene_pipeline.render(
                    &self.graphics.device,
                    &self.graphics.queue,
                    mesh,
                )?;
            }
        }

        // Read back the scene render target pixels.
        let pixels = self.ascii_processor.read_pixels(
            &self.graphics.device,
            &self.graphics.queue,
            &self.scene_pipeline.target_texture,
        )?;

        // Convert pixels to glyph instance data.
        let (sw, sh) = self.graphics.size();
        let instances = self.ascii_processor.pixels_to_instances(&pixels, sw, sh);
        let instance_count = instances.len() as u32;

        // Update instance buffer.
        self.composite_pipeline.update_instances(&self.graphics.queue, &instances);

        // Render glyphs to screen.
        let frame = self.graphics.current_frame()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.composite_pipeline.render(
            &self.graphics.device,
            &self.graphics.queue,
            &view,
            instance_count,
        );

        self.graphics.queue.present(frame);

        Ok(())
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }
}