// Application state: ties together graphics, scene, camera, renderer, and input.
// The main loop updates this state each frame.

use std::time::Instant;
use crate::engine::core::Result;
use crate::ascii::{compute_tiles, ColorMode, Overlay, OverlayCell, SubdivisionPolicy};
use crate::engine::math::{degrees, radians, Mat4, Vec3};
use crate::graphics::{FrameOutcome, GraphicsContext};
use crate::renderer::{
    post_process, AsciiProcessor, CompositePipeline, DepthBuffer, FrameBuffer, GlyphStyle,
    LightUniform, ObjectUniform, PostProcessSettings, ScenePipeline,
};
use crate::scene::{
    Aabb, Camera, CameraMode, CameraRig, Entity, Frustum, Hierarchy, MaterialComponent,
    MaterialRegistry, MeshComponent, Projection, Scene, TransformComponent,
};
use crate::demo::material_spheres;

use super::input::InputState;
use super::metrics::FrameMetrics;

use winit::keyboard::KeyCode;

/// Translates raw input into `CameraRig` commands.
///
/// The rig (scene/camera_rig.rs) owns the actual camera state — orientation,
/// pivot, mode and dampening. This struct only decides which keys and mouse
/// motions map to which rig calls, so camera behaviour stays testable without
/// any winit dependency.
struct CameraInput {
    /// Movement speed (units per second).
    move_speed: f32,
    /// Mouse look sensitivity.
    look_sensitivity: f32,
    /// Latch for the preset-cycling key so one press cycles exactly once.
    cycle_key_was_down: bool,
    /// Wheel notches that first-person mode wants applied to the camera FOV,
    /// consumed by the caller after `update`.
    pending_fov_zoom: f32,
}

impl CameraInput {
    fn new() -> Self {
        Self {
            move_speed: 15.0,
            look_sensitivity: 0.0025,
            cycle_key_was_down: false,
            pending_fov_zoom: 0.0,
        }
    }

    /// Field-of-view change requested by the wheel this frame, in degrees.
    /// Positive means "zoom in" (narrower FOV).
    fn take_fov_zoom(&mut self) -> f32 {
        let z = self.pending_fov_zoom;
        self.pending_fov_zoom = 0.0;
        z * 3.0
    }

    /// Feed one frame of input into the rig, then advance its smoothing.
    fn update(&mut self, rig: &mut CameraRig, input: &mut InputState, dt: f32) {
        // Mouse look (only when LMB is held). The delta is consumed either way,
        // otherwise it would accumulate and snap the view on the next click.
        let (dx, dy) = input.take_mouse_delta();
        if input.is_look_active() {
            rig.rotate(
                -(dx as f32) * self.look_sensitivity,
                -(dy as f32) * self.look_sensitivity,
            );
        }

        // Wheel zooms. In third-person/orbit that means the rig distance; in
        // first person the rig has no distance to change (its zoom() is a
        // documented no-op), so the wheel is reported back to the caller and
        // applied to the camera's field of view instead — otherwise the wheel
        // would appear dead in the default preset.
        self.pending_fov_zoom = 0.0;
        let wheel = input.take_mouse_wheel();
        if wheel != 0.0 {
            if matches!(rig.mode(), CameraMode::FirstPerson) {
                self.pending_fov_zoom = wheel;
            } else {
                rig.zoom(-wheel);
            }
        }

        // C cycles the camera presets (edge-triggered).
        let cycle_down = input.is_key_pressed(KeyCode::KeyC);
        if cycle_down && !self.cycle_key_was_down {
            rig.set_mode(next_mode(rig.mode()));
        }
        self.cycle_key_was_down = cycle_down;

        // WASD moves the pivot along the rig's own basis vectors.
        let forward = rig.forward();
        let right = rig.right();
        let mut move_dir = Vec3::ZERO;
        if input.is_key_pressed(KeyCode::KeyW) {
            move_dir += forward;
        }
        if input.is_key_pressed(KeyCode::KeyS) {
            move_dir -= forward;
        }
        if input.is_key_pressed(KeyCode::KeyA) {
            move_dir -= right;
        }
        if input.is_key_pressed(KeyCode::KeyD) {
            move_dir += right;
        }
        if input.is_key_pressed(KeyCode::ShiftLeft) {
            move_dir += Vec3::UNIT_Y;
        }
        if input.is_key_pressed(KeyCode::ControlLeft) {
            move_dir -= Vec3::UNIT_Y;
        }
        if move_dir.length_squared() > 0.0 {
            rig.move_pivot(move_dir.normalize() * self.move_speed * dt);
        }

        rig.update(dt);
    }
}

/// Cycle order for the preset key: first-person -> third-person -> orbit.
fn next_mode(current: CameraMode) -> CameraMode {
    match current {
        CameraMode::FirstPerson => CameraMode::ThirdPerson {
            distance: 8.0,
            height_offset: 2.0,
        },
        CameraMode::ThirdPerson { .. } => CameraMode::Orbit { distance: 10.0 },
        CameraMode::Orbit { .. } => CameraMode::FirstPerson,
    }
}

/// Cycle order for the colour-mode key, highest fidelity first.
fn next_color_mode(current: ColorMode) -> ColorMode {
    match current {
        ColorMode::TrueColor => ColorMode::Ansi256,
        ColorMode::Ansi256 => ColorMode::Ansi16,
        ColorMode::Ansi16 => ColorMode::Grayscale,
        ColorMode::Grayscale => ColorMode::Monochrome,
        ColorMode::Monochrome => ColorMode::TrueColor,
    }
}

/// Build a first-person rig that reproduces an existing camera's viewpoint.
///
/// The rig stores orientation as yaw/pitch, so they have to be recovered from
/// the camera's look direction. Inverting the rig's own convention
/// (`forward = (cos yaw cos pitch, sin pitch, -sin yaw cos pitch)`) gives
/// `yaw = atan2(-z, x)` and `pitch = asin(y)`. The rig is snapped so the first
/// frame shows the scene from where the scene file asked, with no fly-in.
fn seed_rig_from_camera(camera: &Camera) -> CameraRig {
    let mut rig = CameraRig::new(CameraMode::FirstPerson);
    rig.set_pivot(camera.position);
    let dir = camera.forward();
    rig.set_yaw_pitch((-dir.z).atan2(dir.x), dir.y.clamp(-1.0, 1.0).asin());
    rig.snap();
    rig
}

/// Scene supersampling factor: the render target is this many times the glyph
/// grid on each axis, giving every cell a 2x2 block of subpixels.
const SUBSAMPLE: u32 = 2;

/// Scene file the demo starts with, resolved relative to the working directory
/// first (so an edited file is picked up by `cargo run`) and then relative to
/// the crate root (so it also works when launched from elsewhere).
const STARTUP_SCENE: &str = "assets/scenes/material_spheres.json";

/// Load the startup scene from disk, falling back to the code-built demo scene
/// if the file is missing or malformed. The fallback keeps the engine runnable
/// while a scene file is being edited; which path was taken is logged so a
/// silent fallback can't be mistaken for a successful load.
fn load_startup_scene() -> (Scene, Camera, Vec<LightUniform>, Hierarchy) {
    let candidates = [
        std::path::PathBuf::from(STARTUP_SCENE),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(STARTUP_SCENE),
    ];

    for path in &candidates {
        if !path.exists() {
            continue;
        }
        match crate::scene::load_scene_file(path) {
            Ok(loaded) => {
                eprintln!(
                    "scene: loaded \"{}\" from {} ({} lights)",
                    loaded.name,
                    path.display(),
                    loaded.lights.len()
                );
                return (loaded.scene, loaded.camera, loaded.lights, loaded.hierarchy);
            }
            Err(e) => eprintln!("scene: failed to load {}: {e}", path.display()),
        }
    }

    eprintln!("scene: falling back to the built-in material_spheres demo");
    let (scene, camera) = material_spheres::build_scene();
    (scene, camera, material_spheres::lights(), Hierarchy::new())
}

/// Main application state.
pub struct AppState {
    pub graphics: GraphicsContext,
    scene_pipeline: ScenePipeline,
    composite_pipeline: CompositePipeline,
    ascii_processor: AsciiProcessor,
    camera: Camera,
    camera_rig: CameraRig,
    camera_input: CameraInput,
    input: InputState,
    scene: Scene,
    /// Parent/child links between entities; world matrices are composed through
    /// this each frame.
    hierarchy: Hierarchy,
    /// Lights for the active scene (from the scene file, or the demo defaults).
    lights: Vec<LightUniform>,
    /// Deduplicating material store, rebuilt each frame.
    materials: MaterialRegistry,
    last_frame: Instant,
    metrics: FrameMetrics,
    /// Grid resolution (matches the ASCII cell grid).
    grid_cols: u32,
    grid_rows: u32,
    /// Culling stats for the current frame, logged alongside the FPS line.
    drawn_count: usize,
    culled_count: usize,
    /// Dynamic cell-grid policy. Toggled at runtime with G; when it merges,
    /// the depth buffer is read back to drive the subdivision.
    subdivision: SubdivisionPolicy,
    /// Latch for the subdivision toggle key.
    grid_key_was_down: bool,
    /// Colour fidelity the cells are quantized to. Cycled at runtime with M.
    color_mode: ColorMode,
    /// Latch for the colour-mode key.
    color_key_was_down: bool,
    /// Post-processing settings (bloom / SSAO / gamma / aberration). Toggled with P.
    post: PostProcessSettings,
    /// Latch for the post-processing key.
    post_key_was_down: bool,
    /// UI layer drawn over the scene: HUD text and a crosshair. Toggled with H.
    overlay: Overlay,
    hud_visible: bool,
    hud_key_was_down: bool,
    /// How cells become glyphs: quadrant blocks or the brightness ramp.
    /// Toggled with B.
    glyph_style: GlyphStyle,
    style_key_was_down: bool,
}

impl AppState {
    /// Initialize the application state from a winit window.
    pub fn new(
        window: &winit::window::Window,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let graphics = GraphicsContext::new(window, width, height)?;

        // Glyph grid resolution.
        let grid_cols = 120;
        let grid_rows = 68;

        // The scene renders at TWICE the glyph grid, so each cell owns a 2x2
        // block of subpixels. Quadrant block glyphs turn that into genuinely
        // double the visible resolution, and the ramp style gets free
        // anti-aliasing from averaging the same block.
        let sub_cols = grid_cols * SUBSAMPLE;
        let sub_rows = grid_rows * SUBSAMPLE;

        let scene_format = wgpu::TextureFormat::Rgba8Unorm;
        let scene_pipeline = ScenePipeline::new(
            &graphics.device,
            sub_cols,
            sub_rows,
            scene_format,
        )?;

        // The instance buffer holds the scene cells plus the UI overlay drawn on
        // top of them, so it is sized for both layers at full resolution.
        let composite_pipeline = CompositePipeline::new(
            &graphics.device,
            graphics.config.format,
            grid_cols * grid_rows * 2,
        )?;
        composite_pipeline.upload_atlas(&graphics.queue);

        let ascii_processor = AsciiProcessor::new(
            &graphics.device,
            sub_cols,
            sub_rows,
        );

        let (scene, camera, lights, hierarchy) = load_startup_scene();

        let camera_rig = seed_rig_from_camera(&camera);

        Ok(Self {
            graphics,
            scene_pipeline,
            composite_pipeline,
            ascii_processor,
            camera,
            camera_rig,
            camera_input: CameraInput::new(),
            input: InputState::new(),
            scene,
            hierarchy,
            lights,
            materials: MaterialRegistry::new(),
            last_frame: Instant::now(),
            metrics: FrameMetrics::new(),
            grid_cols,
            grid_rows,
            drawn_count: 0,
            culled_count: 0,
            // Off by default so the baseline image is the plain uniform grid;
            // press G to see the merged layout.
            subdivision: SubdivisionPolicy::uniform(),
            grid_key_was_down: false,
            // Full 24-bit colour by default; M cycles down through the
            // lower-fidelity terminal palettes.
            color_mode: ColorMode::TrueColor,
            color_key_was_down: false,
            // Off by default so the baseline image stays unfiltered; P enables
            // the demo preset.
            post: PostProcessSettings::none(),
            post_key_was_down: false,
            overlay: Overlay::with_glyph_map(grid_cols, grid_rows, crate::ascii::overlay_glyph_of),
            hud_visible: true,
            hud_key_was_down: false,
            glyph_style: GlyphStyle::default(),
            style_key_was_down: false,
        })
    }

    /// Handle a window resize. A minimized window reports 0x0, which must not
    /// reach the aspect-ratio maths or the surface configuration.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.graphics.resize(width, height);
        let aspect = width as f32 / height as f32;
        self.camera.set_aspect(aspect);
    }

    /// Get a mutable reference to the input state.
    pub fn input_mut(&mut self) -> &mut InputState {
        &mut self.input
    }

    /// Render one frame.
    ///
    /// Returns `Ok(())` for frames that were deliberately skipped (minimized or
    /// occluded window, stale swapchain) — those are ordinary window states, not
    /// errors, and must not tear the application down.
    pub fn render(&mut self) -> Result<()> {
        // Nothing to draw into while minimized; also keeps dt from accumulating
        // into one huge step that would jolt the camera on restore.
        if !self.graphics.is_renderable() {
            self.last_frame = Instant::now();
            return Ok(());
        }

        // Compute delta time.
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Begin metrics tracking.
        self.metrics.begin_frame();

        // Drive the camera rig from input, then write its smoothed state into
        // the camera (its projection is left alone).
        self.camera_input.update(&mut self.camera_rig, &mut self.input, dt);
        self.camera_rig.apply_to(&mut self.camera);

        // First-person wheel zoom acts on the field of view, clamped to a range
        // that stays usable (no fish-eye, no telescope).
        let fov_delta = self.camera_input.take_fov_zoom();
        if fov_delta != 0.0 {
            if let Projection::Perspective { fov_y, aspect, near, far } = self.camera.projection {
                let fov_degrees = degrees(fov_y) - fov_delta;
                self.camera.projection = Projection::perspective(
                    radians(fov_degrees.clamp(20.0, 100.0)),
                    aspect,
                    near,
                    far,
                );
            }
        }

        // G toggles the dynamic cell grid (edge-triggered).
        let grid_key_down = self.input.is_key_pressed(winit::keyboard::KeyCode::KeyG);
        if grid_key_down && !self.grid_key_was_down {
            self.subdivision = if self.subdivision.merges() {
                SubdivisionPolicy::uniform()
            } else {
                SubdivisionPolicy::default()
            };
            eprintln!(
                "grid: dynamic cell merging {}",
                if self.subdivision.merges() { "ON" } else { "OFF" }
            );
        }
        self.grid_key_was_down = grid_key_down;

        // M cycles the colour mode (edge-triggered).
        let color_key_down = self.input.is_key_pressed(winit::keyboard::KeyCode::KeyM);
        if color_key_down && !self.color_key_was_down {
            self.color_mode = next_color_mode(self.color_mode);
            eprintln!("color: mode = {:?}", self.color_mode);
        }
        self.color_key_was_down = color_key_down;

        // P toggles post-processing (edge-triggered).
        let post_key_down = self.input.is_key_pressed(winit::keyboard::KeyCode::KeyP);
        if post_key_down && !self.post_key_was_down {
            self.post = if self.post.any_enabled() {
                PostProcessSettings::none()
            } else {
                PostProcessSettings::demo()
            };
            eprintln!(
                "post: effects {}",
                if self.post.any_enabled() { "ON (demo preset)" } else { "OFF" }
            );
        }
        self.post_key_was_down = post_key_down;

        // H toggles the HUD (edge-triggered).
        let hud_key_down = self.input.is_key_pressed(winit::keyboard::KeyCode::KeyH);
        if hud_key_down && !self.hud_key_was_down {
            self.hud_visible = !self.hud_visible;
        }
        self.hud_key_was_down = hud_key_down;

        // B switches between block elements and the brightness ramp.
        let style_key_down = self.input.is_key_pressed(winit::keyboard::KeyCode::KeyB);
        if style_key_down && !self.style_key_was_down {
            self.glyph_style = match self.glyph_style {
                GlyphStyle::Blocks => GlyphStyle::Ramp,
                GlyphStyle::Ramp => GlyphStyle::Blocks,
            };
            eprintln!("glyphs: style = {:?}", self.glyph_style);
        }
        self.style_key_was_down = style_key_down;

        // Update scene uniforms.
        self.scene_pipeline.update_camera(&self.graphics.queue, &self.camera);
        self.scene_pipeline.update_lights(&self.graphics.queue, &self.lights);

        // Collect the draw list: one object per visible mesh entity, carrying its
        // model matrix (from TransformComponent, identity when absent) and
        // deduplicated material slot. Meshes whose world-space bounds fall
        // outside the camera frustum are skipped entirely.
        //
        // The scene's world-space bounds are accumulated over ALL meshes, not
        // just the visible ones, so the shadow frustum stays stable while the
        // camera turns — otherwise off-screen casters would stop shadowing.
        let frustum = Frustum::from_view_projection(&self.camera.view_projection());
        let mesh_entities = self.scene.entities_with::<MeshComponent>();

        // Compose world matrices through the parent/child graph in one memoized
        // pass, so a deep chain isn't re-walked per node.
        let scene_ref = &self.scene;
        let local_of = |e: Entity| {
            scene_ref
                .get_component::<TransformComponent>(e)
                .map(|t| t.world_matrix())
                .unwrap_or(Mat4::IDENTITY)
        };
        let world_matrices = self.hierarchy.world_matrices(&mesh_entities, &local_of);

        let mut meshes: Vec<(_, _, u32)> = Vec::new();
        let mut objects: Vec<ObjectUniform> = Vec::new();
        self.materials.clear();
        let mut scene_bounds: Option<Aabb> = None;
        let mut culled = 0usize;
        for (entity, model) in world_matrices {
            let Some(mesh) = self.scene.get_component::<MeshComponent>(entity) else {
                continue;
            };

            // World-space bounds: local AABB of the mesh, then transformed.
            let Some(local_bounds) = Aabb::from_points(mesh.vertices.iter().map(|v| v.position))
            else {
                continue; // empty mesh — nothing to draw or bound
            };
            let world_bounds = local_bounds.transformed(&model);
            scene_bounds = Some(match scene_bounds {
                Some(acc) => acc.merge(&world_bounds),
                None => world_bounds,
            });

            if !frustum.intersects_aabb(&world_bounds) {
                culled += 1;
                continue;
            }

            let material_index = match self.scene.get_component::<MaterialComponent>(entity) {
                Some(mat) => self.materials.register(mat),
                None => self.materials.register(&MaterialComponent::default()),
            };

            let object_index = objects.len() as u32;
            objects.push(ObjectUniform::new(model, material_index));
            meshes.push((entity, mesh, object_index));
        }
        self.drawn_count = meshes.len();
        self.culled_count = culled;

        // Upload per-object transforms and deduplicated materials to the GPU.
        self.scene_pipeline.upload_objects(&self.graphics.queue, &objects);
        let materials = self.materials.uniforms().to_vec();
        self.scene_pipeline.upload_materials(&self.graphics.queue, &materials);

        // Point the (simplified) shadow camera at light[0], sized to fit the scene.
        if let (Some(light0), Some(bounds)) = (self.lights.first(), scene_bounds) {
            let light_vp = light0.shadow_view_proj(bounds.center(), bounds.radius());
            self.scene_pipeline.update_shadow_camera(&self.graphics.queue, light_vp);
        }

        // Render all meshes in a single render pass (GPU phase 1: scene).
        let gpu_start = Instant::now();
        self.scene_pipeline.render_batched(
            &self.graphics.device,
            &self.graphics.queue,
            &meshes,
            &objects,
            &materials,
        )?;
        self.metrics.record_gpu_phase(gpu_start);

        // Read back the scene render target pixels (double-buffered, non-blocking —
        // may lag the current frame by ~1 frame, never stalls the CPU on the GPU).
        let mut pixels = self.ascii_processor.read_pixels(
            &self.graphics.device,
            &self.graphics.queue,
            &self.scene_pipeline.target_texture,
        );

        // Depth is needed by the dynamic grid and by SSAO; read it back once if
        // either of them wants it, and skip the transfer entirely otherwise.
        let needs_depth = self.subdivision.merges() || self.post.any_enabled();
        let depth = if needs_depth {
            Some(self.ascii_processor.read_depth(
                &self.graphics.device,
                &self.graphics.queue,
                self.scene_pipeline.depth_texture(),
            ))
        } else {
            None
        };

        // Post-processing runs on the subpixel buffer, before glyphs are chosen,
        // so bloom and AO influence which character each cell gets — not just its
        // colour. That is the whole point of doing it "in ASCII space".
        let sub_cols = self.grid_cols * SUBSAMPLE;
        let sub_rows = self.grid_rows * SUBSAMPLE;
        if self.post.any_enabled() {
            if let Ok(mut fb) = FrameBuffer::from_rgba8(&pixels, sub_cols, sub_rows) {
                let depth_buf = depth
                    .as_ref()
                    .and_then(|d| DepthBuffer::from_slice(d, sub_cols, sub_rows).ok());
                post_process::apply_all(&mut fb, depth_buf.as_ref(), &self.post);
                pixels = fb.to_rgba8();
            }
        }

        // Convert subpixels to glyph instances. The dynamic grid merges whole
        // cells, so it works on the cell-resolution depth obtained by taking one
        // subpixel per cell.
        let mut instances = match (self.subdivision.merges(), depth.as_ref()) {
            (true, Some(depth)) => {
                // Merging decisions are per CELL, so the subpixel depth is reduced
                // to one value per cell first.
                let cell_depth = downsample_depth(depth, sub_cols, sub_rows);
                let tiles =
                    compute_tiles(&cell_depth, self.grid_cols, self.grid_rows, &self.subdivision);
                self.ascii_processor
                    .subpixels_to_instances_tiled(&pixels, &tiles, self.glyph_style)
            }
            _ => self
                .ascii_processor
                .subpixels_to_instances(&pixels, self.glyph_style),
        };

        // Quantize to the selected colour fidelity. Done on the instances rather
        // than the pixel buffer so the glyph choice still comes from the full
        // resolution luminance — dropping the palette should change colour, not
        // which character is drawn.
        if self.color_mode != ColorMode::TrueColor {
            for inst in &mut instances {
                let [r, g, b] = self
                    .color_mode
                    .quantize([inst.color_r, inst.color_g, inst.color_b]);
                inst.color_r = r;
                inst.color_g = g;
                inst.color_b = b;
            }
        }
        // Draw the UI layer and append it as extra glyph quads. Appending rather
        // than rewriting scene cells keeps the HUD independent of whether the
        // grid merged any tiles this frame.
        if self.hud_visible {
            self.draw_hud();
            let ui = self.ascii_processor.overlay_to_instances(&self.overlay);
            instances.extend(ui);
        }

        let instance_count = instances.len() as u32;

        // Update instance buffer.
        self.composite_pipeline.update_instances(&self.graphics.queue, &instances);

        // Render glyphs to screen (GPU phase 2: composite).
        let frame = match self.graphics.current_frame() {
            FrameOutcome::Frame(frame) => frame,
            // Transient surface state (resize, minimize, occlusion, stale
            // swapchain): the surface was reconfigured, so drop this frame and
            // try again next tick rather than failing the whole run.
            FrameOutcome::Skip(_reason) => return Ok(()),
        };
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
        self.metrics.set_scene_stats(
            self.drawn_count,
            self.culled_count,
            self.materials.len(),
        );
        self.metrics.set_glyph_count(instance_count as usize);
        self.metrics.end_frame();

        Ok(())
    }

    pub fn fps(&self) -> f32 {
        self.metrics.fps()
    }

    /// Redraw the HUD into the overlay: stats in the top-left, the active
    /// toggles below them, and a centre crosshair.
    fn draw_hud(&mut self) {
        const TEXT: [f32; 3] = [0.85, 0.95, 0.85];
        const DIM: [f32; 3] = [0.45, 0.55, 0.45];

        self.overlay.clear();

        let fps = self.metrics.fps();
        self.overlay
            .draw_text(1, 1, &format!("FPS {fps:.0}"), TEXT);
        self.overlay.draw_text(
            1,
            2,
            &format!("DRAWN {} CULL {}", self.drawn_count, self.culled_count),
            DIM,
        );
        self.overlay.draw_text(
            1,
            3,
            &format!("MAT {} GRID {}", self.materials.len(), if self.subdivision.merges() { "DYN" } else { "UNI" }),
            DIM,
        );
        self.overlay.draw_text(
            1,
            4,
            &format!(
                "COLOR {} POST {}",
                color_mode_label(self.color_mode),
                if self.post.any_enabled() { "ON" } else { "OFF" }
            ),
            DIM,
        );
        self.overlay.draw_text(
            1,
            5,
            &format!("STYLE {}", if self.glyph_style == GlyphStyle::Blocks { "BLOCK" } else { "RAMP" }),
            DIM,
        );
        self.overlay
            .draw_text(1, 6, "KEYS C M P G H B", DIM);

        // Crosshair at the centre of the grid.
        let (cx, cy) = (self.grid_cols / 2, self.grid_rows / 2);
        self.overlay
            .set_cell(cx, cy, OverlayCell::new(crate::ascii::overlay_glyph_of('+'), TEXT));
    }
}

/// Reduce a subpixel depth buffer to one value per glyph cell.
///
/// Takes the MINIMUM of each cell's 2x2 block, i.e. the nearest surface in that
/// cell. Merging decisions must be conservative: if any part of a cell holds
/// near geometry, the cell keeps full detail.
fn downsample_depth(depth: &[f32], sub_cols: u32, sub_rows: u32) -> Vec<f32> {
    let cols = sub_cols / 2;
    let rows = sub_rows / 2;
    let mut out = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let mut nearest = f32::INFINITY;
            for dy in 0..2 {
                for dx in 0..2 {
                    let x = col * 2 + dx;
                    let y = row * 2 + dy;
                    if x >= sub_cols || y >= sub_rows {
                        continue;
                    }
                    let idx = (y as usize) * (sub_cols as usize) + (x as usize);
                    if let Some(&d) = depth.get(idx) {
                        if d.is_finite() && d < nearest {
                            nearest = d;
                        }
                    }
                }
            }
            // Nothing usable in this cell — treat it as empty space (far plane).
            out.push(if nearest.is_finite() { nearest } else { 1.0 });
        }
    }
    out
}

/// Short label for the HUD line.
fn color_mode_label(mode: ColorMode) -> &'static str {
    match mode {
        ColorMode::TrueColor => "TRUE",
        ColorMode::Ansi256 => "256",
        ColorMode::Ansi16 => "16",
        ColorMode::Grayscale => "GRAY",
        ColorMode::Monochrome => "MONO",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::radians;
    use crate::scene::Projection;

    fn camera_looking(from: Vec3, at: Vec3) -> Camera {
        Camera::new(
            from,
            at,
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 16.0 / 9.0, 0.1, 200.0),
        )
    }

    /// The rig must reproduce the camera it was seeded from, otherwise a scene
    /// file's authored viewpoint silently changes on startup.
    #[test]
    fn seeded_rig_reproduces_the_original_view_direction() {
        for (from, at) in [
            (Vec3::new(0.0, 3.0, 8.0), Vec3::new(0.0, -0.5, 0.0)),
            (Vec3::new(-5.0, 1.0, -5.0), Vec3::ZERO),
            (Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO),
            (Vec3::new(7.0, 2.0, 0.0), Vec3::new(0.0, 2.0, 0.0)),
        ] {
            let original = camera_looking(from, at);
            let rig = seed_rig_from_camera(&original);

            assert!(
                (rig.position() - from).length() < 1e-4,
                "first-person rig must sit at the camera position (from {from})"
            );

            let mut rebuilt = camera_looking(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
            rig.apply_to(&mut rebuilt);
            let want = original.forward();
            let got = rebuilt.forward();
            assert!(
                (want - got).length() < 1e-4,
                "look direction drifted: wanted {want}, got {got}"
            );
        }
    }

    #[test]
    fn seeding_preserves_the_camera_projection() {
        let original = camera_looking(Vec3::new(0.0, 3.0, 8.0), Vec3::ZERO);
        let rig = seed_rig_from_camera(&original);
        let mut target = Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::UNIT_Y,
            Projection::orthographic_sized(10.0, 1.0, 0.1, 50.0),
        );
        rig.apply_to(&mut target);
        assert!(
            target.projection.is_orthographic(),
            "apply_to must not overwrite the projection"
        );
    }

    #[test]
    fn straight_down_look_does_not_produce_nan() {
        // asin() at the poles is the classic NaN trap for yaw/pitch recovery.
        let original = camera_looking(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO);
        let rig = seed_rig_from_camera(&original);
        assert!(rig.yaw().is_finite() && rig.pitch().is_finite());
        assert!(rig.position().x.is_finite() && rig.position().y.is_finite());
    }

    #[test]
    fn camera_mode_cycle_visits_every_preset_and_returns() {
        let start = CameraMode::FirstPerson;
        let third = next_mode(start);
        let orbit = next_mode(third);
        let back = next_mode(orbit);
        assert!(matches!(third, CameraMode::ThirdPerson { .. }));
        assert!(matches!(orbit, CameraMode::Orbit { .. }));
        assert!(matches!(back, CameraMode::FirstPerson));
    }
}