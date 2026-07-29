// Application state: ties together graphics, scene, camera, renderer, and input.
// The main loop updates this state each frame.

use std::time::Instant;
use crate::engine::core::Result;
use crate::ascii::{compute_tiles, ColorMode, Overlay, OverlayCell, SubdivisionPolicy};
use crate::engine::math::{degrees, radians, Mat4, Vec3};
use crate::graphics::{FrameOutcome, GraphicsContext};
use crate::renderer::{
    post_process, trace_flags, AsciiProcessor, CompositePipeline, DepthBuffer, FrameBuffer,
    GlyphStyle, InstanceRequest, LightUniform, ObjectUniform, PostProcessSettings, RayTracer,
    ScenePipeline, TraceSettings,
};
use crate::scene::{
    Aabb, Camera, CameraMode, CameraRig, Entity, Frustum, Hierarchy, MaterialComponent,
    MaterialRegistry, MeshComponent, Projection, Scene, TransformComponent,
};
use crate::demo::material_spheres;
use crate::ui::{Console, ConsoleAction, Menu, MenuAction, MenuEvent, MenuItem};

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

/// The settings menu. Item ids are the contract between the menu and
/// `apply_menu_event`, so they are defined here in one place.
fn build_menu() -> Menu {
    Menu::new(
        "ASTRAGLYPH",
        vec![
            MenuItem::button("Resume", "resume"),
            MenuItem::Separator,
            MenuItem::choice(
                "Glyphs",
                "style",
                vec!["Blocks".into(), "Ramp".into()],
                0,
            ),
            MenuItem::choice(
                "Colour",
                "color",
                vec!["True".into(), "256".into(), "16".into(), "Grey".into(), "Mono".into()],
                0,
            ),
            MenuItem::toggle("Ray tracing", "trace", false),
            MenuItem::toggle("Post FX", "post", false),
            MenuItem::toggle("Dynamic grid", "grid", false),
            MenuItem::toggle("HUD", "hud", true),
            MenuItem::Separator,
            MenuItem::submenu(
                "Camera",
                vec![
                    MenuItem::button("First person", "cam_first"),
                    MenuItem::button("Third person", "cam_third"),
                    MenuItem::button("Orbit", "cam_orbit"),
                ],
            ),
            MenuItem::Separator,
            MenuItem::button("Quit", "quit"),
        ],
    )
}

/// The debug console, pre-seeded with a hint so it is not an empty black box on
/// first open.
fn build_console() -> Console {
    let mut console = Console::new();
    console.print("AstraGlyph console. Type 'help' for commands.");
    console
}

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
    /// Acceleration structures for the traced path. `None` on hardware without
    /// ray query, which is what makes the whole traced path optional rather than
    /// a hard requirement.
    ray_tracer: Option<RayTracer>,
    /// Ray budget and feature flags for the traced path.
    trace_settings: TraceSettings,
    /// Whether the traced path is active this frame. Toggled with R for the
    /// raster/traced A/B comparison; forced off when there is no ray tracer.
    traced_enabled: bool,
    trace_key_was_down: bool,
    /// Frames rendered since startup.
    frame_counter: u32,
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
    /// Settings menu (Tab) and debug console (Backquote). While either is open it
    /// takes keyboard focus, so the camera does not move underneath it.
    menu: Menu,
    console: Console,
    /// Edge latches for the keys the UI layers consume.
    ui_keys_down: UiKeyLatches,
    /// Set by the menu's Quit entry or the console's `quit` command; the event
    /// loop polls it so shutdown stays the caller's decision.
    should_exit: bool,
    /// Last camera position reported by the input trace, so it only logs on
    /// actual movement instead of once per frame.
    last_traced_position: Vec3,
    /// When the trace last printed, used to throttle it.
    last_trace_log: Instant,
}

/// Previous state of every key the UI reads, so each press acts once.
///
/// A menu that scrolls once per FRAME instead of once per press is unusable, and
/// this is the whole reason the UI modules take discrete actions rather than
/// polling key state themselves.
#[derive(Default)]
struct UiKeyLatches {
    menu_toggle: bool,
    console_toggle: bool,
    up: bool,
    down: bool,
    enter: bool,
    backspace: bool,
    delete: bool,
    left: bool,
    right: bool,
    home: bool,
    end: bool,
    page_up: bool,
    page_down: bool,
}

/// Edge detector: true only on the frame a key transitions to pressed.
fn pressed_once(down: bool, latch: &mut bool) -> bool {
    let fired = down && !*latch;
    *latch = down;
    fired
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

        // Acceleration structures come first: the traced pipeline needs their
        // bind group layout, and on hardware without ray query there is no
        // layout, no traced pipeline, and no traced shader compiled at all.
        let ray_tracer = if graphics.ray_tracing().is_enabled() {
            Some(RayTracer::new(&graphics.device))
        } else {
            None
        };

        let scene_format = wgpu::TextureFormat::Rgba8Unorm;
        let scene_pipeline = ScenePipeline::new(
            &graphics.device,
            sub_cols,
            sub_rows,
            scene_format,
            ray_tracer.as_ref().map(|rt| rt.layout()),
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

        // Tracing is the default whenever the hardware allows it: it is the point
        // of the phase, and R switches back to rasterising for comparison.
        let traced_enabled = ray_tracer.is_some() && scene_pipeline.supports_tracing();

        Ok(Self {
            graphics,
            scene_pipeline,
            ray_tracer,
            trace_settings: TraceSettings::default(),
            traced_enabled,
            trace_key_was_down: false,
            frame_counter: 0,
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
            menu: build_menu(),
            console: build_console(),
            ui_keys_down: UiKeyLatches::default(),
            should_exit: false,
            last_traced_position: Vec3::ZERO,
            last_trace_log: Instant::now(),
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

    /// Whether a UI layer currently wants typed characters. `main` uses this to
    /// decide if a key press should also be forwarded as text.
    pub fn ui_wants_text(&self) -> bool {
        self.console.is_open()
    }

    /// Whether any UI layer holds keyboard focus, in which case the camera must
    /// not react to movement keys.
    fn ui_has_focus(&self) -> bool {
        self.console.is_open() || self.menu.is_open()
    }

    /// Handle Escape. Returns true if a UI layer consumed it, in which case the
    /// application must NOT exit.
    pub fn handle_escape(&mut self) -> bool {
        if self.console.is_open() {
            self.console.close();
            self.input.clear_typed();
            return true;
        }
        if self.menu.is_open() {
            // Let the menu decide: inside a submenu this steps back out rather
            // than closing, which is what a user expects from Escape.
            self.menu.handle(MenuAction::Back);
            return true;
        }
        false
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

        // UI first: an open menu or console owns the keyboard, so the camera must
        // not also act on the same keys.
        self.update_ui();

        // Drive the camera rig from input, then write its smoothed state into
        // the camera (its projection is left alone). While a UI layer has focus
        // the rig still advances (so smoothing settles) but receives no input.
        if self.ui_has_focus() {
            self.input.take_mouse_delta();
            self.input.take_mouse_wheel();
            self.camera_rig.update(dt);
        } else {
            self.camera_input.update(&mut self.camera_rig, &mut self.input, dt);
        }
        self.camera_rig.apply_to(&mut self.camera);

        // The consumer half of the input trace: shows whether the camera actually
        // saw the events, which is what separates "the OS delivered nothing" from
        // "something swallowed it".
        if self.input.is_tracing() {
            // Throttled: at 1500 FPS an unthrottled line per moved frame buries
            // the very events it is meant to explain.
            let moved = (self.camera.position - self.last_traced_position).length();
            if moved > 1e-4 && self.last_trace_log.elapsed().as_millis() >= 200 {
                eprintln!(
                    "camera: pos={} yaw={:.3} focus={}",
                    self.camera.position,
                    self.camera_rig.yaw(),
                    self.ui_has_focus()
                );
                self.last_traced_position = self.camera.position;
                self.last_trace_log = Instant::now();
            }
        }

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

        // Single-letter shortcuts are suppressed while a UI layer has focus —
        // otherwise typing "post" in the console would flip half the settings.
        let shortcuts_active = !self.ui_has_focus();

        // G toggles the dynamic cell grid (edge-triggered).
        let grid_key_down =
            shortcuts_active && self.input.is_key_pressed(winit::keyboard::KeyCode::KeyG);
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
        let color_key_down =
            shortcuts_active && self.input.is_key_pressed(winit::keyboard::KeyCode::KeyM);
        if color_key_down && !self.color_key_was_down {
            self.color_mode = next_color_mode(self.color_mode);
            eprintln!("color: mode = {:?}", self.color_mode);
        }
        self.color_key_was_down = color_key_down;

        // P toggles post-processing (edge-triggered).
        let post_key_down =
            shortcuts_active && self.input.is_key_pressed(winit::keyboard::KeyCode::KeyP);
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
        let hud_key_down =
            shortcuts_active && self.input.is_key_pressed(winit::keyboard::KeyCode::KeyH);
        if hud_key_down && !self.hud_key_was_down {
            self.hud_visible = !self.hud_visible;
        }
        self.hud_key_was_down = hud_key_down;

        // B switches between block elements and the brightness ramp.
        let style_key_down =
            shortcuts_active && self.input.is_key_pressed(winit::keyboard::KeyCode::KeyB);
        if style_key_down && !self.style_key_was_down {
            self.glyph_style = match self.glyph_style {
                GlyphStyle::Blocks => GlyphStyle::Ramp,
                GlyphStyle::Ramp => GlyphStyle::Blocks,
            };
            eprintln!("glyphs: style = {:?}", self.glyph_style);
        }
        self.style_key_was_down = style_key_down;

        // R switches between rasterised and traced lighting. This is the A/B
        // comparison the phase requires: without it there is no way to tell a
        // traced regression from a traced feature.
        let trace_key_down =
            shortcuts_active && self.input.is_key_pressed(winit::keyboard::KeyCode::KeyR);
        if trace_key_down && !self.trace_key_was_down {
            self.toggle_tracing();
        }
        self.trace_key_was_down = trace_key_down;

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
        let mut all_instances: Vec<(InstanceRequest, &MeshComponent)> = Vec::new();
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

            // Materials are registered before the cull test, not after: a culled
            // mesh still needs a valid material slot because a ray can reach it.
            // As a side effect the material count reports the whole scene rather
            // than only what is on screen, which is the more useful number.
            let material_index = match self.scene.get_component::<MaterialComponent>(entity) {
                Some(mat) => self.materials.register(mat),
                None => self.materials.register(&MaterialComponent::default()),
            };

            all_instances.push((
                InstanceRequest {
                    entity_id: entity.id(),
                    model,
                    material_index,
                },
                mesh,
            ));

            if !frustum.intersects_aabb(&world_bounds) {
                culled += 1;
                continue;
            }

            let object_index = objects.len() as u32;
            objects.push(ObjectUniform::new(model, material_index));
            meshes.push((entity, mesh, object_index));
        }
        self.drawn_count = meshes.len();
        self.culled_count = culled;

        // Acceleration structures track the *unculled* instance list on purpose.
        //
        // Frustum culling is a rasterisation optimisation: a fragment shader can
        // never need a mesh it did not draw. A ray can. An object behind the
        // camera still shows up in a mirror and still blocks a shadow ray, so
        // culling it out of the TLAS would make reflections pop in and out as the
        // camera turns — the same class of bug the shadow frustum already avoids
        // by accumulating bounds over all meshes.
        let traced_active = self.traced_enabled && self.ray_tracer.is_some();
        if traced_active {
            if let Some(rt) = self.ray_tracer.as_mut() {
                rt.update(&self.graphics.device, &self.graphics.queue, &all_instances);
                rt.upload_settings(&self.graphics.queue, &self.trace_settings);
            }
        }

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
        let traced_group = if traced_active {
            self.ray_tracer.as_ref().map(|rt| rt.bind_group())
        } else {
            None
        };
        let gpu_start = Instant::now();
        self.scene_pipeline.render_batched(
            &self.graphics.device,
            &self.graphics.queue,
            &meshes,
            &objects,
            &materials,
            traced_group,
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
        // Screen-space AO is dropped while tracing: the traced path already
        // attenuated the ambient term with rays that can see what the depth
        // buffer cannot, and applying the approximation on top of it would
        // darken every crease twice.
        let post = if traced_active {
            self.post.without_ssao()
        } else {
            self.post
        };
        // Depth is only consumed by the dynamic grid and by screen-space AO, so
        // dropping SSAO also drops the transfer that fed it.
        let needs_depth = self.subdivision.merges() || post.ssao_active();
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
        if post.any_enabled() {
            if let Ok(mut fb) = FrameBuffer::from_rgba8(&pixels, sub_cols, sub_rows) {
                let depth_buf = depth
                    .as_ref()
                    .and_then(|d| DepthBuffer::from_slice(d, sub_cols, sub_rows).ok());
                post_process::apply_all(&mut fb, depth_buf.as_ref(), &post);
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
        // The overlay is redrawn from scratch each frame: HUD first, then the
        // menu or console on top of it so a panel covers the stats rather than
        // fighting them for the same cells.
        let ui_open = self.menu.is_open() || self.console.is_open();
        if self.hud_visible || ui_open {
            self.overlay.clear();
            if self.hud_visible {
                self.draw_hud();
            }
            self.menu.draw(&mut self.overlay);
            self.console.draw(&mut self.overlay);
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
        self.frame_counter = self.frame_counter.wrapping_add(1);

        Ok(())
    }

    pub fn fps(&self) -> f32 {
        self.metrics.fps()
    }

    /// Whether the traced path can be switched on at all on this machine.
    fn tracing_available(&self) -> bool {
        self.ray_tracer.is_some() && self.scene_pipeline.supports_tracing()
    }

    /// Flip between rasterised and traced lighting, reporting the result.
    ///
    /// On hardware without ray query this says so rather than silently doing
    /// nothing — a dead key with no explanation is how the last round of input
    /// bugs got misdiagnosed.
    fn toggle_tracing(&mut self) {
        if !self.tracing_available() {
            eprintln!(
                "raytracing: unavailable ({})",
                self.graphics.ray_tracing().describe()
            );
            return;
        }
        self.traced_enabled = !self.traced_enabled;
        eprintln!(
            "raytracing: lighting = {}",
            if self.traced_enabled { "TRACED" } else { "RASTER" }
        );
    }

    /// Short description of the active lighting path plus its ray budget, for
    /// the HUD and the console.
    fn lighting_summary(&self) -> String {
        if !self.tracing_available() {
            return format!("RASTER ({})", self.graphics.ray_tracing().tag());
        }
        if !self.traced_enabled {
            return "RASTER (RTX IDLE)".to_string();
        }
        let lights = self.lights.len().min(crate::renderer::scene_pass::MAX_LIGHTS) as u32;
        let rays = self.trace_settings.rays_per_fragment(lights);
        let tris = self
            .ray_tracer
            .as_ref()
            .map(|rt| rt.triangle_count())
            .unwrap_or(0);
        format!(
            "TRACED d{} {} rays/px {} tris",
            self.trace_settings.max_depth, rays, tris
        )
    }

    /// Route keyboard input to the menu and console, and apply what they report.
    fn update_ui(&mut self) {
        // Snapshot every key the UI cares about in one pass, so the rest of the
        // function can borrow `self` mutably without a live borrow of `self.input`.
        let raw = [
            KeyCode::Tab,
            KeyCode::Backquote,
            KeyCode::ArrowUp,
            KeyCode::ArrowDown,
            KeyCode::Enter,
            KeyCode::Backspace,
            KeyCode::Delete,
            KeyCode::ArrowLeft,
            KeyCode::ArrowRight,
            KeyCode::Home,
            KeyCode::End,
            KeyCode::PageUp,
            KeyCode::PageDown,
        ]
        .map(|code| self.input.is_key_pressed(code));

        // Tab opens/closes the menu; Backquote the console. They are mutually
        // exclusive so focus is never ambiguous.
        let toggle_menu = pressed_once(raw[0], &mut self.ui_keys_down.menu_toggle);
        let toggle_console = pressed_once(raw[1], &mut self.ui_keys_down.console_toggle);

        if toggle_menu {
            if self.menu.is_open() {
                self.menu.close();
            } else {
                self.console.close();
                self.menu.open();
                self.sync_menu_from_state();
            }
        }
        if toggle_console {
            if self.console.is_open() {
                self.console.close();
            } else {
                self.menu.close();
                self.console.open();
            }
            // Drop the backquote itself so it does not land in the input line.
            self.input.clear_typed();
        }

        // Latch every UI key each frame, whether or not it is consumed, so a key
        // held down while a layer opens does not fire immediately on open.
        let up = pressed_once(raw[2], &mut self.ui_keys_down.up);
        let down = pressed_once(raw[3], &mut self.ui_keys_down.down);
        let enter = pressed_once(raw[4], &mut self.ui_keys_down.enter);
        let backspace = pressed_once(raw[5], &mut self.ui_keys_down.backspace);
        let delete = pressed_once(raw[6], &mut self.ui_keys_down.delete);
        let left = pressed_once(raw[7], &mut self.ui_keys_down.left);
        let right = pressed_once(raw[8], &mut self.ui_keys_down.right);
        let home = pressed_once(raw[9], &mut self.ui_keys_down.home);
        let end = pressed_once(raw[10], &mut self.ui_keys_down.end);
        let page_up = pressed_once(raw[11], &mut self.ui_keys_down.page_up);
        let page_down = pressed_once(raw[12], &mut self.ui_keys_down.page_down);

        if self.console.is_open() {
            let typed = self.input.take_typed();
            for c in typed.chars() {
                self.console.insert_char(c);
            }
            // History uses the vertical arrows, so the console maps them itself.
            let actions = [
                (enter, ConsoleAction::Submit),
                (backspace, ConsoleAction::Backspace),
                (delete, ConsoleAction::Delete),
                (left, ConsoleAction::Left),
                (right, ConsoleAction::Right),
                (home, ConsoleAction::Home),
                (end, ConsoleAction::End),
                (up, ConsoleAction::HistoryPrev),
                (down, ConsoleAction::HistoryNext),
            ];
            for (fired, action) in actions {
                if !fired {
                    continue;
                }
                if let Some(command) = self.console.handle(action) {
                    self.run_command(&command);
                }
            }
            if page_up {
                self.console.scroll_up(4);
            }
            if page_down {
                self.console.scroll_down(4);
            }
            return;
        }

        if self.menu.is_open() {
            // Typed characters are irrelevant to the menu; drop them so they do
            // not appear later in the console.
            self.input.clear_typed();
            let actions = [
                (up, MenuAction::Up),
                (down, MenuAction::Down),
                (enter, MenuAction::Activate),
            ];
            for (fired, action) in actions {
                if !fired {
                    continue;
                }
                if let Some(event) = self.menu.handle(action) {
                    self.apply_menu_event(event);
                }
            }
        }
    }

    /// Push the current engine settings into the menu, so an opened menu shows
    /// what is actually active rather than whatever it was left at.
    fn sync_menu_from_state(&mut self) {
        let style = if self.glyph_style == GlyphStyle::Blocks { 0 } else { 1 };
        let color = match self.color_mode {
            ColorMode::TrueColor => 0,
            ColorMode::Ansi256 => 1,
            ColorMode::Ansi16 => 2,
            ColorMode::Grayscale => 3,
            ColorMode::Monochrome => 4,
        };
        self.menu.set_choice("style", style);
        self.menu.set_choice("color", color);
        self.menu.set_toggle("trace", self.traced_enabled);
        self.menu.set_toggle("post", self.post.any_enabled());
        self.menu.set_toggle("grid", self.subdivision.merges());
        self.menu.set_toggle("hud", self.hud_visible);
    }

    /// Apply what the menu reported.
    fn apply_menu_event(&mut self, event: MenuEvent) {
        match event {
            MenuEvent::Activated(id) => match id.as_str() {
                "resume" => self.menu.close(),
                "quit" => self.should_exit = true,
                "cam_first" => self.camera_rig.set_mode(CameraMode::FirstPerson),
                "cam_third" => self
                    .camera_rig
                    .set_mode(CameraMode::ThirdPerson { distance: 8.0, height_offset: 2.0 }),
                "cam_orbit" => self.camera_rig.set_mode(CameraMode::Orbit { distance: 10.0 }),
                _ => {}
            },
            MenuEvent::Toggled(id, on) => match id.as_str() {
                "trace" => {
                    if on != self.traced_enabled {
                        self.toggle_tracing();
                    }
                    // The menu may have moved ahead of reality (no ray query on
                    // this machine), so push the truth back into it.
                    self.menu.set_toggle("trace", self.traced_enabled);
                }
                "post" => {
                    self.post = if on {
                        PostProcessSettings::demo()
                    } else {
                        PostProcessSettings::none()
                    }
                }
                "grid" => {
                    self.subdivision = if on {
                        SubdivisionPolicy::default()
                    } else {
                        SubdivisionPolicy::uniform()
                    }
                }
                "hud" => self.hud_visible = on,
                _ => {}
            },
            MenuEvent::Chose(id, index) => match id.as_str() {
                "style" => {
                    self.glyph_style = if index == 0 { GlyphStyle::Blocks } else { GlyphStyle::Ramp }
                }
                "color" => {
                    self.color_mode = match index {
                        1 => ColorMode::Ansi256,
                        2 => ColorMode::Ansi16,
                        3 => ColorMode::Grayscale,
                        4 => ColorMode::Monochrome,
                        _ => ColorMode::TrueColor,
                    }
                }
                _ => {}
            },
            MenuEvent::Closed => {}
        }
    }

    /// Whether a console command or menu entry asked the application to quit.
    pub fn should_exit(&self) -> bool {
        self.should_exit
    }

    /// Execute a console command line.
    ///
    /// Command handling lives here rather than in the console module because the
    /// commands act on engine state; the console only owns the text.
    fn run_command(&mut self, line: &str) {
        let mut parts = line.split_whitespace();
        let Some(command) = parts.next() else {
            return;
        };
        let arg = parts.next();

        /// Parse an on/off argument, defaulting to toggling.
        fn on_off(arg: Option<&str>, current: bool) -> bool {
            match arg {
                Some("on") | Some("1") | Some("true") => true,
                Some("off") | Some("0") | Some("false") => false,
                _ => !current,
            }
        }

        match command {
            "help" => {
                self.console.print("commands:");
                self.console.print("  help              this list");
                self.console.print("  fps               current frame rate");
                self.console.print("  scene             entity and material counts");
                self.console.print("  post [on|off]     post-processing");
                self.console.print("  grid [on|off]     dynamic cell grid");
                self.console.print("  hud [on|off]      HUD overlay");
                self.console.print("  style [block|ramp] glyph style");
                self.console.print("  color <true|256|16|grey|mono>");
                self.console.print("  cam <first|third|orbit>");
                self.console.print("  trace [on|off]    traced vs rasterised lighting");
                self.console.print("  rays              ray budget and AS stats");
                self.console.print("  depth <0-4>       reflection/refraction bounces");
                self.console
                    .print("  shadows|reflect|refract|ao [on|off]");
                self.console.print("  samples <shadow|ao> <n>");
                self.console.print("  clear             wipe the scrollback");
                self.console.print("  quit              exit");
            }
            "fps" => {
                let fps = self.metrics.fps();
                self.console.print(format!("{fps:.0} FPS, {} glyphs", self.drawn_count));
            }
            "scene" => {
                let entities = self.scene.all_entities().len();
                let meshes = self.scene.entities_with::<MeshComponent>().len();
                self.console.print(format!(
                    "{entities} entities, {meshes} meshes, {} drawn, {} culled, {} materials",
                    self.drawn_count,
                    self.culled_count,
                    self.materials.len()
                ));
            }
            "post" => {
                let on = on_off(arg, self.post.any_enabled());
                self.post = if on {
                    PostProcessSettings::demo()
                } else {
                    PostProcessSettings::none()
                };
                self.console.print(format!("post: {}", if on { "on" } else { "off" }));
            }
            "grid" => {
                let on = on_off(arg, self.subdivision.merges());
                self.subdivision = if on {
                    SubdivisionPolicy::default()
                } else {
                    SubdivisionPolicy::uniform()
                };
                self.console.print(format!("grid: {}", if on { "dynamic" } else { "uniform" }));
            }
            "hud" => {
                self.hud_visible = on_off(arg, self.hud_visible);
                self.console
                    .print(format!("hud: {}", if self.hud_visible { "on" } else { "off" }));
            }
            "style" => match arg {
                Some("block") | Some("blocks") => {
                    self.glyph_style = GlyphStyle::Blocks;
                    self.console.print("style: blocks");
                }
                Some("ramp") => {
                    self.glyph_style = GlyphStyle::Ramp;
                    self.console.print("style: ramp");
                }
                _ => self.console.print_error("usage: style <block|ramp>"),
            },
            "color" => match arg {
                Some("true") => self.set_color_mode(ColorMode::TrueColor),
                Some("256") => self.set_color_mode(ColorMode::Ansi256),
                Some("16") => self.set_color_mode(ColorMode::Ansi16),
                Some("grey") | Some("gray") => self.set_color_mode(ColorMode::Grayscale),
                Some("mono") => self.set_color_mode(ColorMode::Monochrome),
                _ => self
                    .console
                    .print_error("usage: color <true|256|16|grey|mono>"),
            },
            "cam" => match arg {
                Some("first") => {
                    self.camera_rig.set_mode(CameraMode::FirstPerson);
                    self.console.print("camera: first person");
                }
                Some("third") => {
                    self.camera_rig
                        .set_mode(CameraMode::ThirdPerson { distance: 8.0, height_offset: 2.0 });
                    self.console.print("camera: third person");
                }
                Some("orbit") => {
                    self.camera_rig.set_mode(CameraMode::Orbit { distance: 10.0 });
                    self.console.print("camera: orbit");
                }
                _ => self.console.print_error("usage: cam <first|third|orbit>"),
            },
            "trace" => {
                if !self.tracing_available() {
                    self.console.print_error(format!(
                        "raytracing unavailable: {}",
                        self.graphics.ray_tracing().describe()
                    ));
                } else {
                    let on = on_off(arg, self.traced_enabled);
                    if on != self.traced_enabled {
                        self.toggle_tracing();
                    }
                    self.console
                        .print(format!("trace: {}", if self.traced_enabled { "on" } else { "off" }));
                }
            }
            "rays" => {
                self.console.print(self.lighting_summary());
                self.console.print(format!(
                    "gpu: {}",
                    self.graphics.ray_tracing().describe()
                ));
                if let Some(rt) = self.ray_tracer.as_ref() {
                    self.console.print(format!(
                        "as: {} instances, {} tris, {} blas builds, {} tlas builds",
                        rt.instance_count(),
                        rt.triangle_count(),
                        rt.blas_builds(),
                        rt.tlas_builds()
                    ));
                    self.console.print(format!(
                        "settings: depth {} shadow x{} ao x{} radius {:.2}",
                        self.trace_settings.max_depth,
                        self.trace_settings.shadow_samples,
                        self.trace_settings.ao_samples,
                        self.trace_settings.light_radius,
                    ));
                }
            }
            // Per-feature switches, so a suspicious image can be attributed to one
            // kind of ray instead of to "the tracer".
            "shadows" => self.toggle_trace_flag(trace_flags::SHADOWS, "shadows", arg),
            "reflect" => self.toggle_trace_flag(trace_flags::REFLECTIONS, "reflect", arg),
            "refract" => self.toggle_trace_flag(trace_flags::REFRACTION, "refract", arg),
            "ao" => self.toggle_trace_flag(trace_flags::AMBIENT_OCCLUSION, "ao", arg),
            "samples" => match (arg, parts.next().and_then(|v| v.parse::<u32>().ok())) {
                (Some("shadow"), Some(n)) if n >= 1 && n <= 32 => {
                    self.trace_settings.shadow_samples = n;
                    self.console.print(format!("shadow samples: {n}"));
                }
                (Some("ao"), Some(n)) if n <= 32 => {
                    self.trace_settings.ao_samples = n;
                    self.console.print(format!("ao samples: {n}"));
                }
                _ => self
                    .console
                    .print_error("usage: samples <shadow|ao> <count>"),
            },
            "depth" => match arg.and_then(|a| a.parse::<u32>().ok()) {
                Some(d) if d <= 4 => {
                    self.trace_settings.max_depth = d;
                    self.console.print(format!("depth: {d} bounces"));
                }
                _ => self.console.print_error("usage: depth <0-4>"),
            },
            "clear" => {
                self.console.handle(ConsoleAction::Clear);
            }
            "quit" | "exit" => self.should_exit = true,
            other => self
                .console
                .print_error(format!("unknown command: {other} (try 'help')")),
        }
    }

    /// Flip one traced feature bit and report it. `arg` follows the same
    /// `on`/`off`/absent-means-toggle convention as the other console switches.
    fn toggle_trace_flag(&mut self, flag: u32, name: &str, arg: Option<&str>) {
        let current = self.trace_settings.has(flag);
        let on = match arg {
            Some("on") | Some("1") | Some("true") => true,
            Some("off") | Some("0") | Some("false") => false,
            _ => !current,
        };
        self.trace_settings.set(flag, on);
        self.console
            .print(format!("{name}: {}", if on { "on" } else { "off" }));
    }

    fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
        self.console.print(format!("color: {mode:?}"));
    }

    /// Redraw the HUD into the overlay: stats in the top-left, the active
    /// toggles below them, and a centre crosshair.
    fn draw_hud(&mut self) {
        const TEXT: [f32; 3] = [0.85, 0.95, 0.85];
        const DIM: [f32; 3] = [0.45, 0.55, 0.45];

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
        // Which lighting path is running, and what it costs. Without this line
        // the R toggle is unverifiable by eye: raster and traced can look similar
        // on a simple scene, and "did it switch?" must not be a guess.
        self.overlay.draw_text(
            1,
            6,
            &format!("LIGHT {}", self.lighting_summary()),
            if self.traced_enabled { [0.5, 0.85, 1.0] } else { DIM },
        );
        // Say plainly when a UI layer owns the keyboard. A frozen camera with no
        // explanation reads as a broken build, which is exactly how it was
        // reported — the state was correct, it just was not visible anywhere.
        if self.ui_has_focus() {
            let owner = if self.console.is_open() { "CONSOLE" } else { "MENU" };
            self.overlay.draw_text(
                1,
                7,
                &format!("{owner} HAS FOCUS - ESC TO CLOSE"),
                [1.0, 0.75, 0.3],
            );
        } else {
            self.overlay
                .draw_text(1, 7, "TAB MENU  ` CONSOLE", DIM);
        }

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

    /// Movement keys must actually move the rig. This is the plumbing the UI
    /// integration could silently cut: if focus handling or key snapshotting
    /// swallowed WASD, everything would still compile and render.
    #[test]
    fn wasd_moves_the_rig_pivot() {
        use winit::event::ElementState;

        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_pivot(Vec3::ZERO);
        rig.set_yaw_pitch(0.0, 0.0);
        rig.snap();
        let start = rig.pivot();

        let mut input = InputState::new();
        let mut camera_input = CameraInput::new();

        input.key_event(KeyCode::KeyW, ElementState::Pressed);
        for _ in 0..10 {
            camera_input.update(&mut rig, &mut input, 1.0 / 60.0);
        }

        let moved = rig.pivot() - start;
        assert!(
            moved.length() > 0.1,
            "W should move the pivot, moved by {} units",
            moved.length()
        );

        // Releasing must stop the motion.
        input.key_event(KeyCode::KeyW, ElementState::Released);
        let held = rig.pivot();
        for _ in 0..10 {
            camera_input.update(&mut rig, &mut input, 1.0 / 60.0);
        }
        assert!(
            (rig.pivot() - held).length() < 1e-3,
            "pivot kept moving after the key was released"
        );
    }

    /// Mouse look must reach the rig while the look button is held, and be
    /// ignored otherwise.
    #[test]
    fn mouse_look_rotates_only_while_the_button_is_held() {
        use winit::event::{ElementState, MouseButton};

        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_yaw_pitch(0.0, 0.0);
        rig.snap();
        let mut input = InputState::new();
        let mut camera_input = CameraInput::new();

        // Without the button held the delta must be discarded, not banked.
        input.mouse_motion(50.0, 0.0);
        camera_input.update(&mut rig, &mut input, 1.0 / 60.0);
        assert!(rig.yaw().abs() < 1e-6, "look must need the button held");

        input.mouse_button_event(MouseButton::Left, ElementState::Pressed);
        input.mouse_motion(50.0, 0.0);
        camera_input.update(&mut rig, &mut input, 1.0 / 60.0);
        assert!(rig.yaw().abs() > 1e-4, "held button should rotate, yaw = {}", rig.yaw());
    }

    /// A freshly built AppState must not have a UI layer holding focus, or the
    /// camera would be dead on arrival.
    #[test]
    fn ui_starts_without_focus() {
        assert!(!build_menu().is_open(), "the menu must start closed");
        assert!(!build_console().is_open(), "the console must start closed");
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