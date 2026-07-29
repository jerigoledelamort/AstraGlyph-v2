// Camera rig — reusable camera presets (first-person, third-person, orbit) plus
// framerate-independent smoothing ("dampening") of the camera transform.
//
// Design notes:
// - Pure logic: no GPU, no winit, no input handling. The app layer feeds plain
//   numbers (mouse deltas already scaled by sensitivity, movement deltas in world
//   units, dt in seconds) and reads back a position/target pair. This keeps the rig
//   trivially unit-testable and lets the same code drive a player camera, a cutscene
//   camera or an editor camera.
// - The rig keeps TWO states: the *desired* state (derived analytically from mode +
//   pivot + yaw/pitch) and the *smoothed* state (what the renderer should actually
//   use). `update(dt)` moves the latter toward the former.
// - Smoothing uses exponential decay with a half-life instead of a fixed per-frame
//   lerp factor. A fixed factor (`current += (target - current) * 0.1`) is a bug:
//   at 300 FPS the camera snaps, at 30 FPS it drags. See `blend_factor` for the
//   formula and its derivation.
// - Yaw/pitch -> direction convention is identical to the first-person controller in
//   `app/state.rs`, so the two can be swapped without the view flipping:
//       forward = ( cos(yaw)*cos(pitch), sin(pitch), -sin(yaw)*cos(pitch) )
//       right   = ( sin(yaw),            0.0,         cos(yaw)            )
//   `right` is deliberately horizontal (no roll), which is what a character camera
//   wants: strafing must not lift you off the ground when looking up.

use crate::engine::math::{radians, Vec3};
use crate::scene::camera::Camera;

use std::f32::consts::{LN_2, PI, TAU};

/// Camera placement preset.
///
/// The mode only decides *where* the camera sits relative to the pivot and *what*
/// it looks at; yaw/pitch/pivot are shared state owned by the rig, so switching
/// modes preserves the player's orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraMode {
    /// Eye sits exactly at the pivot and looks along yaw/pitch (classic FPS).
    FirstPerson,
    /// Camera trails the pivot by `distance` along the look direction and is
    /// lifted by `height_offset`, always looking at the pivot itself.
    ThirdPerson {
        /// Horizontal-ish trailing distance behind the pivot, in world units.
        distance: f32,
        /// Vertical lift above the pivot, in world units (may be negative).
        height_offset: f32,
    },
    /// Camera orbits the pivot on a sphere of radius `distance`, looking inward.
    ///
    /// Positionally this is `ThirdPerson { height_offset: 0.0 }`; it exists as a
    /// separate preset because the intent differs (inspecting a fixed object vs.
    /// following a moving character), and because zoom/pivot are driven by
    /// different app-level code paths.
    Orbit {
        /// Orbit radius in world units.
        distance: f32,
    },
}

/// Camera presets + dampening. Produces a position/target pair for a [`Camera`].
#[derive(Clone, Debug)]
pub struct CameraRig {
    /// Active placement preset (distances already clamped to the sane range).
    mode: CameraMode,
    /// Point of interest: the player/object the rig is built around.
    pivot: Vec3,
    /// Horizontal look angle in radians, wrapped to [-PI, PI).
    yaw: f32,
    /// Vertical look angle in radians, clamped to +/-89 degrees.
    pitch: f32,
    /// Smoothing half-life in seconds. 0.0 means "no smoothing".
    smoothing: f32,
    /// Smoothed eye position (the value the renderer consumes).
    position: Vec3,
    /// Smoothed look-at point (the value the renderer consumes).
    target: Vec3,
}

impl CameraRig {
    /// Pitch limit. Stopping short of straight up/down keeps the `look_at` basis
    /// well-conditioned: at exactly +/-90 degrees `forward` becomes parallel to the
    /// world up vector and the view matrix degenerates.
    const PITCH_LIMIT_DEGREES: f32 = 89.0;

    /// Smallest allowed orbit/trail distance. Zero would put the eye on the pivot,
    /// making the look-at direction a zero vector.
    pub const MIN_DISTANCE: f32 = 0.25;

    /// Largest allowed orbit/trail distance — keeps the camera inside any sane
    /// depth range so geometry does not fall behind the far plane.
    pub const MAX_DISTANCE: f32 = 500.0;

    /// Default smoothing half-life: the camera covers half of the remaining gap
    /// every 80 ms. Snappy enough for gameplay, still visibly damped.
    pub const DEFAULT_SMOOTHING: f32 = 0.08;

    /// Create a rig in the given mode, pivoted at the origin and facing +X.
    ///
    /// The smoothed state starts already settled, so the first rendered frame is
    /// not an interpolation from the origin.
    pub fn new(mode: CameraMode) -> Self {
        let mut rig = Self {
            mode: Self::sanitize_mode(mode),
            pivot: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            smoothing: Self::DEFAULT_SMOOTHING,
            position: Vec3::ZERO,
            target: Vec3::ZERO,
        };
        rig.snap();
        rig
    }

    /// Active placement preset.
    pub fn mode(&self) -> CameraMode {
        self.mode
    }

    /// Switch preset. The smoothed state is intentionally left alone so the
    /// transition is animated by `update`; call [`CameraRig::snap`] for a hard cut.
    pub fn set_mode(&mut self, mode: CameraMode) {
        self.mode = Self::sanitize_mode(mode);
    }

    /// Point of interest the camera is built around.
    pub fn pivot(&self) -> Vec3 {
        self.pivot
    }

    /// Teleport the pivot (e.g. after a respawn). Follow [`CameraRig::snap`] if the
    /// camera should not visibly fly across the level.
    ///
    /// A non-finite pivot is ignored: once NaN reaches the smoothed state it
    /// survives every subsequent `lerp` (`NaN * 0.0` is still NaN), so a single
    /// bad frame would blank the view until the next [`CameraRig::snap`].
    pub fn set_pivot(&mut self, pivot: Vec3) {
        if Self::is_finite(pivot) {
            self.pivot = pivot;
        }
    }

    /// Move the pivot by a world-space delta (the usual per-frame walk update).
    ///
    /// Non-finite deltas are ignored, for the same reason as [`CameraRig::set_pivot`].
    pub fn move_pivot(&mut self, delta: Vec3) {
        if Self::is_finite(delta) {
            self.pivot += delta;
        }
    }

    /// Horizontal look angle in radians.
    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Vertical look angle in radians.
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Apply a look delta (already scaled by the caller's mouse sensitivity).
    ///
    /// Pitch is clamped to +/-89 degrees; yaw wraps so it cannot drift into the
    /// range where `f32` loses angular precision during a long session. A
    /// non-finite delta leaves its axis untouched (see [`CameraRig::set_yaw_pitch`]).
    pub fn rotate(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.set_yaw_pitch(self.yaw + yaw_delta, self.pitch + pitch_delta);
    }

    /// Set the look angles directly. Pitch is clamped, yaw is wrapped.
    ///
    /// Each axis is filtered independently: a non-finite value is *dropped* (the
    /// previous angle survives) rather than coerced to 0.0. NaN would otherwise
    /// poison every derived vector — `clamp` propagates it — and snapping the view
    /// to "yaw 0, level" because of one bad delta is a visible glitch, whereas
    /// ignoring the frame is invisible.
    pub fn set_yaw_pitch(&mut self, yaw: f32, pitch: f32) {
        if yaw.is_finite() {
            self.yaw = Self::wrap_angle(yaw);
        }
        if pitch.is_finite() {
            let limit = radians(Self::PITCH_LIMIT_DEGREES);
            self.pitch = pitch.clamp(-limit, limit);
        }
    }

    /// Unit look direction implied by yaw/pitch.
    pub fn forward(&self) -> Vec3 {
        let (sin_y, cos_y) = (self.yaw.sin(), self.yaw.cos());
        let (sin_p, cos_p) = (self.pitch.sin(), self.pitch.cos());
        Vec3::new(cos_y * cos_p, sin_p, -sin_y * cos_p)
    }

    /// Unit right vector in the horizontal plane (never tilts with pitch).
    pub fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos())
    }

    /// Current orbit/trail distance, or `None` in first-person.
    pub fn distance(&self) -> Option<f32> {
        match self.mode {
            CameraMode::FirstPerson => None,
            CameraMode::ThirdPerson { distance, .. } | CameraMode::Orbit { distance } => {
                Some(distance)
            }
        }
    }

    /// Change the orbit/trail distance by `delta` (positive pulls the camera away),
    /// clamped to [`CameraRig::MIN_DISTANCE`]..=[`CameraRig::MAX_DISTANCE`].
    ///
    /// A no-op in first-person: there is no distance to scale there, and silently
    /// mutating an unused field would surprise a caller that switches modes later.
    pub fn zoom(&mut self, delta: f32) {
        if !delta.is_finite() {
            return;
        }
        match &mut self.mode {
            CameraMode::FirstPerson => {}
            CameraMode::ThirdPerson { distance, .. } | CameraMode::Orbit { distance } => {
                *distance = (*distance + delta).clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE);
            }
        }
    }

    /// Where the camera wants to be, given the current mode/pivot/orientation.
    pub fn desired_position(&self) -> Vec3 {
        match self.mode {
            CameraMode::FirstPerson => self.pivot,
            CameraMode::ThirdPerson { distance, height_offset } => {
                self.pivot - self.forward() * distance + Vec3::UNIT_Y * height_offset
            }
            CameraMode::Orbit { distance } => self.pivot - self.forward() * distance,
        }
    }

    /// What the camera wants to look at, given the current mode.
    pub fn desired_target(&self) -> Vec3 {
        match self.mode {
            // One unit ahead is enough: only the direction matters to `look_at`.
            CameraMode::FirstPerson => self.pivot + self.forward(),
            CameraMode::ThirdPerson { .. } | CameraMode::Orbit { .. } => self.pivot,
        }
    }

    /// Smoothing half-life in seconds (0.0 = instant).
    pub fn smoothing(&self) -> f32 {
        self.smoothing
    }

    /// Set the smoothing half-life: the time it takes the camera to close half of
    /// the remaining gap to the desired state. Non-positive or non-finite values
    /// mean "no smoothing" and are stored as exactly 0.0.
    pub fn set_smoothing(&mut self, half_life_seconds: f32) {
        self.smoothing = if half_life_seconds.is_finite() && half_life_seconds > 0.0 {
            half_life_seconds
        } else {
            0.0
        };
    }

    /// Advance the smoothed state toward the desired state by `dt` seconds.
    ///
    /// A no-op for `dt <= 0.0` (paused frame, or a clock that went backwards) and
    /// for non-finite `dt`, both of which would otherwise inject NaN.
    pub fn update(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        // Both endpoints are sampled once, before either is written: position and
        // target must be blended against the *same* desired pose, or a mode that
        // derives one from the other would smooth along a moving reference.
        let desired_position = self.desired_position();
        let desired_target = self.desired_target();
        let t = self.blend_factor(dt);
        self.position = self.position.lerp(desired_position, t);
        self.target = self.target.lerp(desired_target, t);
    }

    /// Smoothed eye position — feed this to the renderer.
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Smoothed look-at point — feed this to the renderer.
    pub fn target(&self) -> Vec3 {
        self.target
    }

    /// Collapse the smoothed state onto the desired state. Use after a teleport or
    /// a mode switch, where interpolating would sweep the camera through geometry.
    pub fn snap(&mut self) {
        self.position = self.desired_position();
        self.target = self.desired_target();
    }

    /// Copy the smoothed transform into an existing camera, leaving its projection
    /// (and therefore its aspect ratio, set on resize) untouched.
    pub fn apply_to(&self, camera: &mut Camera) {
        camera.position = self.position;
        camera.target = self.target;
    }

    /// Fraction of the remaining gap to close over `dt`.
    ///
    /// Exponential decay: the gap `g` obeys `g(t) = g(0) * 2^(-t / half_life)`, so
    /// after `dt` the surviving fraction is `2^(-dt/half_life)` and the closed
    /// fraction is `1 - 2^(-dt/half_life)` = `1 - exp(-dt * ln2 / half_life)`.
    ///
    /// Why this and not a constant per-frame lerp: because decay composes over
    /// time, N steps of `dt/N` collapse to exactly one step of `dt`, which makes
    /// the motion identical at 30 and 300 FPS. The result is also always inside
    /// [0, 1], so the camera approaches the target monotonically and can neither
    /// overshoot nor oscillate — even for an enormous `dt` after a tab-out, where
    /// the factor simply saturates at 1.0.
    fn blend_factor(&self, dt: f32) -> f32 {
        if self.smoothing <= 0.0 {
            return 1.0;
        }
        let decayed = (-dt * LN_2 / self.smoothing).exp();
        (1.0 - decayed).clamp(0.0, 1.0)
    }

    /// Force a mode's distance into the valid range (also fixes NaN distances).
    fn sanitize_mode(mode: CameraMode) -> CameraMode {
        let fix = |d: f32| {
            if d.is_nan() {
                Self::MIN_DISTANCE
            } else {
                d.clamp(Self::MIN_DISTANCE, Self::MAX_DISTANCE)
            }
        };
        match mode {
            CameraMode::FirstPerson => CameraMode::FirstPerson,
            CameraMode::ThirdPerson { distance, height_offset } => CameraMode::ThirdPerson {
                distance: fix(distance),
                height_offset: if height_offset.is_finite() { height_offset } else { 0.0 },
            },
            CameraMode::Orbit { distance } => CameraMode::Orbit { distance: fix(distance) },
        }
    }

    /// Wrap an angle into [-PI, PI) so long play sessions keep full precision.
    ///
    /// The non-finite guard is redundant for the current caller but kept local:
    /// `rem_euclid` on an infinity yields NaN, and this helper must never be the
    /// place a NaN enters the rig.
    fn wrap_angle(angle: f32) -> f32 {
        if !angle.is_finite() {
            return 0.0;
        }
        (angle + PI).rem_euclid(TAU) - PI
    }

    /// Whether every component of `v` is finite. Used to reject poisoned input at
    /// the setters, which is the only place it can still be contained.
    fn is_finite(v: Vec3) -> bool {
        v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
    }
}

impl Default for CameraRig {
    fn default() -> Self {
        Self::new(CameraMode::FirstPerson)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::degrees;
    use crate::scene::camera::Projection;

    const EPS: f32 = 1e-5;

    fn third_person(distance: f32, height_offset: f32) -> CameraRig {
        CameraRig::new(CameraMode::ThirdPerson { distance, height_offset })
    }

    #[test]
    fn pitch_clamped_at_both_extremes() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        let limit = radians(89.0);

        rig.rotate(0.0, radians(400.0));
        assert!((rig.pitch() - limit).abs() < EPS, "pitch = {}", degrees(rig.pitch()));

        rig.rotate(0.0, radians(-400.0));
        assert!((rig.pitch() + limit).abs() < EPS, "pitch = {}", degrees(rig.pitch()));

        // Direct setter must clamp too, not just the incremental path.
        rig.set_yaw_pitch(0.0, 100.0);
        assert!((rig.pitch() - limit).abs() < EPS);
        rig.set_yaw_pitch(0.0, -100.0);
        assert!((rig.pitch() + limit).abs() < EPS);

        // Within range the value passes through untouched.
        rig.set_yaw_pitch(0.0, radians(30.0));
        assert!((rig.pitch() - radians(30.0)).abs() < EPS);
    }

    #[test]
    fn yaw_wraps_but_preserves_direction() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_yaw_pitch(radians(30.0), 0.0);
        let forward_30 = rig.forward();

        // 390 degrees is the same heading as 30, and must wrap into (-PI, PI].
        rig.set_yaw_pitch(radians(390.0), 0.0);
        assert!(rig.yaw().abs() <= PI + EPS);
        assert!((rig.forward() - forward_30).length() < 1e-4);
    }

    #[test]
    fn forward_and_right_are_orthonormal_basis_vectors() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        // Sweep a grid of angles: the invariants must hold everywhere, not just at 0.
        for yaw_deg in [-170.0f32, -90.0, -33.0, 0.0, 45.0, 120.0, 179.0] {
            for pitch_deg in [-89.0f32, -60.0, -12.0, 0.0, 25.0, 89.0] {
                rig.set_yaw_pitch(radians(yaw_deg), radians(pitch_deg));
                let f = rig.forward();
                let r = rig.right();
                assert!((f.length() - 1.0).abs() < EPS, "forward not unit at {yaw_deg}/{pitch_deg}");
                assert!((r.length() - 1.0).abs() < EPS, "right not unit at {yaw_deg}/{pitch_deg}");
                assert!(r.y.abs() < EPS, "right must stay horizontal");
                assert!(f.dot(r).abs() < EPS, "forward/right not perpendicular");
            }
        }
    }

    #[test]
    fn forward_matches_controller_convention() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_yaw_pitch(0.0, 0.0);
        // yaw=0, pitch=0 must look down +X with right along +Z, as in app/state.rs.
        assert!((rig.forward() - Vec3::UNIT_X).length() < EPS);
        assert!((rig.right() - Vec3::UNIT_Z).length() < EPS);

        rig.set_yaw_pitch(radians(90.0), 0.0);
        assert!((rig.forward() - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4);
        assert!((rig.right() - Vec3::UNIT_X).length() < 1e-4);
    }

    #[test]
    fn first_person_sits_on_the_pivot_and_looks_ahead() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_pivot(Vec3::new(3.0, 1.5, -2.0));
        rig.set_yaw_pitch(radians(40.0), radians(-15.0));

        assert!((rig.desired_position() - rig.pivot()).length() < EPS);
        // Look direction must equal `forward`, at unit distance.
        let dir = rig.desired_target() - rig.desired_position();
        assert!((dir.length() - 1.0).abs() < EPS);
        assert!((dir - rig.forward()).length() < EPS);
    }

    #[test]
    fn orbit_keeps_configured_radius_and_targets_the_pivot() {
        let mut rig = CameraRig::new(CameraMode::Orbit { distance: 7.5 });
        rig.set_pivot(Vec3::new(-4.0, 2.0, 6.0));

        for yaw_deg in [0.0f32, 55.0, 140.0, -120.0] {
            for pitch_deg in [-70.0f32, 0.0, 30.0] {
                rig.set_yaw_pitch(radians(yaw_deg), radians(pitch_deg));
                let d = (rig.desired_position() - rig.pivot()).length();
                assert!((d - 7.5).abs() < 1e-4, "radius drifted to {d}");
                assert!((rig.desired_target() - rig.pivot()).length() < EPS);
            }
        }
        // Camera is placed behind the pivot, i.e. opposite the look direction.
        rig.set_yaw_pitch(0.0, 0.0);
        assert!((rig.desired_position() - (rig.pivot() - Vec3::UNIT_X * 7.5)).length() < EPS);
    }

    #[test]
    fn third_person_trails_at_distance_and_lifts_by_offset() {
        // No lift: the eye is exactly `distance` away from the pivot.
        let mut rig = third_person(6.0, 0.0);
        rig.set_pivot(Vec3::new(1.0, 0.0, 1.0));
        rig.set_yaw_pitch(radians(25.0), radians(10.0));
        let d = (rig.desired_position() - rig.pivot()).length();
        assert!((d - 6.0).abs() < 1e-4, "distance = {d}");
        assert!((rig.desired_target() - rig.pivot()).length() < EPS);

        // With lift: the offset is purely vertical and composes as a right triangle.
        let mut lifted = third_person(6.0, 2.0);
        lifted.set_pivot(Vec3::ZERO);
        lifted.set_yaw_pitch(0.0, 0.0);
        let p = lifted.desired_position();
        assert!((p.y - 2.0).abs() < EPS, "height offset not applied");
        let expected = (6.0f32 * 6.0 + 2.0 * 2.0).sqrt();
        assert!((p.length() - expected).abs() < 1e-4);
        assert!((lifted.desired_target() - Vec3::ZERO).length() < EPS);
    }

    #[test]
    fn zoom_clamps_and_ignores_first_person() {
        let mut rig = CameraRig::new(CameraMode::Orbit { distance: 10.0 });
        rig.zoom(5.0);
        assert_eq!(rig.distance(), Some(15.0));
        rig.zoom(-3.0);
        assert_eq!(rig.distance(), Some(12.0));

        // Lower clamp.
        rig.zoom(-1000.0);
        assert_eq!(rig.distance(), Some(CameraRig::MIN_DISTANCE));
        // Upper clamp.
        rig.zoom(1e9);
        assert_eq!(rig.distance(), Some(CameraRig::MAX_DISTANCE));

        // Third-person zooms too, and keeps its height offset.
        let mut tp = third_person(4.0, 1.0);
        tp.zoom(2.0);
        assert_eq!(tp.mode(), CameraMode::ThirdPerson { distance: 6.0, height_offset: 1.0 });

        // First-person has nothing to zoom.
        let mut fp = CameraRig::new(CameraMode::FirstPerson);
        fp.zoom(100.0);
        assert_eq!(fp.mode(), CameraMode::FirstPerson);
        assert_eq!(fp.distance(), None);
    }

    #[test]
    fn degenerate_distances_are_sanitized() {
        // Zero distance would collapse eye onto target and break look_at.
        let rig = CameraRig::new(CameraMode::Orbit { distance: 0.0 });
        assert_eq!(rig.distance(), Some(CameraRig::MIN_DISTANCE));

        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_mode(CameraMode::ThirdPerson { distance: 1e9, height_offset: 0.0 });
        assert_eq!(rig.distance(), Some(CameraRig::MAX_DISTANCE));
    }

    #[test]
    fn new_starts_settled_and_snap_collapses_the_gap() {
        let rig = CameraRig::new(CameraMode::Orbit { distance: 5.0 });
        assert!((rig.position() - rig.desired_position()).length() < EPS);
        assert!((rig.target() - rig.desired_target()).length() < EPS);

        let mut rig = rig;
        rig.set_pivot(Vec3::new(50.0, -20.0, 30.0));
        rig.set_yaw_pitch(radians(77.0), radians(-40.0));
        // Desired moved far away; smoothed state has not followed yet.
        assert!((rig.position() - rig.desired_position()).length() > 1.0);

        rig.snap();
        assert!((rig.position() - rig.desired_position()).length() < EPS);
        assert!((rig.target() - rig.desired_target()).length() < EPS);
    }

    #[test]
    fn update_with_zero_or_invalid_dt_is_a_noop() {
        let mut rig = CameraRig::new(CameraMode::Orbit { distance: 5.0 });
        rig.set_pivot(Vec3::new(10.0, 0.0, 0.0));
        let before_pos = rig.position();
        let before_tgt = rig.target();

        rig.update(0.0);
        assert_eq!(rig.position(), before_pos);
        assert_eq!(rig.target(), before_tgt);

        rig.update(-0.5);
        assert_eq!(rig.position(), before_pos);

        rig.update(f32::NAN);
        assert!(rig.position().x.is_finite() && rig.position().y.is_finite());
        assert_eq!(rig.position(), before_pos);
    }

    #[test]
    fn update_converges_monotonically_toward_the_target() {
        let mut rig = CameraRig::new(CameraMode::Orbit { distance: 5.0 });
        rig.set_smoothing(0.1);
        rig.set_pivot(Vec3::new(20.0, 5.0, -8.0));

        let mut gap = (rig.desired_position() - rig.position()).length();
        assert!(gap > 1.0);
        for _ in 0..30 {
            rig.update(1.0 / 60.0);
            let next = (rig.desired_position() - rig.position()).length();
            assert!(next < gap, "gap did not shrink: {next} >= {gap}");
            gap = next;
        }

        // A huge dt (tab-out freeze) must land on the target without overshooting.
        rig.update(10.0);
        let residual = (rig.desired_position() - rig.position()).length();
        assert!(residual < 1e-4, "did not converge, residual = {residual}");
        assert!((rig.target() - rig.desired_target()).length() < 1e-4);
    }

    #[test]
    fn smoothing_never_overshoots_a_single_axis() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_smoothing(0.05);
        rig.set_pivot(Vec3::new(0.0, 100.0, 0.0));
        // Deliberately large steps relative to the half-life: a naive
        // `pos += (target - pos) * k * dt` would blow past the target here.
        for _ in 0..10 {
            rig.update(0.5);
            assert!(rig.position().y <= 100.0 + EPS, "overshoot: {}", rig.position().y);
            assert!(rig.position().y >= 0.0, "moved backwards: {}", rig.position().y);
        }
    }

    #[test]
    fn smoothing_is_framerate_independent() {
        let mut fine = CameraRig::new(CameraMode::Orbit { distance: 5.0 });
        fine.set_smoothing(0.25);
        fine.set_pivot(Vec3::new(30.0, 10.0, -15.0));
        let initial_gap = (fine.desired_position() - fine.position()).length();

        let mut coarse = fine.clone();

        // 100 small steps vs. one big step over the same wall-clock second.
        for _ in 0..100 {
            fine.update(0.01);
        }
        coarse.update(1.0);

        let divergence = (fine.position() - coarse.position()).length();
        assert!(
            divergence < initial_gap * 0.02,
            "framerate dependent: divergence {divergence} of gap {initial_gap}"
        );
    }

    #[test]
    fn zero_smoothing_is_instant() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_smoothing(0.0);
        rig.set_pivot(Vec3::new(7.0, -3.0, 11.0));
        rig.update(1.0 / 240.0);
        assert!((rig.position() - rig.desired_position()).length() < EPS);
        assert!((rig.target() - rig.desired_target()).length() < EPS);

        // Negative / non-finite half-lives collapse to "instant", not to a panic
        // or a division by zero.
        rig.set_smoothing(-1.0);
        assert_eq!(rig.smoothing(), 0.0);
        rig.set_smoothing(f32::NAN);
        assert_eq!(rig.smoothing(), 0.0);
        rig.set_smoothing(f32::INFINITY);
        assert_eq!(rig.smoothing(), 0.0);

        rig.set_pivot(Vec3::new(-1.0, 2.0, -3.0));
        rig.update(0.016);
        assert!((rig.position() - rig.desired_position()).length() < EPS);
    }

    #[test]
    fn set_mode_animates_but_snap_cuts() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_smoothing(0.2);
        let before = rig.position();

        rig.set_mode(CameraMode::Orbit { distance: 12.0 });
        // Mode switch alone must not teleport the smoothed state.
        assert_eq!(rig.position(), before);
        assert!((rig.desired_position() - before).length() > 1.0);

        rig.snap();
        assert!((rig.position() - rig.desired_position()).length() < EPS);
    }

    #[test]
    fn move_pivot_accumulates() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.move_pivot(Vec3::new(1.0, 0.0, 0.0));
        rig.move_pivot(Vec3::new(0.0, 2.0, 0.0));
        rig.move_pivot(Vec3::new(0.0, 0.0, -3.0));
        assert_eq!(rig.pivot(), Vec3::new(1.0, 2.0, -3.0));
    }

    #[test]
    fn apply_to_writes_transform_and_preserves_projection() {
        let projection = Projection::perspective(radians(60.0), 16.0 / 9.0, 0.1, 100.0);
        let mut camera = Camera::new(Vec3::ZERO, Vec3::UNIT_Z, Vec3::UNIT_Y, projection);

        let mut rig = CameraRig::new(CameraMode::Orbit { distance: 4.0 });
        rig.set_pivot(Vec3::new(2.0, 3.0, 4.0));
        rig.snap();
        rig.apply_to(&mut camera);

        assert_eq!(camera.position, rig.position());
        assert_eq!(camera.target, rig.target());
        // Projection (and the aspect ratio inside it) must survive untouched.
        assert_eq!(camera.projection, projection);
        assert_eq!(camera.up, Vec3::UNIT_Y);
    }

    #[test]
    fn wrapped_yaw_stays_in_range() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        for step in -12i32..=12 {
            rig.set_yaw_pitch(radians(step as f32 * 55.0), 0.0);
            let y = rig.yaw();
            assert!(y >= -PI - EPS && y <= PI + EPS, "yaw escaped the range: {y}");
        }

        // +/-PI is the same heading either way; only the direction has to survive.
        rig.set_yaw_pitch(PI, 0.0);
        assert!((rig.yaw().abs() - PI).abs() < EPS, "yaw = {}", rig.yaw());
        assert!((rig.forward() - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn non_finite_look_input_is_dropped_not_coerced() {
        let mut rig = CameraRig::new(CameraMode::FirstPerson);
        rig.set_yaw_pitch(radians(70.0), radians(-25.0));
        let (yaw, pitch) = (rig.yaw(), rig.pitch());
        // Guard the test itself: a coerced-to-zero bug must be detectable.
        assert!(yaw.abs() > 0.1 && pitch.abs() > 0.1);

        rig.rotate(f32::NAN, 0.0);
        assert!((rig.yaw() - yaw).abs() < EPS, "yaw snapped to {}", rig.yaw());
        assert!((rig.pitch() - pitch).abs() < EPS);

        rig.rotate(0.0, f32::INFINITY);
        assert!((rig.yaw() - yaw).abs() < EPS);
        assert!((rig.pitch() - pitch).abs() < EPS, "pitch snapped to {}", rig.pitch());

        rig.set_yaw_pitch(f32::NEG_INFINITY, f32::NAN);
        assert!((rig.yaw() - yaw).abs() < EPS);
        assert!((rig.pitch() - pitch).abs() < EPS);

        // The axes are filtered independently: a half-bad call still applies the
        // good half instead of discarding the whole frame.
        rig.set_yaw_pitch(f32::NAN, radians(10.0));
        assert!((rig.yaw() - yaw).abs() < EPS);
        assert!((rig.pitch() - radians(10.0)).abs() < EPS);
    }

    #[test]
    fn non_finite_pivot_cannot_poison_the_smoothed_state() {
        let mut rig = CameraRig::new(CameraMode::Orbit { distance: 5.0 });
        rig.set_smoothing(0.1);
        rig.set_pivot(Vec3::new(2.0, 3.0, 4.0));

        rig.set_pivot(Vec3::new(f32::NAN, 0.0, 0.0));
        assert_eq!(rig.pivot(), Vec3::new(2.0, 3.0, 4.0));
        rig.move_pivot(Vec3::new(0.0, f32::INFINITY, 0.0));
        assert_eq!(rig.pivot(), Vec3::new(2.0, 3.0, 4.0));

        // Why this matters: NaN in `position` survives every later lerp, because
        // `NaN * 0.0` is still NaN. Rejecting it at the setter is the only cheap
        // place to stop it — afterwards only `snap` could recover the view.
        for _ in 0..5 {
            rig.update(1.0 / 60.0);
        }
        let p = rig.position();
        let t = rig.target();
        assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite(), "position = {p}");
        assert!(t.x.is_finite() && t.y.is_finite() && t.z.is_finite(), "target = {t}");
        // And the rig is still converging on the real pivot, not a stale one.
        assert!((rig.desired_target() - Vec3::new(2.0, 3.0, 4.0)).length() < EPS);
    }

    #[test]
    fn default_rig_is_first_person_at_origin() {
        let rig = CameraRig::default();
        assert_eq!(rig.mode(), CameraMode::FirstPerson);
        assert_eq!(rig.pivot(), Vec3::ZERO);
        assert_eq!(rig.position(), Vec3::ZERO);
        assert!(rig.smoothing() > 0.0);
    }
}
