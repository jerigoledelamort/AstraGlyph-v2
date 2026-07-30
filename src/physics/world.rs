// The physics world: holds bodies, steps them, and resolves their contacts.
//
// The step order is fixed and matters:
//
//   1. integrate    — gravity and velocity move every dynamic body
//   2. detect       — find every overlapping pair
//   3. resolve      — impulses to correct velocity, then positional correction
//
// Detection after integration rather than before, because a body that has not
// moved yet cannot have a new contact; and positional correction after the
// velocity impulse, because the impulse is what stops the bodies approaching and
// the correction only cleans up the overlap that already happened. Reversing
// either pair makes bodies sink one frame and jump the next.

use crate::engine::geometry::collision::{self, Contact};
use crate::engine::geometry::{Basis, Shape, WorldShape};
use crate::engine::math::Vec3;
use crate::engine::geometry::{ray, Ray, RayHit};

use super::body::{RigidBody, DEFAULT_GRAVITY, REST_SPEED};

/// Fraction of the remaining overlap corrected per step.
///
/// Not 1.0: correcting the whole overlap in one step overshoots, because the
/// contact was detected at the end of a finite time step and the bodies are
/// already moving apart. Overshooting makes a resting stack pop upward. Not too
/// small either, or bodies visibly sink into the floor before recovering.
const CORRECTION_RATE: f32 = 0.5;

/// Overlap left uncorrected, so resting bodies keep a stable contact instead of
/// oscillating between "touching" and "not touching" every frame.
const PENETRATION_ALLOWANCE: f32 = 5.0e-4;

/// Largest time step the world will integrate in one go.
///
/// A long step (an alt-tab, a breakpoint, a stalled frame) would move a body
/// further than its own size and straight through a wall — the classic tunnelling
/// failure. Longer requests are split into several substeps rather than clamped,
/// so a slow frame runs slow rather than silently skipping simulation.
const MAX_SUBSTEP: f32 = 1.0 / 120.0;

/// Fraction of the thinnest collider a body may cross in one substep.
///
/// A fixed substep is not enough on its own. Discrete collision detection only
/// sees an *overlap*, so a body has to still be inside the obstacle when the
/// substep ends. Worse, if it passes the obstacle's centre the contact normal
/// flips and the solver reads the body as already separating — it gets a contact
/// and declines to act on it, which is exactly what a fast body did through a
/// 1-metre wall at 120 m/s with 1/120 s substeps.
///
/// So the substep is sized by motion rather than by time: no body advances more
/// than this fraction of the smallest collider in the world. That keeps every
/// approach resolvable without paying for tiny substeps when nothing is moving.
const MAX_TRAVEL_FRACTION: f32 = 0.4;

/// Floor on the motion-derived substep, so a pathologically fast body or a
/// pathologically small collider cannot drive the substep toward zero and hang
/// the frame. Reaching this floor means tunnelling is possible again — which is
/// why `substeps()` is public: it is the visible symptom.
const MIN_SUBSTEP: f32 = 1.0 / 2000.0;

/// Most substeps a single `step` call will run.
///
/// Without this, a very long delta turns into thousands of substeps and the frame
/// that was already slow becomes slower still — a death spiral. Hitting the cap
/// means time is genuinely lost, which is the right failure: the simulation stays
/// stable and only falls behind.
const MAX_SUBSTEPS: u32 = 8;

/// A handle to a body in the world. Stable across steps; invalidated only by
/// removal, which this world does not offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyId(usize);

impl BodyId {
    /// Index into the world's body list.
    pub fn index(self) -> usize {
        self.0
    }
}

/// What a raycast against the world found.
#[derive(Clone, Copy, Debug)]
pub struct WorldRayHit {
    /// The body that was hit.
    pub body: BodyId,
    /// Where, how far and which way.
    pub hit: RayHit,
}

/// The physics world.
pub struct PhysicsWorld {
    bodies: Vec<RigidBody>,
    /// Uniform acceleration applied to every dynamic body.
    pub gravity: Vec3,
    /// Contacts found during the most recent step, kept for inspection and for
    /// the HUD's contact count.
    contacts: Vec<(BodyId, BodyId, Contact)>,
    /// Substeps run during the most recent `step`, so a caller can see when the
    /// substep cap is being hit rather than guessing why physics lags.
    substeps: u32,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            gravity: DEFAULT_GRAVITY,
            contacts: Vec::new(),
            substeps: 0,
        }
    }

    /// Add a body and return its handle.
    pub fn add(&mut self, body: RigidBody) -> BodyId {
        self.bodies.push(body);
        BodyId(self.bodies.len() - 1)
    }

    pub fn body(&self, id: BodyId) -> Option<&RigidBody> {
        self.bodies.get(id.0)
    }

    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut RigidBody> {
        self.bodies.get_mut(id.0)
    }

    pub fn bodies(&self) -> &[RigidBody] {
        &self.bodies
    }

    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Contacts from the most recent step.
    pub fn contacts(&self) -> &[(BodyId, BodyId, Contact)] {
        &self.contacts
    }

    /// Substeps run in the most recent `step` call.
    pub fn substeps(&self) -> u32 {
        self.substeps
    }

    /// A body's collision shape placed in world space.
    ///
    /// The basis is identity because this world has no angular state (see
    /// `body.rs`); the shapes and the collision routines support orientation, so
    /// adding rotation later does not need a new representation.
    pub fn world_shape(&self, id: BodyId) -> Option<WorldShape> {
        let body = self.body(id)?;
        Some(WorldShape::oriented(body.position, body.shape, Basis::IDENTITY))
    }

    /// Advance the simulation by `dt`, split into substeps short enough that no
    /// body can cross an obstacle within one of them.
    pub fn step(&mut self, dt: f32) {
        self.substeps = 0;
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        let mut remaining = dt;
        while remaining > 0.0 && self.substeps < MAX_SUBSTEPS {
            // Recomputed each iteration: an impulse in the previous substep may
            // have changed the fastest speed in the world by an order of
            // magnitude, and sizing the rest of the frame on a stale figure is
            // how a collision response tunnels out the far side.
            let h = remaining.min(self.safe_substep());
            self.substep(h);
            remaining -= h;
            self.substeps += 1;
        }
    }

    /// Longest substep in which no body travels more than
    /// `MAX_TRAVEL_FRACTION` of the thinnest collider present.
    fn safe_substep(&self) -> f32 {
        let fastest = self
            .bodies
            .iter()
            .filter(|b| b.is_movable())
            .map(|b| b.velocity.length())
            .fold(0.0f32, f32::max);
        if fastest <= 1.0e-6 {
            return MAX_SUBSTEP;
        }
        // The thinnest collider is what a body can slip through. A plane has no
        // thickness at all, so it cannot be tunnelled *past* in the same sense —
        // its half-space extends forever — and it is excluded rather than
        // collapsing the substep to the floor.
        let thinnest = self
            .bodies
            .iter()
            .filter_map(|b| match b.shape {
                Shape::Sphere { radius } => Some(radius.abs()),
                Shape::Box { half_extents } => Some(
                    half_extents
                        .x
                        .abs()
                        .min(half_extents.y.abs())
                        .min(half_extents.z.abs()),
                ),
                Shape::Plane { .. } => None,
            })
            .filter(|r| *r > 1.0e-6)
            .fold(f32::INFINITY, f32::min);
        if !thinnest.is_finite() {
            return MAX_SUBSTEP;
        }
        let travel_limit = thinnest * MAX_TRAVEL_FRACTION;
        (travel_limit / fastest).clamp(MIN_SUBSTEP, MAX_SUBSTEP)
    }

    /// One fixed-size step.
    fn substep(&mut self, dt: f32) {
        for body in &mut self.bodies {
            body.on_ground = false;
            body.integrate(self.gravity, dt);
        }
        self.detect_contacts();
        self.resolve_contacts();
    }

    /// Fill `contacts` with every overlapping pair.
    ///
    /// O(n^2) over pairs, with the cheap bounding-sphere rejection inside
    /// `collision::collide`. A broadphase grid would be the next step; at the
    /// handful of bodies a 120x68 character scene holds, building one would cost
    /// more than it saves, and pretending otherwise would be premature.
    fn detect_contacts(&mut self) {
        self.contacts.clear();
        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                let a = &self.bodies[i];
                let b = &self.bodies[j];
                // Two immovable bodies can overlap forever without consequence,
                // and resolving them is a no-op that would still cost the test.
                if !a.is_movable() && !b.is_movable() {
                    continue;
                }
                let sa = WorldShape::oriented(a.position, a.shape, Basis::IDENTITY);
                let sb = WorldShape::oriented(b.position, b.shape, Basis::IDENTITY);
                if let Some(contact) = collision::collide(&sa, &sb) {
                    if contact.depth > collision::CONTACT_SLOP {
                        self.contacts.push((BodyId(i), BodyId(j), contact));
                    }
                }
            }
        }
    }

    /// Apply an impulse and a positional correction for every contact.
    fn resolve_contacts(&mut self) {
        // Collected first so the loop is not iterating `self.contacts` while
        // mutating `self.bodies`.
        let contacts = self.contacts.clone();
        for (ia, ib, contact) in contacts {
            self.resolve_one(ia, ib, &contact);
        }
    }

    fn resolve_one(&mut self, ia: BodyId, ib: BodyId, contact: &Contact) {
        let (inv_a, inv_b, restitution, friction) = {
            let a = &self.bodies[ia.0];
            let b = &self.bodies[ib.0];
            (
                if a.is_movable() { a.inverse_mass } else { 0.0 },
                if b.is_movable() { b.inverse_mass } else { 0.0 },
                // The pair is as bouncy as its *less* bouncy half: dropping a
                // ball on sand should not bounce, however elastic the ball is.
                a.material.restitution.min(b.material.restitution),
                // Friction combines the same way, for the same reason.
                a.material.friction.min(b.material.friction),
            )
        };
        let inv_sum = inv_a + inv_b;
        if inv_sum <= 0.0 {
            return;
        }

        // `contact.normal` points from a toward b, so b separating means moving
        // along +normal.
        let n = contact.normal;
        let relative = self.bodies[ib.0].velocity - self.bodies[ia.0].velocity;
        let approach = relative.dot(n);

        // Already separating: an impulse here would pull them back together.
        // Positional correction still runs, because the overlap is real.
        if approach < 0.0 {
            let restitution = if -approach < REST_SPEED {
                // A body settling under gravity approaches at a fraction of a
                // millimetre per step. Bouncing that back is what makes a resting
                // ball buzz on the floor forever.
                0.0
            } else {
                restitution
            };
            let magnitude = -(1.0 + restitution) * approach / inv_sum;
            let impulse = n * magnitude;
            self.bodies[ia.0].apply_impulse(-impulse);
            self.bodies[ib.0].apply_impulse(impulse);

            // Coulomb friction along the tangent, clamped to the normal impulse.
            if friction > 0.0 {
                let relative = self.bodies[ib.0].velocity - self.bodies[ia.0].velocity;
                let tangent_v = relative - n * relative.dot(n);
                let speed = tangent_v.length();
                if speed > 1.0e-5 {
                    let t = tangent_v / speed;
                    // Unclamped, friction could reverse the tangential motion and
                    // add energy; the clamp is what keeps it dissipative.
                    let jt = (-speed / inv_sum).max(-friction * magnitude);
                    let friction_impulse = t * jt;
                    self.bodies[ia.0].apply_impulse(-friction_impulse);
                    self.bodies[ib.0].apply_impulse(friction_impulse);
                }
            }
        }

        // Positional correction: split by inverse mass so the lighter body moves
        // further, and leave `PENETRATION_ALLOWANCE` behind so a resting contact
        // stays a contact.
        let correction = (contact.depth - PENETRATION_ALLOWANCE).max(0.0) * CORRECTION_RATE;
        if correction > 0.0 {
            let shift = n * (correction / inv_sum);
            if self.bodies[ia.0].is_movable() {
                self.bodies[ia.0].position = self.bodies[ia.0].position - shift * inv_a;
            }
            if self.bodies[ib.0].is_movable() {
                self.bodies[ib.0].position = self.bodies[ib.0].position + shift * inv_b;
            }
        }

        // "On the ground" means supported against gravity, which is a question
        // about the contact normal rather than about which body is below.
        let up = -self.gravity;
        if up.length_squared() > 1.0e-12 {
            let up = up.normalize();
            // a is supported if the contact pushes it along up, i.e. against -n.
            if n.dot(up) < -0.5 {
                self.bodies[ia.0].on_ground = true;
            }
            if n.dot(up) > 0.5 {
                self.bodies[ib.0].on_ground = true;
            }
        }
    }

    /// Nearest body along a ray.
    ///
    /// This is the gameplay raycast: click-to-move, line of sight, picking. It
    /// uses `engine::geometry::ray`, the same intersections the CPU tracer uses,
    /// so what the player can click is exactly what the renderer draws.
    pub fn raycast(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<WorldRayHit> {
        let mut nearest = t_max;
        let mut found = None;
        for (index, body) in self.bodies.iter().enumerate() {
            let shape = WorldShape::oriented(body.position, body.shape, Basis::IDENTITY);
            if let Some(hit) = ray::intersect(ray, &shape, t_min, nearest) {
                nearest = hit.t;
                found = Some(WorldRayHit {
                    body: BodyId(index),
                    hit,
                });
            }
        }
        found
    }

    /// Whether anything blocks the straight line between two points.
    ///
    /// Bodies whose collider *contains* an endpoint are ignored. Nudging the
    /// endpoints inward is not enough, and was the first thing tried: a ray
    /// starting inside a sphere still reports the far surface it exits through
    /// (deliberately — a refraction ray depends on that), at a `t` well inside
    /// the segment. So an observer would block every line of sight it asked
    /// about, which is every line of sight in a game.
    pub fn line_of_sight(&self, from: Vec3, to: Vec3) -> bool {
        let delta = to - from;
        let distance = delta.length();
        if distance < 1.0e-5 {
            return true;
        }
        let ray = Ray::new(from, delta);
        let margin = (distance * 0.001).min(0.01);
        let t_max = distance - margin;
        for body in &self.bodies {
            let shape = WorldShape::oriented(body.position, body.shape, Basis::IDENTITY);
            if shape_contains(&shape, from) || shape_contains(&shape, to) {
                continue;
            }
            if ray::intersect(&ray, &shape, margin, t_max).is_some() {
                return false;
            }
        }
        true
    }

    /// Sum of the kinetic energy of every dynamic body.
    ///
    /// Exposed because it is the honest way to test that a solver dissipates
    /// energy rather than manufacturing it: a stack that gains energy will
    /// eventually explode, and that shows up here long before it is visible.
    pub fn kinetic_energy(&self) -> f32 {
        self.bodies
            .iter()
            .filter(|b| b.is_movable())
            .map(|b| 0.5 * b.mass() * b.velocity.length_squared())
            .sum()
    }
}

/// Whether a point is inside (or on) a shape.
///
/// Only volumes can contain a point; a plane is a surface, so it never does. That
/// asymmetry is why this is not a `Shape` method — the answer for a plane is "the
/// question does not apply", and a caller that wanted a half-space test would want
/// the signed distance instead.
fn shape_contains(shape: &WorldShape, point: Vec3) -> bool {
    match shape.shape {
        Shape::Sphere { radius } => (point - shape.origin).length_squared() <= radius * radius,
        Shape::Box { half_extents } => {
            let local = shape.basis.to_local(point - shape.origin);
            local.x.abs() <= half_extents.x
                && local.y.abs() <= half_extents.y
                && local.z.abs() <= half_extents.z
        }
        Shape::Plane { .. } => false,
    }
}

/// Build a world-space ray from a pixel on the ASCII grid.
///
/// This is the other half of gameplay raycasting: the grid is the only thing the
/// player can point at, and it is not the window. Kept here rather than on
/// `Camera` because it is a physics-query concern, and because it needs the grid
/// dimensions the camera knows nothing about.
pub fn ray_through_grid_cell(
    camera: &crate::scene::Camera,
    col: u32,
    row: u32,
    cols: u32,
    rows: u32,
) -> Ray {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let forward = camera.forward();
    let right = camera.right();
    // Re-derived rather than taken from `camera.up`, which is only a hint and
    // need not be perpendicular to the view direction.
    let up = right.cross(forward).normalize();
    // Cell centres, with y flipped: grid row 0 is the top of the screen, which is
    // +y in view space.
    let ndc_x = 2.0 * (col as f32 + 0.5) / cols as f32 - 1.0;
    let ndc_y = 1.0 - 2.0 * (row as f32 + 0.5) / rows as f32;

    match camera.projection {
        crate::scene::Projection::Perspective { fov_y, aspect, .. } => {
            let half_height = (fov_y * 0.5).tan();
            let offset = right * (ndc_x * half_height * aspect) + up * (ndc_y * half_height);
            Ray::new(camera.position, forward + offset)
        }
        crate::scene::Projection::Orthographic {
            left,
            right: r,
            bottom,
            top,
            ..
        } => {
            let half_width = (r - left).abs() * 0.5;
            let half_height = (top - bottom).abs() * 0.5;
            let offset = right * (ndc_x * half_width) + up * (ndc_y * half_height);
            Ray::new(camera.position + offset, forward)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::radians;
    use crate::physics::body::Material;
    use crate::scene::{Camera, Projection};

    fn ball(position: Vec3, radius: f32, mass: f32) -> RigidBody {
        RigidBody::dynamic(position, Shape::Sphere { radius }, mass)
    }

    fn ground_at(y: f32) -> RigidBody {
        RigidBody::immovable(
            Vec3::new(0.0, y, 0.0),
            Shape::Plane {
                normal: Vec3::UNIT_Y,
                half_size: 50.0,
            },
        )
    }

    // --- the phase's completion criterion: bodies must not interpenetrate ---

    /// Two dynamic spheres started overlapping must end up apart, and stay apart.
    /// This is Phase 5.1's stated criterion, so it is asserted on the geometry
    /// rather than on the absence of a panic.
    #[test]
    fn two_overlapping_bodies_separate_and_stay_separated() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::ZERO; // isolate the contact from the fall
        let a = world.add(ball(Vec3::new(-0.4, 0.0, 0.0), 1.0, 1.0));
        let b = world.add(ball(Vec3::new(0.4, 0.0, 0.0), 1.0, 1.0));

        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }

        let pa = world.body(a).unwrap().position;
        let pb = world.body(b).unwrap().position;
        let gap = (pb - pa).length();
        assert!(
            gap >= 2.0 - 1.0e-2,
            "spheres of radius 1 must end up at least 2 apart, got {gap}"
        );
        // And symmetrically: equal masses, equal and opposite displacement.
        assert!(
            (pa.x + pb.x).abs() < 1e-3,
            "equal masses should separate symmetrically: {pa} and {pb}"
        );
    }

    /// A ball dropped on the ground must come to rest *on* it, not sink through
    /// and not hover above it.
    #[test]
    fn a_dropped_ball_rests_on_the_ground() {
        let mut world = PhysicsWorld::new();
        world.add(ground_at(0.0));
        let b = world.add(ball(Vec3::new(0.0, 6.0, 0.0), 1.0, 1.0).with_material(Material::soft()));

        for _ in 0..400 {
            world.step(1.0 / 60.0);
        }

        let y = world.body(b).unwrap().position.y;
        assert!(
            (y - 1.0).abs() < 0.02,
            "a unit ball should rest with its centre at y = 1, got {y}"
        );
        assert!(
            world.body(b).unwrap().velocity.length() < 0.05,
            "it should be at rest, speed = {}",
            world.body(b).unwrap().velocity.length()
        );
        assert!(world.body(b).unwrap().on_ground, "and know it is grounded");
    }

    /// The classic solver failure: a resting body trades a tiny impulse with the
    /// floor every frame and buzzes forever. Measured on kinetic energy, which
    /// catches it long before it is visible.
    #[test]
    fn a_resting_ball_does_not_jitter() {
        let mut world = PhysicsWorld::new();
        world.add(ground_at(0.0));
        world.add(ball(Vec3::new(0.0, 1.0, 0.0), 1.0, 1.0).with_material(Material::soft()));

        // Let it settle.
        for _ in 0..200 {
            world.step(1.0 / 60.0);
        }
        let settled = world.kinetic_energy();
        for _ in 0..200 {
            world.step(1.0 / 60.0);
        }
        let later = world.kinetic_energy();
        assert!(
            settled < 0.01 && later < 0.01,
            "a resting ball should hold almost no kinetic energy: {settled} then {later}"
        );
    }

    /// An explicit-Euler integrator or an unclamped friction term manufactures
    /// energy. A dropped ball must never end up with more than it started with.
    #[test]
    fn the_solver_never_manufactures_energy() {
        let mut world = PhysicsWorld::new();
        world.add(ground_at(0.0));
        let b = world.add(ball(Vec3::new(0.0, 5.0, 0.0), 1.0, 1.0));
        let start_height = 5.0;
        let potential = 1.0 * -world.gravity.y * (start_height - 1.0);

        let mut peak = 0.0f32;
        for _ in 0..1200 {
            world.step(1.0 / 60.0);
            peak = peak.max(world.kinetic_energy());
        }
        assert!(
            peak <= potential * 1.05,
            "peak kinetic energy {peak} exceeded the {potential} available from the drop"
        );
        // And it settles rather than bouncing forever.
        assert!(world.body(b).unwrap().velocity.length() < 0.5);
    }

    #[test]
    fn a_bouncy_ball_bounces_higher_than_a_soft_one() {
        let apex = |material: Material| {
            let mut world = PhysicsWorld::new();
            world.add(ground_at(0.0));
            let b = world.add(ball(Vec3::new(0.0, 4.0, 0.0), 1.0, 1.0).with_material(material));
            let mut hit_floor = false;
            let mut apex = 0.0f32;
            for _ in 0..600 {
                world.step(1.0 / 60.0);
                let body = world.body(b).unwrap();
                if body.position.y < 1.05 {
                    hit_floor = true;
                } else if hit_floor {
                    apex = apex.max(body.position.y);
                }
            }
            apex
        };
        let bouncy = apex(Material::bouncy());
        let soft = apex(Material::soft());
        assert!(
            bouncy > soft + 0.1,
            "bouncy apex {bouncy} should clearly exceed soft apex {soft}"
        );
    }

    /// A light body pushed against a heavy one must move much more than the heavy
    /// one. Splitting the correction evenly instead of by inverse mass is an easy
    /// mistake that this catches.
    #[test]
    fn positional_correction_splits_by_inverse_mass() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::ZERO;
        let light = world.add(ball(Vec3::new(-0.5, 0.0, 0.0), 1.0, 1.0));
        let heavy = world.add(ball(Vec3::new(0.5, 0.0, 0.0), 1.0, 100.0));
        let (l0, h0) = (
            world.body(light).unwrap().position,
            world.body(heavy).unwrap().position,
        );
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        let dl = (world.body(light).unwrap().position - l0).length();
        let dh = (world.body(heavy).unwrap().position - h0).length();
        assert!(
            dl > dh * 10.0,
            "the light body should absorb the separation: light moved {dl}, heavy {dh}"
        );
    }

    #[test]
    fn immovable_bodies_never_move_however_hard_they_are_pushed() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::ZERO;
        let wall = world.add(RigidBody::immovable(
            Vec3::ZERO,
            Shape::Box {
                half_extents: Vec3::new(1.0, 5.0, 5.0),
            },
        ));
        world.add(ball(Vec3::new(1.5, 0.0, 0.0), 1.0, 50.0).with_velocity(Vec3::new(-50.0, 0.0, 0.0)));
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        assert_eq!(world.body(wall).unwrap().position, Vec3::ZERO);
        assert_eq!(world.body(wall).unwrap().velocity, Vec3::ZERO);
    }

    /// Two static bodies must not even be tested: resolving them is a no-op, and
    /// reporting the contact would inflate the contact count with pairs nothing
    /// can be done about.
    #[test]
    fn two_static_bodies_produce_no_contact() {
        let mut world = PhysicsWorld::new();
        world.add(RigidBody::immovable(Vec3::ZERO, Shape::Sphere { radius: 1.0 }));
        world.add(RigidBody::immovable(
            Vec3::new(0.5, 0.0, 0.0),
            Shape::Sphere { radius: 1.0 },
        ));
        world.step(1.0 / 60.0);
        assert!(world.contacts().is_empty());
    }

    // --- substepping ---

    /// A fast body must not pass through a thin wall. This is what substepping is
    /// for: at 1/60 s a body at 300 m/s moves 5 m per step, which is wider than
    /// most walls.
    #[test]
    fn a_fast_body_does_not_tunnel_through_a_wall() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::ZERO;
        world.add(RigidBody::immovable(
            Vec3::ZERO,
            Shape::Box {
                half_extents: Vec3::new(0.5, 10.0, 10.0),
            },
        ));
        let bullet = world.add(
            ball(Vec3::new(-8.0, 0.0, 0.0), 0.5, 1.0).with_velocity(Vec3::new(120.0, 0.0, 0.0)),
        );
        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }
        let x = world.body(bullet).unwrap().position.x;
        assert!(
            x < 0.5,
            "the body ended up at x = {x}, on the far side of the wall"
        );
    }

    #[test]
    fn a_long_delta_is_split_into_substeps_and_capped() {
        let mut world = PhysicsWorld::new();
        world.add(ball(Vec3::ZERO, 1.0, 1.0));
        world.step(1.0 / 60.0);
        assert_eq!(
            world.substeps(),
            2,
            "a slow body gets the full 1/120 s substep, so 1/60 s is two of them"
        );
        // A one-second stall must not become hundreds of substeps.
        world.step(1.0);
        assert_eq!(world.substeps(), MAX_SUBSTEPS);
    }

    /// The substep must shrink with speed, not stay fixed. This is the mechanism
    /// the anti-tunnelling test relies on, so it is worth pinning directly: a
    /// fixed substep would report the same figure for both.
    #[test]
    fn the_substep_shrinks_as_bodies_speed_up() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::ZERO;
        world.add(RigidBody::immovable(
            Vec3::new(50.0, 0.0, 0.0),
            Shape::Box {
                half_extents: Vec3::new(0.5, 10.0, 10.0),
            },
        ));
        let slow = world.safe_substep();

        world.add(ball(Vec3::ZERO, 0.5, 1.0).with_velocity(Vec3::new(200.0, 0.0, 0.0)));
        let fast = world.safe_substep();
        assert!(
            fast < slow,
            "a 200 m/s body must shorten the substep: {fast} vs {slow}"
        );
        // And it must never collapse to zero, which would hang the frame.
        assert!(fast >= MIN_SUBSTEP, "substep {fast} fell below the floor");
    }

    /// A plane has no thickness, so counting it as the "thinnest collider" would
    /// peg every substep at the floor and make every frame 2000 substeps long — a
    /// performance cliff rather than a wrong answer, and therefore exactly the
    /// kind of thing that ships unnoticed.
    #[test]
    fn a_plane_does_not_collapse_the_substep() {
        let mut world = PhysicsWorld::new();
        world.add(ground_at(0.0));
        world.add(ball(Vec3::new(0.0, 5.0, 0.0), 1.0, 1.0).with_velocity(Vec3::new(50.0, 0.0, 0.0)));
        assert!(
            world.safe_substep() > MIN_SUBSTEP * 4.0,
            "substep {} suggests the plane was counted as a thin collider",
            world.safe_substep()
        );
    }

    #[test]
    fn a_nonsensical_delta_is_ignored_rather_than_propagated() {
        let mut world = PhysicsWorld::new();
        let b = world.add(ball(Vec3::new(0.0, 5.0, 0.0), 1.0, 1.0));
        for dt in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            world.step(dt);
        }
        let p = world.body(b).unwrap().position;
        assert_eq!(p, Vec3::new(0.0, 5.0, 0.0));
        assert!(p.y.is_finite());
        assert_eq!(world.substeps(), 0);
    }

    // --- raycasting ---

    /// The phase criterion: a raycast from the camera must hit a specific body.
    #[test]
    fn a_raycast_from_the_camera_hits_the_expected_body() {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::ZERO;
        let near = world.add(ball(Vec3::new(0.0, 0.0, -3.0), 1.0, 1.0));
        let far = world.add(ball(Vec3::new(0.0, 0.0, -10.0), 1.0, 1.0));
        let aside = world.add(ball(Vec3::new(8.0, 0.0, -3.0), 1.0, 1.0));

        let camera = Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 200.0),
        );
        // Dead centre of a 120x68 grid.
        let ray = ray_through_grid_cell(&camera, 60, 34, 120, 68);
        let hit = world.raycast(&ray, 0.001, 1000.0).expect("should hit something");
        assert_eq!(hit.body, near, "the nearest body must win, not the far one");
        assert_ne!(hit.body, far);
        assert_ne!(hit.body, aside);
        assert!((hit.hit.t - 2.0).abs() < 0.05, "t = {}", hit.hit.t);
    }

    /// A cell in the upper half of the grid must cast upward. A y-flip here sends
    /// every click to the mirror image of where the player aimed.
    #[test]
    fn grid_rays_are_not_vertically_flipped() {
        let camera = Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 200.0),
        );
        let top = ray_through_grid_cell(&camera, 60, 2, 120, 68);
        let bottom = ray_through_grid_cell(&camera, 60, 65, 120, 68);
        assert!(top.direction().y > 0.1, "top of the grid should aim up: {}", top.direction());
        assert!(
            bottom.direction().y < -0.1,
            "bottom should aim down: {}",
            bottom.direction()
        );
    }

    #[test]
    fn grid_rays_are_not_horizontally_flipped() {
        let camera = Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 200.0),
        );
        // Looking down -Z with +Y up, the camera's right is +X.
        let right_ray = ray_through_grid_cell(&camera, 115, 34, 120, 68);
        assert!(
            right_ray.direction().dot(camera.right()) > 0.1,
            "the right of the grid should aim along the camera's right"
        );
    }

    #[test]
    fn a_raycast_into_empty_space_finds_nothing() {
        let mut world = PhysicsWorld::new();
        world.add(ball(Vec3::new(0.0, 0.0, -3.0), 1.0, 1.0));
        let ray = Ray::new(Vec3::ZERO, Vec3::UNIT_Y);
        assert!(world.raycast(&ray, 0.001, 1000.0).is_none());
    }

    #[test]
    fn t_max_bounds_the_raycast() {
        let mut world = PhysicsWorld::new();
        world.add(ball(Vec3::new(0.0, 0.0, -50.0), 1.0, 1.0));
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        assert!(world.raycast(&ray, 0.001, 10.0).is_none());
        assert!(world.raycast(&ray, 0.001, 1000.0).is_some());
    }

    // --- line of sight ---

    #[test]
    fn line_of_sight_is_blocked_by_a_body_between_the_endpoints() {
        let mut world = PhysicsWorld::new();
        world.add(RigidBody::immovable(
            Vec3::ZERO,
            Shape::Box {
                half_extents: Vec3::new(0.5, 5.0, 5.0),
            },
        ));
        assert!(
            !world.line_of_sight(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0)),
            "the wall is in the way"
        );
        assert!(
            world.line_of_sight(Vec3::new(-5.0, 8.0, 0.0), Vec3::new(5.0, 8.0, 0.0)),
            "above the wall the line is clear"
        );
    }

    /// A query from inside a body's own collider must not report itself as an
    /// obstruction — which is every line-of-sight query an entity makes about
    /// itself.
    #[test]
    fn line_of_sight_ignores_a_body_exactly_at_an_endpoint() {
        let mut world = PhysicsWorld::new();
        // The observer's own body, centred at the query origin.
        world.add(RigidBody::immovable(Vec3::ZERO, Shape::Sphere { radius: 1.0 }));
        assert!(
            world.line_of_sight(Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0)),
            "a body at the origin should not block the line from that origin"
        );
    }

    #[test]
    fn a_degenerate_line_of_sight_is_clear() {
        let world = PhysicsWorld::new();
        assert!(world.line_of_sight(Vec3::ONE, Vec3::ONE));
    }

    // --- bookkeeping ---

    #[test]
    fn body_ids_are_stable_and_index_the_body_list() {
        let mut world = PhysicsWorld::new();
        let a = world.add(ball(Vec3::ZERO, 1.0, 1.0));
        let b = world.add(ball(Vec3::new(5.0, 0.0, 0.0), 1.0, 1.0));
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        assert_eq!(world.len(), 2);
        assert!(!world.is_empty());
        // Adding more must not disturb existing handles.
        world.add(ball(Vec3::new(10.0, 0.0, 0.0), 1.0, 1.0));
        assert!((world.body(a).unwrap().position - Vec3::ZERO).length() < 1e-6);
    }

    #[test]
    fn an_empty_world_steps_without_incident() {
        let mut world = PhysicsWorld::new();
        world.step(1.0 / 60.0);
        assert!(world.contacts().is_empty());
        assert_eq!(world.kinetic_energy(), 0.0);
    }

    #[test]
    fn on_ground_is_derived_from_gravity_not_from_the_y_axis() {
        // Gravity along +X: "ground" is a wall on the +X side.
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::new(9.81, 0.0, 0.0);
        world.add(RigidBody::immovable(
            Vec3::new(5.0, 0.0, 0.0),
            Shape::Box {
                half_extents: Vec3::new(1.0, 10.0, 10.0),
            },
        ));
        let b = world.add(ball(Vec3::ZERO, 1.0, 1.0));
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        assert!(
            world.body(b).unwrap().on_ground,
            "a body pressed against a +X wall by +X gravity is grounded"
        );
    }
}
