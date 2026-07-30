// Rigid body state and the integrator that advances it.
//
// Linear dynamics only: position, velocity, mass. No angular velocity, no
// inertia tensor, no torque. That is a deliberate scope choice, not an oversight
// — the engine renders to an 120x68 character grid, where a rolling sphere and a
// sliding one are the same handful of glyphs, and an inertia tensor would double
// the solver's complexity for detail the output cannot show. The gap is recorded
// in the phase notes rather than papered over.

use crate::engine::geometry::Shape;
use crate::engine::math::Vec3;

/// Default gravity: roughly Earth's, pointing down.
pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Below this speed a body resting on a surface is snapped to a stop.
///
/// Without it, gravity and the contact solver trade a fraction of a millimetre
/// back and forth forever, and the body visibly buzzes. The threshold is on the
/// velocity *along the contact normal*, so a body sliding sideways keeps sliding.
pub const REST_SPEED: f32 = 0.05;

/// How much of the approach speed is returned on impact, per body. The pair's
/// restitution is the smaller of the two, so one inelastic body is enough to make
/// a collision inelastic — which matches the intuition that dropping a ball on
/// sand does not bounce.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// 0 = perfectly inelastic, 1 = no energy lost.
    pub restitution: f32,
    /// Tangential friction coefficient, 0 = frictionless.
    pub friction: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            restitution: 0.25,
            friction: 0.4,
        }
    }
}

impl Material {
    /// A bouncy material.
    pub const fn bouncy() -> Self {
        Self {
            restitution: 0.8,
            friction: 0.2,
        }
    }

    /// A material that absorbs impacts.
    pub const fn soft() -> Self {
        Self {
            restitution: 0.0,
            friction: 0.8,
        }
    }
}

/// A rigid body: a shape with mass, a position and a velocity.
#[derive(Clone, Copy, Debug)]
pub struct RigidBody {
    /// Centre of mass in world space. For a plane body this is a point on it.
    pub position: Vec3,
    /// Linear velocity.
    pub velocity: Vec3,
    /// Collision shape, in the body's local space.
    pub shape: Shape,
    /// Surface response properties.
    pub material: Material,
    /// Reciprocal mass. Zero means infinite mass, i.e. immovable — stored as the
    /// reciprocal because that is the form every impulse calculation needs, and
    /// because "infinite mass" is then the representable value 0 rather than a
    /// special case guarded at each use.
    pub inverse_mass: f32,
    /// Whether gravity and integration apply. A static body still collides.
    pub is_static: bool,
    /// Set by the solver when the body is resting on something, so a caller can
    /// tell "on the ground" from "falling".
    pub on_ground: bool,
}

impl RigidBody {
    /// A dynamic body of the given mass. A non-positive mass is treated as
    /// static: it is almost certainly a mistake, and the alternative
    /// (dividing by it) produces infinities that spread through the whole
    /// simulation before anything visible goes wrong.
    pub fn dynamic(position: Vec3, shape: Shape, mass: f32) -> Self {
        let (inverse_mass, is_static) = if mass > 0.0 && mass.is_finite() {
            (1.0 / mass, false)
        } else {
            (0.0, true)
        };
        Self {
            position,
            velocity: Vec3::ZERO,
            shape,
            material: Material::default(),
            inverse_mass,
            is_static,
            on_ground: false,
        }
    }

    /// An immovable body: collides, never moves, ignores gravity.
    pub fn immovable(position: Vec3, shape: Shape) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            shape,
            material: Material::default(),
            inverse_mass: 0.0,
            is_static: true,
            on_ground: true,
        }
    }

    /// Replace the surface material, chaining from a constructor.
    pub fn with_material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    /// Give the body an initial velocity.
    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }

    /// Mass in kilograms, or `f32::INFINITY` for a static body.
    pub fn mass(&self) -> f32 {
        if self.inverse_mass > 0.0 {
            1.0 / self.inverse_mass
        } else {
            f32::INFINITY
        }
    }

    /// Whether the solver may move this body.
    pub fn is_movable(&self) -> bool {
        !self.is_static && self.inverse_mass > 0.0
    }

    /// Advance position by velocity, and velocity by acceleration.
    ///
    /// Semi-implicit Euler: velocity is updated *before* position, so the
    /// position step uses the new velocity. It costs the same as explicit Euler
    /// and is stable under the repeated small impulses a contact solver applies,
    /// where explicit Euler gains energy and eventually launches resting bodies.
    pub fn integrate(&mut self, gravity: Vec3, dt: f32) {
        if !self.is_movable() || dt <= 0.0 {
            return;
        }
        self.velocity = self.velocity + gravity * dt;
        self.position = self.position + self.velocity * dt;
    }

    /// Apply an instantaneous change in momentum.
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if !self.is_movable() {
            return;
        }
        self.velocity = self.velocity + impulse * self.inverse_mass;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sphere(position: Vec3, mass: f32) -> RigidBody {
        RigidBody::dynamic(position, Shape::Sphere { radius: 1.0 }, mass)
    }

    #[test]
    fn dynamic_body_stores_the_reciprocal_of_its_mass() {
        let b = unit_sphere(Vec3::ZERO, 4.0);
        assert!((b.inverse_mass - 0.25).abs() < 1e-6);
        assert!((b.mass() - 4.0).abs() < 1e-6);
        assert!(b.is_movable());
    }

    /// A zero or negative mass is a caller error. Dividing by it would spread
    /// infinities through the whole simulation before anything visibly broke, so
    /// it degrades to static instead.
    #[test]
    fn non_positive_mass_degrades_to_static_rather_than_dividing_by_zero() {
        for mass in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let b = unit_sphere(Vec3::ZERO, mass);
            assert!(!b.is_movable(), "mass {mass} should have produced a static body");
            assert_eq!(b.inverse_mass, 0.0);
        }
    }

    #[test]
    fn immovable_body_never_moves_under_gravity() {
        let mut b = RigidBody::immovable(Vec3::ZERO, Shape::Sphere { radius: 1.0 });
        for _ in 0..100 {
            b.integrate(DEFAULT_GRAVITY, 1.0 / 60.0);
        }
        assert_eq!(b.position, Vec3::ZERO);
        assert_eq!(b.velocity, Vec3::ZERO);
        assert!(b.mass().is_infinite());
    }

    #[test]
    fn immovable_body_ignores_impulses() {
        let mut b = RigidBody::immovable(Vec3::ZERO, Shape::Sphere { radius: 1.0 });
        b.apply_impulse(Vec3::new(1000.0, 0.0, 0.0));
        assert_eq!(b.velocity, Vec3::ZERO);
    }

    #[test]
    fn free_fall_matches_the_analytic_solution_closely() {
        let mut b = unit_sphere(Vec3::new(0.0, 100.0, 0.0), 1.0);
        let dt = 1.0 / 240.0;
        let steps = 240; // one second
        for _ in 0..steps {
            b.integrate(DEFAULT_GRAVITY, dt);
        }
        // v = g*t exactly, for any Euler variant.
        assert!(
            (b.velocity.y - DEFAULT_GRAVITY.y).abs() < 1e-3,
            "velocity after 1s = {}",
            b.velocity.y
        );
        // s = 0.5*g*t^2 = -4.905. Semi-implicit Euler overshoots by g*dt*t/2,
        // which at dt = 1/240 is about 2cm — close enough that a wrong sign or a
        // missing dt would stand out immediately.
        let drop = 100.0 - b.position.y;
        assert!(
            (drop - 4.905).abs() < 0.05,
            "fell {drop} m in 1 s, expected ~4.905"
        );
    }

    /// Velocity must be updated before position. Explicit Euler (position first)
    /// gains energy under the repeated impulses a contact solver applies, and
    /// eventually launches bodies that should be at rest.
    #[test]
    fn integration_is_semi_implicit() {
        let mut b = unit_sphere(Vec3::ZERO, 1.0);
        let dt = 0.5;
        b.integrate(Vec3::new(0.0, -10.0, 0.0), dt);
        // Semi-implicit: v = -5, then y += v*dt = -2.5.
        // Explicit would leave y at 0 after the first step.
        assert!((b.velocity.y + 5.0).abs() < 1e-6, "v = {}", b.velocity.y);
        assert!(
            (b.position.y + 2.5).abs() < 1e-6,
            "position must use the NEW velocity: y = {}",
            b.position.y
        );
    }

    #[test]
    fn zero_or_negative_dt_is_a_no_op() {
        let mut b = unit_sphere(Vec3::new(0.0, 5.0, 0.0), 1.0);
        b.integrate(DEFAULT_GRAVITY, 0.0);
        assert_eq!(b.position.y, 5.0);
        b.integrate(DEFAULT_GRAVITY, -1.0);
        assert_eq!(b.position.y, 5.0);
        assert_eq!(b.velocity, Vec3::ZERO);
    }

    #[test]
    fn impulse_changes_velocity_in_inverse_proportion_to_mass() {
        let mut light = unit_sphere(Vec3::ZERO, 1.0);
        let mut heavy = unit_sphere(Vec3::ZERO, 10.0);
        let impulse = Vec3::new(10.0, 0.0, 0.0);
        light.apply_impulse(impulse);
        heavy.apply_impulse(impulse);
        assert!((light.velocity.x - 10.0).abs() < 1e-5);
        assert!((heavy.velocity.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn material_presets_are_ordered_as_their_names_claim() {
        assert!(Material::bouncy().restitution > Material::default().restitution);
        assert!(Material::default().restitution > Material::soft().restitution);
        assert!(Material::soft().friction > Material::bouncy().friction);
    }

    #[test]
    fn builders_chain_without_losing_earlier_settings() {
        let b = unit_sphere(Vec3::ZERO, 2.0)
            .with_material(Material::bouncy())
            .with_velocity(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(b.material, Material::bouncy());
        assert_eq!(b.velocity, Vec3::new(1.0, 2.0, 3.0));
        assert!((b.mass() - 2.0).abs() < 1e-6, "mass must survive the builders");
    }
}
