// Ray/shape intersection maths, shared by the CPU fallback tracer (Phase 4.4)
// and gameplay raycasting (Phase 5.1).
//
// Deliberately one module rather than one per consumer. A raycast used for
// click-to-move that disagreed with the raycast used for reflections would put
// the player somewhere other than where the mirror shows them, and that class of
// bug is invisible in both features' own tests.

use crate::engine::geometry::shapes::{Basis, Shape, WorldShape};
use crate::engine::math::Vec3;

/// A ray with a unit direction.
///
/// The direction is normalized at construction so `t` is always a distance in
/// world units. Callers that pass an unnormalized direction otherwise get a `t`
/// scaled by its length, which silently breaks every distance comparison
/// downstream — shadow ray limits most of all.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    direction: Vec3,
}

impl Ray {
    /// Build a ray, normalizing the direction. A zero direction degenerates to
    /// +Z rather than to NaN, so a malformed ray misses instead of poisoning
    /// every arithmetic result that touches it.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        let len_sq = direction.length_squared();
        let direction = if len_sq > 1e-20 {
            direction / len_sq.sqrt()
        } else {
            Vec3::UNIT_Z
        };
        Self { origin, direction }
    }

    /// The unit direction.
    pub fn direction(&self) -> Vec3 {
        self.direction
    }

    /// The point at distance `t` along the ray.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// A ray hit: how far along, where, and which way the surface faces.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// Distance along the ray.
    pub t: f32,
    /// World-space hit point.
    pub point: Vec3,
    /// Outward unit surface normal at the hit point. Always points *against* the
    /// incoming ray, so shading and offsetting work the same whether the surface
    /// was entered from outside or from within.
    pub normal: Vec3,
    /// Whether the ray struck the outside of the surface. A refraction ray
    /// leaving a sphere needs this to know which way the index ratio goes.
    pub front_face: bool,
}

/// Nearest intersection of `ray` with `shape` within `(t_min, t_max)`.
pub fn intersect(ray: &Ray, shape: &WorldShape, t_min: f32, t_max: f32) -> Option<RayHit> {
    match shape.shape {
        Shape::Sphere { radius } => intersect_sphere(ray, shape.origin, radius, t_min, t_max),
        Shape::Plane { normal, half_size } => {
            intersect_plane(ray, shape.origin, normal, half_size, t_min, t_max)
        }
        Shape::Box { half_extents } => {
            intersect_obb(ray, shape.origin, half_extents, &shape.basis, t_min, t_max)
        }
    }
}

/// Ray/sphere intersection.
///
/// Solves |o + td - c|^2 = r^2 as a quadratic in t. The `b/2` form is used
/// (`half_b`) so the discriminant is `half_b^2 - a*c` with no factor of four to
/// lose track of.
pub fn intersect_sphere(
    ray: &Ray,
    center: Vec3,
    radius: f32,
    t_min: f32,
    t_max: f32,
) -> Option<RayHit> {
    let oc = ray.origin - center;
    let a = ray.direction.length_squared();
    let half_b = oc.dot(ray.direction);
    let c = oc.length_squared() - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_d = discriminant.sqrt();

    // Near root first; fall back to the far one so a ray starting inside the
    // sphere still reports the surface it exits through, which is what a
    // refraction ray inside glass depends on.
    let mut t = (-half_b - sqrt_d) / a;
    if t < t_min || t > t_max {
        t = (-half_b + sqrt_d) / a;
        if t < t_min || t > t_max {
            return None;
        }
    }

    let point = ray.at(t);
    let outward = (point - center) / radius.max(1e-6);
    Some(oriented_hit(ray, t, point, outward))
}

/// Ray/plane intersection, bounded to a square patch of side `2 * half_size`
/// centred on `origin`.
pub fn intersect_plane(
    ray: &Ray,
    origin: Vec3,
    normal: Vec3,
    half_size: f32,
    t_min: f32,
    t_max: f32,
) -> Option<RayHit> {
    let n = normal.normalize();
    let denom = n.dot(ray.direction);
    // Parallel (or near enough that the division would explode): no hit. A ray
    // exactly in the plane technically intersects everywhere, which is not a
    // useful answer for either shading or collision.
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (origin - ray.origin).dot(n) / denom;
    if t < t_min || t > t_max {
        return None;
    }
    let point = ray.at(t);
    // Bound the patch: measure the offset from the centre along two in-plane
    // axes and reject anything outside the square.
    if half_size.is_finite() && half_size > 0.0 {
        let (u, v) = plane_basis(n);
        let d = point - origin;
        if d.dot(u).abs() > half_size || d.dot(v).abs() > half_size {
            return None;
        }
    }
    Some(oriented_hit(ray, t, point, n))
}

/// Ray/axis-aligned-box intersection, by the slab method.
pub fn intersect_box(
    ray: &Ray,
    center: Vec3,
    half_extents: Vec3,
    t_min: f32,
    t_max: f32,
) -> Option<RayHit> {
    let o = ray.origin - center;
    let d = ray.direction;
    let mut near = t_min;
    let mut far = t_max;
    // Which axis produced `near`, so the normal can be recovered without
    // re-deriving it from the hit point (which is ambiguous exactly on an edge).
    let mut axis = 0usize;

    let o = [o.x, o.y, o.z];
    let d = [d.x, d.y, d.z];
    let h = [half_extents.x, half_extents.y, half_extents.z];

    for i in 0..3 {
        if d[i].abs() < 1e-9 {
            // Parallel to this slab: either always inside it or never.
            if o[i].abs() > h[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d[i];
        let mut t0 = (-h[i] - o[i]) * inv;
        let mut t1 = (h[i] - o[i]) * inv;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        if t0 > near {
            near = t0;
            axis = i;
        }
        far = far.min(t1);
        if far < near {
            return None;
        }
    }

    let t = near;
    if t < t_min || t > t_max {
        return None;
    }
    let point = ray.at(t);
    let mut outward = Vec3::ZERO;
    let sign = if (point - center).dot(axis_vector(axis)) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    outward = outward + axis_vector(axis) * sign;
    Some(oriented_hit(ray, t, point, outward))
}

/// Ray/oriented-box intersection.
///
/// The ray is rotated into the box's local frame, where the box *is* axis
/// aligned, and the resulting hit is rotated back out. Doing it this way means
/// there is one slab implementation rather than two that can disagree — the
/// axis-aligned case is just this one with an identity basis, and it short-cuts
/// to `intersect_box` so the common case pays nothing for the generality.
pub fn intersect_obb(
    ray: &Ray,
    center: Vec3,
    half_extents: Vec3,
    basis: &Basis,
    t_min: f32,
    t_max: f32,
) -> Option<RayHit> {
    if basis.is_identity() {
        return intersect_box(ray, center, half_extents, t_min, t_max);
    }
    // Local frame: the box sits at the origin, axis aligned.
    let local_origin = basis.to_local(ray.origin - center);
    let local_dir = basis.to_local(ray.direction());
    let local_ray = Ray::new(local_origin, local_dir);
    // `t` is preserved by a rotation, so the distance needs no correction.
    let hit = intersect_box(&local_ray, Vec3::ZERO, half_extents, t_min, t_max)?;
    Some(RayHit {
        t: hit.t,
        point: center + basis.to_world(hit.point),
        normal: basis.to_world(hit.normal),
        front_face: hit.front_face,
    })
}

/// Ray/triangle intersection (Möller–Trumbore).
///
/// Not used by the analytic tracer, which has no triangles, but gameplay
/// raycasting against actual mesh geometry needs it and it belongs with the rest
/// of the intersection maths rather than in whichever module reaches for it
/// first.
pub fn intersect_triangle(
    ray: &Ray,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    t_min: f32,
    t_max: f32,
) -> Option<RayHit> {
    let edge1 = b - a;
    let edge2 = c - a;
    let h = ray.direction.cross(edge2);
    let det = edge1.dot(h);
    if det.abs() < 1e-9 {
        // Ray lies in the triangle's plane.
        return None;
    }
    let inv_det = 1.0 / det;
    let s = ray.origin - a;
    let u = s.dot(h) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(edge1);
    let v = ray.direction.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(q) * inv_det;
    if t < t_min || t > t_max {
        return None;
    }
    let point = ray.at(t);
    // Geometric normal from the winding, not from a vertex attribute: the
    // attribute can disagree with the geometry, and back-face decisions must
    // follow what the triangle actually is.
    let outward = edge1.cross(edge2).normalize();
    Some(oriented_hit(ray, t, point, outward))
}

/// Build a hit with the normal turned to face the incoming ray.
fn oriented_hit(ray: &Ray, t: f32, point: Vec3, outward: Vec3) -> RayHit {
    let front_face = outward.dot(ray.direction) < 0.0;
    RayHit {
        t,
        point,
        normal: if front_face { outward } else { -outward },
        front_face,
    }
}

/// Two orthonormal in-plane axes for a unit normal.
fn plane_basis(n: Vec3) -> (Vec3, Vec3) {
    // Pick the reference axis least aligned with n, so the cross product does
    // not collapse.
    let reference = if n.y.abs() < 0.9 {
        Vec3::UNIT_Y
    } else {
        Vec3::UNIT_X
    };
    let u = n.cross(reference).normalize();
    let v = n.cross(u);
    (u, v)
}

fn axis_vector(axis: usize) -> Vec3 {
    match axis {
        0 => Vec3::UNIT_X,
        1 => Vec3::UNIT_Y,
        _ => Vec3::UNIT_Z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T_MAX: f32 = 1.0e6;

    #[test]
    fn ray_normalizes_its_direction() {
        let r = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -5.0));
        assert!((r.direction().length() - 1.0).abs() < 1e-6);
        // t must therefore be a real distance.
        assert!((r.at(3.0) - Vec3::new(0.0, 0.0, -3.0)).length() < 1e-5);
    }

    #[test]
    fn zero_direction_degenerates_instead_of_producing_nan() {
        let r = Ray::new(Vec3::ONE, Vec3::ZERO);
        assert!(r.direction().x.is_finite() && r.direction().length() > 0.5);
    }

    #[test]
    fn sphere_hit_from_outside_reports_the_near_surface() {
        let sphere = WorldShape::new(Vec3::new(0.0, 0.0, -5.0), Shape::Sphere { radius: 1.0 });
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let hit = intersect(&ray, &sphere, 0.001, T_MAX).expect("should hit");
        assert!((hit.t - 4.0).abs() < 1e-4, "t = {}", hit.t);
        assert!(hit.front_face);
        // Facing the camera, i.e. +Z.
        assert!((hit.normal - Vec3::UNIT_Z).length() < 1e-4, "{}", hit.normal);
    }

    #[test]
    fn sphere_miss_returns_none() {
        let sphere = WorldShape::new(Vec3::new(0.0, 0.0, -5.0), Shape::Sphere { radius: 1.0 });
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(intersect(&ray, &sphere, 0.001, T_MAX).is_none());
    }

    /// A refraction ray inside glass has to find the far surface, and the normal
    /// it reports must face the ray — otherwise the index ratio inverts the wrong
    /// way and the glass turns into a mirror.
    #[test]
    fn ray_starting_inside_a_sphere_exits_through_the_far_surface() {
        let sphere = WorldShape::new(Vec3::ZERO, Shape::Sphere { radius: 2.0 });
        let ray = Ray::new(Vec3::ZERO, Vec3::UNIT_X);
        let hit = intersect(&ray, &sphere, 0.001, T_MAX).expect("should exit");
        assert!((hit.t - 2.0).abs() < 1e-4, "t = {}", hit.t);
        assert!(!hit.front_face, "exiting counts as a back face");
        assert!(
            (hit.normal - (-Vec3::UNIT_X)).length() < 1e-4,
            "normal must face the ray: {}",
            hit.normal
        );
    }

    #[test]
    fn t_min_rejects_a_hit_the_ray_already_passed() {
        let sphere = WorldShape::new(Vec3::new(0.0, 0.0, -5.0), Shape::Sphere { radius: 1.0 });
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        assert!(intersect(&ray, &sphere, 10.0, T_MAX).is_none());
        assert!(intersect(&ray, &sphere, 0.001, 3.0).is_none());
    }

    #[test]
    fn plane_hit_reports_the_normal_and_distance() {
        let plane = WorldShape::new(
            Vec3::new(0.0, -2.0, 0.0),
            Shape::Plane {
                normal: Vec3::UNIT_Y,
                half_size: 25.0,
            },
        );
        let ray = Ray::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = intersect(&ray, &plane, 0.001, T_MAX).expect("should hit the ground");
        assert!((hit.t - 5.0).abs() < 1e-4, "t = {}", hit.t);
        assert!((hit.normal - Vec3::UNIT_Y).length() < 1e-4);
        assert!(hit.front_face);
    }

    /// The engine's ground is a 50-unit plane, not an infinite one. A tracer that
    /// treated it as infinite would put a horizon in every reflection where the
    /// real scene has an edge.
    #[test]
    fn plane_is_bounded_by_its_half_size() {
        let plane = WorldShape::new(
            Vec3::ZERO,
            Shape::Plane {
                normal: Vec3::UNIT_Y,
                half_size: 5.0,
            },
        );
        let inside = Ray::new(Vec3::new(4.0, 3.0, 0.0), -Vec3::UNIT_Y);
        assert!(intersect(&inside, &plane, 0.001, T_MAX).is_some());
        let outside = Ray::new(Vec3::new(6.0, 3.0, 0.0), -Vec3::UNIT_Y);
        assert!(
            intersect(&outside, &plane, 0.001, T_MAX).is_none(),
            "a hit 6 units out on a half-size-5 patch means the bound is not applied"
        );
    }

    #[test]
    fn plane_parallel_to_the_ray_is_a_miss() {
        let plane = WorldShape::new(
            Vec3::ZERO,
            Shape::Plane {
                normal: Vec3::UNIT_Y,
                half_size: 10.0,
            },
        );
        let ray = Ray::new(Vec3::new(0.0, 1.0, 0.0), Vec3::UNIT_X);
        assert!(intersect(&ray, &plane, 0.001, T_MAX).is_none());
    }

    #[test]
    fn box_hit_reports_the_face_normal() {
        let b = WorldShape::new(
            Vec3::ZERO,
            Shape::Box {
                half_extents: Vec3::ONE,
            },
        );
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), -Vec3::UNIT_Z);
        let hit = intersect(&ray, &b, 0.001, T_MAX).expect("should hit");
        assert!((hit.t - 4.0).abs() < 1e-4, "t = {}", hit.t);
        assert!(
            (hit.normal - Vec3::UNIT_Z).length() < 1e-4,
            "normal = {}",
            hit.normal
        );
    }

    #[test]
    fn box_miss_beside_the_slab_returns_none() {
        let b = WorldShape::new(
            Vec3::ZERO,
            Shape::Box {
                half_extents: Vec3::ONE,
            },
        );
        let ray = Ray::new(Vec3::new(3.0, 0.0, 5.0), -Vec3::UNIT_Z);
        assert!(intersect(&ray, &b, 0.001, T_MAX).is_none());
    }

    #[test]
    fn box_hit_from_each_axis_picks_that_axis_normal() {
        let b = WorldShape::new(
            Vec3::ZERO,
            Shape::Box {
                half_extents: Vec3::ONE,
            },
        );
        for (from, dir, want) in [
            (Vec3::new(5.0, 0.0, 0.0), -Vec3::UNIT_X, Vec3::UNIT_X),
            (Vec3::new(-5.0, 0.0, 0.0), Vec3::UNIT_X, -Vec3::UNIT_X),
            (Vec3::new(0.0, 5.0, 0.0), -Vec3::UNIT_Y, Vec3::UNIT_Y),
            (Vec3::new(0.0, -5.0, 0.0), Vec3::UNIT_Y, -Vec3::UNIT_Y),
        ] {
            let hit = intersect(&Ray::new(from, dir), &b, 0.001, T_MAX)
                .unwrap_or_else(|| panic!("no hit from {from}"));
            assert!(
                (hit.normal - want).length() < 1e-4,
                "from {from}: wanted {want}, got {}",
                hit.normal
            );
        }
    }

    /// A rotated box must be hit where it actually is. Before `Basis` existed the
    /// rotation was silently dropped, which is invisible in an axis-aligned test.
    #[test]
    fn rotated_box_is_hit_on_its_rotated_face() {
        let basis = Basis::from_matrix(&crate::engine::math::Mat4::rotation_y(
            std::f32::consts::FRAC_PI_4,
        ));
        let long_box = Shape::Box {
            half_extents: Vec3::new(3.0, 0.5, 0.5),
        };
        let rotated = WorldShape::oriented(Vec3::ZERO, long_box, basis);
        let axis_aligned = WorldShape::new(Vec3::ZERO, long_box);

        // The box is 6 long in x and 1 thick in z. Unrotated, nothing of it
        // reaches z = 2. Rotated 45 degrees about Y, its long axis swings out to
        // z ~= 2.47, so a ray travelling along -X at z = 2 now passes through it.
        let ray = Ray::new(Vec3::new(6.0, 0.0, 2.0), -Vec3::UNIT_X);
        assert!(
            intersect(&ray, &rotated, 0.001, T_MAX).is_some(),
            "the rotated box should be in the ray's path"
        );
        assert!(
            intersect(&ray, &axis_aligned, 0.001, T_MAX).is_none(),
            "the unrotated box should not be — otherwise the test proves nothing"
        );
    }

    #[test]
    fn identity_basis_matches_the_axis_aligned_path_exactly() {
        let shape = Shape::Box {
            half_extents: Vec3::new(1.0, 2.0, 0.5),
        };
        let ray = Ray::new(Vec3::new(0.3, 0.4, 5.0), -Vec3::UNIT_Z);
        let plain = intersect_box(&ray, Vec3::ZERO, Vec3::new(1.0, 2.0, 0.5), 0.001, T_MAX).unwrap();
        let via_obb = intersect(
            &ray,
            &WorldShape::new(Vec3::ZERO, shape),
            0.001,
            T_MAX,
        )
        .unwrap();
        assert!((plain.t - via_obb.t).abs() < 1e-6);
        assert!((plain.normal - via_obb.normal).length() < 1e-6);
    }

    /// A 90-degree rotation must move the reported normal with the face.
    #[test]
    fn rotated_box_normal_comes_back_in_world_space() {
        let basis = Basis::from_matrix(&crate::engine::math::Mat4::rotation_y(
            std::f32::consts::FRAC_PI_2,
        ));
        let shape = WorldShape::oriented(
            Vec3::ZERO,
            Shape::Box {
                half_extents: Vec3::new(1.0, 1.0, 2.0),
            },
            basis,
        );
        // After a quarter turn about Y the local +Z face points along world -X
        // (or +X), so a ray from +X must come back with a normal along +X.
        let ray = Ray::new(Vec3::new(6.0, 0.0, 0.0), -Vec3::UNIT_X);
        let hit = intersect(&ray, &shape, 0.001, T_MAX).expect("should hit");
        assert!(
            (hit.normal - Vec3::UNIT_X).length() < 1e-4,
            "normal = {}",
            hit.normal
        );
        // The long axis is now along X, so the hit is 2 units from the centre.
        assert!((hit.t - 4.0).abs() < 1e-4, "t = {}", hit.t);
    }

    #[test]
    fn triangle_hit_inside_and_miss_outside() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        let hit_ray = Ray::new(Vec3::new(0.0, 0.5, 5.0), -Vec3::UNIT_Z);
        let hit = intersect_triangle(&hit_ray, a, b, c, 0.001, T_MAX).expect("should hit");
        assert!((hit.t - 5.0).abs() < 1e-4);
        // The winding here faces +Z, and the ray comes from +Z, so it is a front face.
        assert!(hit.front_face);

        let miss_ray = Ray::new(Vec3::new(0.0, -1.0, 5.0), -Vec3::UNIT_Z);
        assert!(intersect_triangle(&miss_ray, a, b, c, 0.001, T_MAX).is_none());
    }

    /// The winding, not a vertex attribute, decides front/back. This is the same
    /// trap the rasteriser hit: a mesh with attribute normals that disagree with
    /// its winding vanishes under back-face culling, and a test on the attribute
    /// never notices.
    #[test]
    fn triangle_normal_comes_from_the_winding() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        let ray = Ray::new(Vec3::new(0.0, 0.5, 5.0), -Vec3::UNIT_Z);
        let front = intersect_triangle(&ray, a, b, c, 0.001, T_MAX).unwrap();
        // Reversing the winding must flip which side counts as the front.
        let back = intersect_triangle(&ray, b, a, c, 0.001, T_MAX).unwrap();
        assert!(front.front_face);
        assert!(!back.front_face);
        // Either way the reported normal faces the ray.
        assert!(front.normal.z > 0.9, "{}", front.normal);
        assert!(back.normal.z > 0.9, "{}", back.normal);
    }

    #[test]
    fn triangle_parallel_to_the_ray_is_a_miss() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        let ray = Ray::new(Vec3::new(0.0, 0.5, 0.0), Vec3::UNIT_X);
        assert!(intersect_triangle(&ray, a, b, c, 0.001, T_MAX).is_none());
    }

    /// Analytic and mesh intersection must agree, since the tracer uses one and
    /// gameplay raycasting may use the other on the very same object.
    #[test]
    fn sphere_and_its_inscribed_triangle_agree_on_depth_ordering() {
        let sphere = WorldShape::new(Vec3::ZERO, Shape::Sphere { radius: 1.0 });
        let ray = Ray::new(Vec3::new(0.0, 0.0, 4.0), -Vec3::UNIT_Z);
        let analytic = intersect(&ray, &sphere, 0.001, T_MAX).unwrap();
        // A triangle chord across the sphere at z = 0 must be strictly farther
        // than the sphere's front surface at z = 1.
        let chord = intersect_triangle(
            &ray,
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.001,
            T_MAX,
        )
        .unwrap();
        assert!(
            analytic.t < chord.t,
            "analytic t {} should precede the chord at t {}",
            analytic.t,
            chord.t
        );
    }
}
