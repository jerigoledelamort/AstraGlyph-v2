// Shape/shape collision detection, producing contacts the solver can act on.
//
// A boolean "do these overlap?" is not enough to build physics on. Separating two
// bodies requires knowing *which way* to push and *how far*, so every test here
// returns a `Contact` with a normal and a penetration depth rather than a `bool`.
// Getting only the boolean out of a narrow-phase test is the classic reason
// bodies jitter: the solver has to guess a direction, and guesses differently on
// consecutive frames.
//
// Shapes come from `super::shapes`, the same definitions the CPU tracer
// intersects, so a body cannot collide somewhere other than where it is drawn.

use super::shapes::{Basis, Shape, WorldShape};
use crate::engine::math::Vec3;

/// Below this overlap a contact is treated as touching rather than penetrating.
/// Two resting bodies always overlap by a hair once gravity has pressed them
/// together; treating that as a collision to be resolved is what makes a stack
/// vibrate forever.
pub const CONTACT_SLOP: f32 = 1.0e-4;

/// A resolved overlap between two shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Unit vector pointing from the first shape toward the second: the direction
    /// the second body must move to separate.
    pub normal: Vec3,
    /// How far the shapes overlap along `normal`. Always positive.
    pub depth: f32,
    /// A point in the contact region, world space. Used as the application point
    /// for the impulse.
    pub point: Vec3,
}

impl Contact {
    /// The same contact seen from the other shape: the normal flips, the depth
    /// and the point do not.
    ///
    /// Needed because every test below is written for one ordering of its
    /// operands, and the dispatcher reuses them in both.
    pub fn flipped(self) -> Self {
        Self {
            normal: -self.normal,
            depth: self.depth,
            point: self.point,
        }
    }
}

/// Two shapes and the contact between them.
#[derive(Clone, Copy, Debug)]
pub struct ContactPair {
    pub first: usize,
    pub second: usize,
    pub contact: Contact,
}

/// Whether two shapes are close enough to be worth an exact test.
///
/// Bounding spheres, because rotation cannot change a bounding radius — which is
/// exactly what makes this a valid early rejection for oriented boxes too.
pub fn broad_phase_overlap(a: &WorldShape, b: &WorldShape) -> bool {
    // A plane is unbounded in the directions its patch spans, and its bounding
    // radius describes the patch rather than an enclosing volume the way a
    // sphere's does. Rejecting a plane pair on radius alone would drop contacts
    // with a body sitting well inside a large ground plane.
    if matches!(a.shape, Shape::Plane { .. }) || matches!(b.shape, Shape::Plane { .. }) {
        return true;
    }
    let reach = a.bounding_radius() + b.bounding_radius();
    (b.origin - a.origin).length_squared() <= reach * reach
}

/// Contact between any two shapes, or `None` if they do not overlap.
///
/// The normal always points from `a` toward `b`.
pub fn collide(a: &WorldShape, b: &WorldShape) -> Option<Contact> {
    if !broad_phase_overlap(a, b) {
        return None;
    }
    match (a.shape, b.shape) {
        (Shape::Sphere { radius: ra }, Shape::Sphere { radius: rb }) => {
            sphere_sphere(a.origin, ra, b.origin, rb)
        }
        (Shape::Sphere { radius }, Shape::Box { half_extents }) => {
            sphere_box(a.origin, radius, b.origin, half_extents, &b.basis)
        }
        (Shape::Box { half_extents }, Shape::Sphere { radius }) => {
            // Written for (sphere, box); reuse it and flip rather than writing a
            // second version that can disagree about the sign.
            sphere_box(b.origin, radius, a.origin, half_extents, &a.basis)
                .map(Contact::flipped)
        }
        (Shape::Sphere { radius }, Shape::Plane { normal, half_size }) => {
            sphere_plane(a.origin, radius, b.origin, normal, half_size)
        }
        (Shape::Plane { normal, half_size }, Shape::Sphere { radius }) => {
            sphere_plane(b.origin, radius, a.origin, normal, half_size).map(Contact::flipped)
        }
        (
            Shape::Box {
                half_extents: ha,
            },
            Shape::Box {
                half_extents: hb,
            },
        ) => box_box(a.origin, ha, &a.basis, b.origin, hb, &b.basis),
        (Shape::Box { half_extents }, Shape::Plane { normal, half_size }) => {
            box_plane(a.origin, half_extents, &a.basis, b.origin, normal, half_size)
        }
        (Shape::Plane { normal, half_size }, Shape::Box { half_extents }) => {
            box_plane(b.origin, half_extents, &b.basis, a.origin, normal, half_size)
                .map(Contact::flipped)
        }
        // Two planes: no useful contact. Both are infinite in their own span, so
        // either they never meet or they meet along a line, and neither case has
        // a penetration depth a solver could act on.
        (Shape::Plane { .. }, Shape::Plane { .. }) => None,
    }
}

/// Sphere/sphere: one distance comparison.
pub fn sphere_sphere(ca: Vec3, ra: f32, cb: Vec3, rb: f32) -> Option<Contact> {
    let delta = cb - ca;
    let dist_sq = delta.length_squared();
    let reach = ra + rb;
    if dist_sq >= reach * reach {
        return None;
    }
    let dist = dist_sq.sqrt();
    // Concentric spheres have no separating direction. +Y is arbitrary but must
    // be *consistent*: a normal that varies between frames makes the pair
    // oscillate instead of separating.
    let normal = if dist > 1.0e-6 {
        delta / dist
    } else {
        Vec3::UNIT_Y
    };
    Some(Contact {
        normal,
        depth: reach - dist,
        // Midway through the overlap region.
        point: ca + normal * (ra - (reach - dist) * 0.5),
    })
}

/// Sphere/box (oriented). The normal points from the sphere toward the box.
///
/// Works by clamping the sphere's centre into the box's local extents: the
/// closest point on a box to any point is that clamp, which reduces the whole
/// test to sphere-versus-point.
pub fn sphere_box(
    sphere_center: Vec3,
    radius: f32,
    box_center: Vec3,
    half_extents: Vec3,
    basis: &Basis,
) -> Option<Contact> {
    let local = basis.to_local(sphere_center - box_center);
    let clamped = Vec3::new(
        local.x.clamp(-half_extents.x, half_extents.x),
        local.y.clamp(-half_extents.y, half_extents.y),
        local.z.clamp(-half_extents.z, half_extents.z),
    );
    let offset = local - clamped;
    let dist_sq = offset.length_squared();

    if dist_sq > radius * radius {
        return None;
    }

    if dist_sq > 1.0e-12 {
        // Centre outside the box: the closest point is on the surface, and the
        // direction from it to the centre is the contact normal.
        let dist = dist_sq.sqrt();
        let local_normal = offset / dist;
        // From sphere to box, i.e. against the outward direction.
        let normal = basis.to_world(-local_normal);
        return Some(Contact {
            normal,
            depth: radius - dist,
            point: box_center + basis.to_world(clamped),
        });
    }

    // Centre inside the box: the clamp gave back the centre itself, so there is
    // no direction to read off it. Push out along the *least* penetrated face —
    // the shortest way out is the only choice that does not shove the sphere
    // through the box.
    let gaps = [
        half_extents.x - local.x.abs(),
        half_extents.y - local.y.abs(),
        half_extents.z - local.z.abs(),
    ];
    let mut axis = 0usize;
    for i in 1..3 {
        if gaps[i] < gaps[axis] {
            axis = i;
        }
    }
    let sign = if local_component(local, axis) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let local_axis = unit_axis(axis) * sign;
    Some(Contact {
        // Sphere is inside, so pushing it out means moving it along +local_axis;
        // the normal points from sphere to box, hence the negation.
        normal: basis.to_world(-local_axis),
        depth: radius + gaps[axis],
        point: sphere_center,
    })
}

/// Sphere/plane. The normal points from the sphere toward the plane.
///
/// The plane's patch bound is applied to the *projected* centre, so a sphere
/// hanging off the edge of the ground does not collide with ground that is not
/// there.
pub fn sphere_plane(
    sphere_center: Vec3,
    radius: f32,
    plane_origin: Vec3,
    plane_normal: Vec3,
    half_size: f32,
) -> Option<Contact> {
    let n = plane_normal.normalize();
    let to_sphere = sphere_center - plane_origin;
    let signed = to_sphere.dot(n);
    if signed.abs() >= radius {
        return None;
    }
    // In-plane offset, tested against the patch.
    if half_size.is_finite() && half_size > 0.0 {
        let (u, v) = plane_axes(n);
        if to_sphere.dot(u).abs() > half_size || to_sphere.dot(v).abs() > half_size {
            return None;
        }
    }
    // A sphere below the plane is pushed back down, not yanked through it.
    let outward = if signed >= 0.0 { n } else { -n };
    Some(Contact {
        normal: -outward,
        depth: radius - signed.abs(),
        point: sphere_center - outward * signed.abs(),
    })
}

/// Box/plane. The normal points from the box toward the plane.
///
/// The deepest vertex decides: a box's penetration into a plane is the largest
/// penetration of any of its eight corners, and using the centre instead would
/// let a corner sink in unnoticed until half the box was through.
pub fn box_plane(
    box_center: Vec3,
    half_extents: Vec3,
    basis: &Basis,
    plane_origin: Vec3,
    plane_normal: Vec3,
    half_size: f32,
) -> Option<Contact> {
    let n = plane_normal.normalize();
    let centre_side = (box_center - plane_origin).dot(n);
    // Which face of the plane the box is on; a box below is pushed back down.
    let outward = if centre_side >= 0.0 { n } else { -n };

    // Projected radius of the box onto the plane normal.
    let reach = (basis.x.dot(outward) * half_extents.x).abs()
        + (basis.y.dot(outward) * half_extents.y).abs()
        + (basis.z.dot(outward) * half_extents.z).abs();
    let distance = centre_side.abs();
    if distance >= reach {
        return None;
    }

    // Deepest corner: step from the centre against the outward normal on every
    // axis that points that way.
    let mut deepest = box_center;
    for i in 0..3 {
        let axis = basis.axis(i);
        let extent = local_component(half_extents, i);
        deepest = deepest + axis * (-axis.dot(outward).signum() * extent);
    }

    if half_size.is_finite() && half_size > 0.0 {
        let (u, v) = plane_axes(n);
        let to_box = box_center - plane_origin;
        if to_box.dot(u).abs() > half_size + reach || to_box.dot(v).abs() > half_size + reach {
            return None;
        }
    }

    Some(Contact {
        normal: -outward,
        depth: reach - distance,
        point: deepest,
    })
}

/// Box/box by the separating-axis theorem (15 axes: 3 + 3 face normals and their
/// 9 cross products).
///
/// The axis of *minimum* overlap is the contact normal. Any separating axis
/// proves no collision, so the loop can exit early; but when they do collide,
/// every axis has to be measured, because picking a non-minimal one would push
/// the boxes apart the long way round — visibly teleporting them.
pub fn box_box(
    ca: Vec3,
    ha: Vec3,
    ba: &Basis,
    cb: Vec3,
    hb: Vec3,
    bb: &Basis,
) -> Option<Contact> {
    let delta = cb - ca;
    let mut best_axis = Vec3::UNIT_Y;
    let mut best_overlap = f32::INFINITY;

    let test = |axis: Vec3, best_axis: &mut Vec3, best_overlap: &mut f32| -> bool {
        let len_sq = axis.length_squared();
        // Cross products of near-parallel axes degenerate to zero length and
        // carry no information; skipping them is correct, testing them is a
        // division by almost nothing.
        if len_sq < 1.0e-8 {
            return true;
        }
        let axis = axis / len_sq.sqrt();
        let ra = projected_radius(ha, ba, axis);
        let rb = projected_radius(hb, bb, axis);
        let separation = delta.dot(axis).abs();
        let overlap = ra + rb - separation;
        if overlap <= 0.0 {
            return false; // separating axis found
        }
        if overlap < *best_overlap {
            *best_overlap = overlap;
            // Point the axis from a toward b, so the sign of the final normal
            // does not depend on which way the axis happened to be built.
            *best_axis = if delta.dot(axis) < 0.0 { -axis } else { axis };
        }
        true
    };

    for i in 0..3 {
        if !test(ba.axis(i), &mut best_axis, &mut best_overlap) {
            return None;
        }
        if !test(bb.axis(i), &mut best_axis, &mut best_overlap) {
            return None;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            if !test(
                ba.axis(i).cross(bb.axis(j)),
                &mut best_axis,
                &mut best_overlap,
            ) {
                return None;
            }
        }
    }

    if !best_overlap.is_finite() {
        return None;
    }

    Some(Contact {
        normal: best_axis,
        depth: best_overlap,
        // Approximated as the point on b's surface along the contact normal.
        // A proper manifold would need clipped face polygons; for the single
        // deepest-point solver here this is the point that matters.
        point: cb - best_axis * projected_radius(hb, bb, best_axis),
    })
}

/// Half-width of an oriented box projected onto a unit axis.
fn projected_radius(half_extents: Vec3, basis: &Basis, axis: Vec3) -> f32 {
    (basis.x.dot(axis) * half_extents.x).abs()
        + (basis.y.dot(axis) * half_extents.y).abs()
        + (basis.z.dot(axis) * half_extents.z).abs()
}

fn local_component(v: Vec3, index: usize) -> f32 {
    match index {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn unit_axis(index: usize) -> Vec3 {
    match index {
        0 => Vec3::UNIT_X,
        1 => Vec3::UNIT_Y,
        _ => Vec3::UNIT_Z,
    }
}

/// Two orthonormal in-plane axes for a unit normal.
fn plane_axes(n: Vec3) -> (Vec3, Vec3) {
    let reference = if n.y.abs() < 0.9 {
        Vec3::UNIT_Y
    } else {
        Vec3::UNIT_X
    };
    let u = n.cross(reference).normalize();
    let v = n.cross(u);
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::Mat4;

    fn sphere(center: Vec3, radius: f32) -> WorldShape {
        WorldShape::new(center, Shape::Sphere { radius })
    }

    fn boxed(center: Vec3, half: Vec3) -> WorldShape {
        WorldShape::new(center, Shape::Box { half_extents: half })
    }

    fn ground(y: f32) -> WorldShape {
        WorldShape::new(
            Vec3::new(0.0, y, 0.0),
            Shape::Plane {
                normal: Vec3::UNIT_Y,
                half_size: 25.0,
            },
        )
    }

    // --- sphere/sphere ---

    #[test]
    fn separated_spheres_do_not_collide() {
        assert!(collide(&sphere(Vec3::ZERO, 1.0), &sphere(Vec3::new(3.0, 0.0, 0.0), 1.0)).is_none());
    }

    #[test]
    fn touching_spheres_are_not_a_collision() {
        // Exactly reach apart: no overlap to resolve.
        assert!(collide(&sphere(Vec3::ZERO, 1.0), &sphere(Vec3::new(2.0, 0.0, 0.0), 1.0)).is_none());
    }

    #[test]
    fn overlapping_spheres_report_depth_and_direction() {
        let c = collide(&sphere(Vec3::ZERO, 1.0), &sphere(Vec3::new(1.5, 0.0, 0.0), 1.0))
            .expect("should collide");
        assert!((c.depth - 0.5).abs() < 1e-5, "depth = {}", c.depth);
        assert!(
            (c.normal - Vec3::UNIT_X).length() < 1e-5,
            "normal must point from a to b: {}",
            c.normal
        );
        // The contact point lies between the two centres.
        assert!(c.point.x > 0.0 && c.point.x < 1.5, "point = {}", c.point);
    }

    /// Concentric spheres have no separating direction. The normal must still be
    /// finite and, crucially, the same every call — an unstable normal makes a
    /// pair oscillate rather than separate.
    #[test]
    fn concentric_spheres_get_a_stable_fallback_normal() {
        let a = collide(&sphere(Vec3::ZERO, 1.0), &sphere(Vec3::ZERO, 1.0)).unwrap();
        let b = collide(&sphere(Vec3::ZERO, 1.0), &sphere(Vec3::ZERO, 1.0)).unwrap();
        assert!(a.normal.length() > 0.5, "normal must not be zero");
        assert!(a.normal.x.is_finite() && a.normal.y.is_finite());
        assert_eq!(a.normal, b.normal, "the fallback normal must be deterministic");
    }

    /// Swapping the operands must flip the normal and nothing else. A test that
    /// only ever checks one ordering would miss a dispatcher that forgot to flip.
    #[test]
    fn swapping_operands_flips_only_the_normal() {
        let a = sphere(Vec3::ZERO, 1.0);
        let b = sphere(Vec3::new(1.5, 0.0, 0.0), 1.0);
        let ab = collide(&a, &b).unwrap();
        let ba = collide(&b, &a).unwrap();
        assert!((ab.normal + ba.normal).length() < 1e-5, "normals must oppose");
        assert!((ab.depth - ba.depth).abs() < 1e-5, "depth must not change");
    }

    // --- sphere/plane ---

    #[test]
    fn sphere_resting_above_a_plane_does_not_collide() {
        assert!(collide(&sphere(Vec3::new(0.0, 2.0, 0.0), 1.0), &ground(0.0)).is_none());
    }

    #[test]
    fn sphere_sinking_into_a_plane_is_pushed_up() {
        let c = collide(&sphere(Vec3::new(0.0, 0.75, 0.0), 1.0), &ground(0.0)).unwrap();
        assert!((c.depth - 0.25).abs() < 1e-5, "depth = {}", c.depth);
        // Normal points from sphere to plane, i.e. downward; the solver moves the
        // sphere along -normal.
        assert!(
            (c.normal - (-Vec3::UNIT_Y)).length() < 1e-5,
            "normal = {}",
            c.normal
        );
    }

    /// A sphere *below* the plane must be pushed further down, not yanked up
    /// through it. Getting this wrong makes anything that clips through the floor
    /// snap violently to the top of it.
    #[test]
    fn sphere_below_a_plane_is_pushed_further_down() {
        let c = collide(&sphere(Vec3::new(0.0, -0.75, 0.0), 1.0), &ground(0.0)).unwrap();
        assert!(
            (c.normal - Vec3::UNIT_Y).length() < 1e-5,
            "normal should point up (from sphere toward plane): {}",
            c.normal
        );
        assert!((c.depth - 0.25).abs() < 1e-5);
    }

    /// The engine's ground is a 50-unit patch, not an infinite plane. A sphere
    /// past its edge must fall, not stand on geometry that is not there.
    #[test]
    fn sphere_beyond_the_plane_patch_does_not_collide() {
        let inside = collide(&sphere(Vec3::new(20.0, 0.5, 0.0), 1.0), &ground(0.0));
        assert!(inside.is_some(), "20 units out is inside a half-size-25 patch");
        let outside = collide(&sphere(Vec3::new(40.0, 0.5, 0.0), 1.0), &ground(0.0));
        assert!(
            outside.is_none(),
            "40 units out is past the patch edge, so there is nothing to stand on"
        );
    }

    // --- sphere/box ---

    #[test]
    fn sphere_beside_a_box_does_not_collide() {
        assert!(collide(&sphere(Vec3::new(5.0, 0.0, 0.0), 1.0), &boxed(Vec3::ZERO, Vec3::ONE)).is_none());
    }

    #[test]
    fn sphere_touching_a_box_face_gets_that_face_normal() {
        // Sphere centre 1.5 out on +X, box reaches 1.0: overlap 0.5.
        let c = collide(
            &sphere(Vec3::new(1.5, 0.0, 0.0), 1.0),
            &boxed(Vec3::ZERO, Vec3::ONE),
        )
        .unwrap();
        assert!((c.depth - 0.5).abs() < 1e-5, "depth = {}", c.depth);
        assert!(
            (c.normal - (-Vec3::UNIT_X)).length() < 1e-5,
            "normal should point from the sphere toward the box: {}",
            c.normal
        );
    }

    /// A sphere whose centre is *inside* a box has no closest surface point to
    /// read a direction from. It must leave through the nearest face — pushing it
    /// out along any other axis shoves it through the box.
    #[test]
    fn sphere_inside_a_box_exits_through_the_nearest_face() {
        // Deep in x, shallow in y: y is the shortest way out.
        let c = collide(
            &sphere(Vec3::new(0.1, 2.4, 0.0), 0.2),
            &boxed(Vec3::ZERO, Vec3::new(3.0, 2.5, 3.0)),
        )
        .unwrap();
        assert!(
            c.normal.y.abs() > 0.99,
            "should exit along y, the shallow axis: {}",
            c.normal
        );
        assert!(c.depth > 0.0);
    }

    /// Rotation must actually be honoured. A 45-degree box reaches further along
    /// its diagonal than an axis-aligned one, so a sphere placed in that gap
    /// collides with one and not the other.
    #[test]
    fn rotated_box_collides_where_it_actually_is() {
        let basis = Basis::from_matrix(&Mat4::rotation_y(std::f32::consts::FRAC_PI_4));
        let half = Vec3::new(2.0, 1.0, 0.5);
        let rotated = WorldShape::oriented(Vec3::ZERO, Shape::Box { half_extents: half }, basis);
        let aligned = boxed(Vec3::ZERO, half);
        // 2.2 out along the box's own +x, i.e. 0.2 beyond its 2.0 face: a genuine
        // surface contact for the rotated box. In world space the 45-degree turn
        // puts that at (1.556, 0, -1.556), which is 1.556 out on z — far past the
        // aligned box's 0.5 extent plus the sphere's 0.4 radius.
        let probe = sphere(Vec3::new(1.556, 0.0, -1.556), 0.4);
        let c = collide(&probe, &rotated).expect("the rotated box's +x face is right there");
        assert!((c.depth - 0.2).abs() < 1e-2, "depth = {}", c.depth);
        assert!(
            collide(&probe, &aligned).is_none(),
            "the aligned box does not reach — otherwise the test proves nothing"
        );
    }

    // --- box/box ---

    #[test]
    fn separated_boxes_do_not_collide() {
        assert!(collide(&boxed(Vec3::ZERO, Vec3::ONE), &boxed(Vec3::new(5.0, 0.0, 0.0), Vec3::ONE)).is_none());
    }

    #[test]
    fn overlapping_boxes_report_the_minimum_overlap_axis() {
        // 1.5 apart on x with 1.0 half-extents each: 0.5 overlap on x, 2.0 on the
        // other two. The solver must be told to push along x, the cheap way out.
        let c = collide(
            &boxed(Vec3::ZERO, Vec3::ONE),
            &boxed(Vec3::new(1.5, 0.0, 0.0), Vec3::ONE),
        )
        .unwrap();
        assert!((c.depth - 0.5).abs() < 1e-5, "depth = {}", c.depth);
        assert!(
            (c.normal - Vec3::UNIT_X).length() < 1e-4,
            "normal should be +x, the minimum-overlap axis: {}",
            c.normal
        );
    }

    /// The minimum-overlap choice is the whole point of SAT for a solver. If the
    /// deepest axis were picked instead, a pair overlapping slightly on one axis
    /// and heavily on another would be flung apart the long way.
    #[test]
    fn minimum_overlap_axis_is_chosen_over_a_deeper_one() {
        // Big flat slab and a cube resting slightly inside its top face: the y
        // overlap is tiny, the x and z overlaps are large.
        let slab = boxed(Vec3::ZERO, Vec3::new(10.0, 1.0, 10.0));
        let cube = boxed(Vec3::new(0.0, 1.9, 0.0), Vec3::ONE);
        let c = collide(&slab, &cube).unwrap();
        assert!(
            c.normal.y.abs() > 0.99,
            "must separate along y: {}",
            c.normal
        );
        assert!((c.depth - 0.1).abs() < 1e-4, "depth = {}", c.depth);
    }

    /// A 45-degree rotation is what separates real SAT from an AABB test: the
    /// cross-product axes are the only ones that can find the separating
    /// direction between two differently-oriented boxes.
    #[test]
    fn rotated_boxes_use_the_cross_product_axes() {
        let basis = Basis::from_matrix(&Mat4::rotation_y(std::f32::consts::FRAC_PI_4));
        let a = boxed(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        // Placed on the diagonal where an AABB test would say "clear" but the
        // rotated box's corner reaches in.
        let b = WorldShape::oriented(
            Vec3::new(1.9, 0.0, 0.0),
            Shape::Box {
                half_extents: Vec3::new(1.0, 1.0, 1.0),
            },
            basis,
        );
        let c = collide(&a, &b).expect("the rotated corner reaches into a");
        assert!(c.depth > 0.0 && c.depth.is_finite());
        assert!(c.normal.length() > 0.99, "normal must be a unit vector");
        // Pushing along the reported normal by the reported depth must separate
        // them — the property the solver actually relies on.
        let moved = WorldShape::oriented(
            b.origin + c.normal * (c.depth + 1e-3),
            b.shape,
            basis,
        );
        assert!(
            collide(&a, &moved).is_none(),
            "moving along the normal by the depth must resolve the overlap"
        );
    }

    /// The same separation property, on the axis-aligned path.
    #[test]
    fn resolving_by_the_reported_contact_separates_the_pair() {
        for (offset, half_a, half_b) in [
            (Vec3::new(1.5, 0.0, 0.0), Vec3::ONE, Vec3::ONE),
            (Vec3::new(0.2, 1.9, 0.1), Vec3::new(10.0, 1.0, 10.0), Vec3::ONE),
            (Vec3::new(0.5, 0.5, 0.5), Vec3::ONE, Vec3::splat(0.5)),
        ] {
            let a = boxed(Vec3::ZERO, half_a);
            let b = boxed(offset, half_b);
            let c = collide(&a, &b).unwrap_or_else(|| panic!("expected overlap at {offset}"));
            let moved = boxed(b.origin + c.normal * (c.depth + 1e-3), half_b);
            assert!(
                collide(&a, &moved).is_none(),
                "resolving at {offset} left them overlapping"
            );
        }
    }

    // --- box/plane ---

    #[test]
    fn box_above_a_plane_does_not_collide() {
        assert!(collide(&boxed(Vec3::new(0.0, 3.0, 0.0), Vec3::ONE), &ground(0.0)).is_none());
    }

    #[test]
    fn box_sinking_into_a_plane_reports_the_deepest_corner() {
        let c = collide(&boxed(Vec3::new(0.0, 0.75, 0.0), Vec3::ONE), &ground(0.0)).unwrap();
        assert!((c.depth - 0.25).abs() < 1e-5, "depth = {}", c.depth);
        assert!((c.normal - (-Vec3::UNIT_Y)).length() < 1e-5);
        // The contact point is the bottom face, not the centre.
        assert!(c.point.y < 0.0, "contact point should be below the plane: {}", c.point);
    }

    /// A rotated box's corner dips lower than its face. Using the centre or the
    /// unrotated extent would let the corner sink in unnoticed.
    #[test]
    fn rotated_box_reaches_deeper_into_a_plane_than_an_aligned_one() {
        let basis = Basis::from_matrix(&Mat4::rotation_z(std::f32::consts::FRAC_PI_4));
        let half = Vec3::ONE;
        let y = 1.2;
        let aligned = collide(&boxed(Vec3::new(0.0, y, 0.0), half), &ground(0.0));
        let rotated = collide(
            &WorldShape::oriented(
                Vec3::new(0.0, y, 0.0),
                Shape::Box { half_extents: half },
                basis,
            ),
            &ground(0.0),
        );
        assert!(
            aligned.is_none(),
            "an aligned unit box at y=1.2 clears the ground"
        );
        let rotated = rotated.expect("a 45-degree box reaches sqrt(2) down and does not");
        assert!(rotated.depth > 0.0);
    }

    // --- broad phase ---

    #[test]
    fn broad_phase_rejects_distant_pairs_and_keeps_close_ones() {
        assert!(!broad_phase_overlap(
            &sphere(Vec3::ZERO, 1.0),
            &sphere(Vec3::new(100.0, 0.0, 0.0), 1.0)
        ));
        assert!(broad_phase_overlap(
            &sphere(Vec3::ZERO, 1.0),
            &sphere(Vec3::new(1.5, 0.0, 0.0), 1.0)
        ));
    }

    /// A plane's bounding radius describes its patch, not an enclosing volume, so
    /// rejecting on it would drop a body sitting in the middle of a large ground
    /// plane — the single most common contact in any scene.
    #[test]
    fn broad_phase_never_rejects_a_plane() {
        // Ground centred at the origin, sphere 20 units out: the centre distance
        // exceeds the sphere's radius plus the plane's, yet they do touch.
        let far = sphere(Vec3::new(20.0, 0.5, 0.0), 1.0);
        assert!(broad_phase_overlap(&far, &ground(0.0)));
        assert!(
            collide(&far, &ground(0.0)).is_some(),
            "and the exact test agrees, which is what makes the rejection wrong"
        );
    }

    #[test]
    fn two_planes_produce_no_contact() {
        assert!(collide(&ground(0.0), &ground(0.5)).is_none());
    }

    #[test]
    fn contact_slop_is_small_but_nonzero() {
        // Zero slop makes resting bodies jitter; a large one makes them sink.
        assert!(CONTACT_SLOP > 0.0 && CONTACT_SLOP < 1.0e-2);
    }
}
