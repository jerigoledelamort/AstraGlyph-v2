// View-frustum culling math: Aabb, Plane, Frustum (ROADMAP Phase 2.2).
//
// Pure CPU math, no GPU state — the renderer asks `Frustum::intersects_aabb`
// before recording draw calls for an entity.
//
// Design notes:
// - Planes are extracted from the view-projection matrix with the Gribb-Hartmann
//   method. That works in *any* space: feed a projection matrix and you get
//   camera-space planes, feed projection * view and you get world-space planes.
// - The project's `Mat4::perspective` is the OpenGL convention (clip z maps to
//   [-1, 1], m[11] = -1), so the visibility test is -w <= {x,y,z} <= w. The near
//   plane therefore comes from `row2 + row3`, NOT from `row2` alone (which is the
//   Direct3D / [0,1]-depth form and would silently place the near plane wrong).
// - `Mat4` is COLUMN-major: element (row, col) = m[col * 4 + row]. Gribb-Hartmann
//   needs ROWS, so row i is (m[i], m[4+i], m[8+i], m[12+i]).
// - Every plane is normalized so `signed_distance` is a true euclidean distance;
//   the sphere test depends on that.
// - All culling tests are *conservative*: they may report "visible" for a box that
//   is actually outside (the classic near-corner false positive), but they must
//   never report "not visible" for something on screen — a false negative makes
//   geometry pop out of existence.

use crate::engine::math::{Mat4, Vec3};

/// Axis-aligned bounding box, stored as two opposite corners.
///
/// Invariant expected by all methods: `min.k <= max.k` on every axis. The
/// constructors here maintain it; if you build one by hand, keep it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Create a box from two corners, sorting them per-axis so the invariant holds.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            min: Vec3::new(min.x.min(max.x), min.y.min(max.y), min.z.min(max.z)),
            max: Vec3::new(min.x.max(max.x), min.y.max(max.y), min.z.max(max.z)),
        }
    }

    /// Tightest box containing all `points`. `None` for an empty iterator —
    /// an empty box has no meaningful min/max, and returning a degenerate box
    /// at the origin would silently cull (or un-cull) empty meshes.
    pub fn from_points(points: impl Iterator<Item = Vec3>) -> Option<Self> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;

        for p in points {
            any = true;
            min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
            max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
        }

        if any {
            Some(Self { min, max })
        } else {
            None
        }
    }

    /// Geometric center of the box.
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Full size of the box along each axis (`max - min`).
    pub fn extents(self) -> Vec3 {
        self.max - self.min
    }

    /// Half size of the box along each axis — the vector from center to `max`.
    pub fn half_extents(self) -> Vec3 {
        self.extents() * 0.5
    }

    /// Radius of the sphere centered at `center()` that encloses the box.
    pub fn radius(self) -> f32 {
        self.half_extents().length()
    }

    /// Smallest box containing both `self` and `other`.
    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Inclusive point-in-box test (points exactly on a face count as inside).
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// The eight corners, ordered by the bit pattern of (x, y, z) picking min/max.
    pub fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// AABB of the eight transformed corners.
    ///
    /// Needed once entities carry model matrices (Phase 2.1): a local-space mesh
    /// bound is transformed to world space here before being culled. Intended for
    /// affine matrices (translate / rotate / scale); the result is a re-fitted
    /// box, so repeated rotation slowly inflates it — always transform the
    /// *local* bound, never a previously transformed one.
    pub fn transformed(&self, m: &Mat4) -> Aabb {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for corner in self.corners() {
            let p = m.transform_point(corner);
            min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
            max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
        }

        Aabb { min, max }
    }
}

/// Infinite plane in the form `dot(normal, p) + distance = 0`.
///
/// `distance` is the signed offset of the plane along `normal` (negated), i.e.
/// for a normalized plane the plane's closest point to the origin is
/// `normal * -distance`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
}

impl Plane {
    /// Create a plane from a normal and a `distance` offset (not normalized).
    pub const fn new(normal: Vec3, distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Create a plane from the raw coefficients `a*x + b*y + c*z + d = 0`,
    /// scaled so `normal` has unit length. Degenerate input (zero normal, e.g.
    /// from a singular matrix) is returned unscaled rather than producing NaNs.
    pub fn from_coefficients(a: f32, b: f32, c: f32, d: f32) -> Self {
        let normal = Vec3::new(a, b, c);
        let len = normal.length();
        if len < 1e-20 {
            return Self { normal, distance: d };
        }
        let inv = 1.0 / len;
        Self {
            normal: normal * inv,
            distance: d * inv,
        }
    }

    /// Signed distance from the plane to `p`: positive on the side the normal
    /// points at, negative behind, zero exactly on the plane. Only a true
    /// distance if the plane is normalized (`from_coefficients` guarantees it).
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.distance
    }
}

/// Six-sided view frustum with inward-facing normals: a point is visible iff
/// `signed_distance` is non-negative against all six planes.
///
/// Built from a view-projection matrix, so the planes live in whatever space the
/// matrix's input is (world space for `projection * view`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// Index of the left plane in `planes()`.
    pub const LEFT: usize = 0;
    /// Index of the right plane in `planes()`.
    pub const RIGHT: usize = 1;
    /// Index of the bottom plane in `planes()`.
    pub const BOTTOM: usize = 2;
    /// Index of the top plane in `planes()`.
    pub const TOP: usize = 3;
    /// Index of the near plane in `planes()`.
    pub const NEAR: usize = 4;
    /// Index of the far plane in `planes()`.
    pub const FAR: usize = 5;

    /// Wrap six pre-built planes. Normals must face *inward* and should be
    /// normalized, otherwise the sphere test loses its meaning.
    pub const fn from_planes(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Extract the six planes from a view-projection matrix (Gribb-Hartmann).
    ///
    /// Derivation for the OpenGL clip volume used by `Mat4::perspective`
    /// (`-w <= x,y,z <= w`), with `row_i` the i-th row of `view_proj`:
    /// left = row0 + row3, right = row3 - row0, bottom = row1 + row3,
    /// top = row3 - row1, near = row2 + row3, far = row3 - row2.
    /// Each combination is the coefficient vector of an inward-facing plane.
    pub fn from_view_projection(view_proj: &Mat4) -> Self {
        let m = &view_proj.m;

        // Rows of a column-major matrix: row i = (m[i], m[4+i], m[8+i], m[12+i]).
        let row = |i: usize| [m[i], m[4 + i], m[8 + i], m[12 + i]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));

        let add = |a: [f32; 4], b: [f32; 4]| {
            Plane::from_coefficients(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])
        };
        let sub = |a: [f32; 4], b: [f32; 4]| {
            Plane::from_coefficients(a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])
        };

        Self {
            planes: [
                add(r0, r3), // left:   x_clip + w >= 0
                sub(r3, r0), // right:  w - x_clip >= 0
                add(r1, r3), // bottom: y_clip + w >= 0
                sub(r3, r1), // top:    w - y_clip >= 0
                add(r2, r3), // near:   z_clip + w >= 0
                sub(r3, r2), // far:    w - z_clip >= 0
            ],
        }
    }

    /// The six planes in the order left, right, bottom, top, near, far.
    pub fn planes(&self) -> &[Plane; 6] {
        &self.planes
    }

    /// True if `p` lies inside (or exactly on) all six planes.
    pub fn contains_point(&self, p: Vec3) -> bool {
        self.planes.iter().all(|plane| plane.signed_distance(p) >= 0.0)
    }

    /// Conservative box-vs-frustum test.
    ///
    /// For each plane we evaluate only the box corner farthest along the plane
    /// normal ("positive vertex"): if even that corner is behind the plane, the
    /// whole box is, and the box is definitely invisible. Boxes that survive all
    /// six planes are reported visible — including a few that merely straddle
    /// two planes near a frustum edge, which is the accepted false positive.
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        let center = aabb.center();
        // `Aabb` has public fields, so a hand-built box can violate min <= max
        // and yield negative half extents. Take the magnitude: a negative
        // projected radius would shrink the box and could cull visible geometry,
        // and a false negative is the one failure mode we must never allow.
        let e = aabb.half_extents();
        let half = Vec3::new(e.x.abs(), e.y.abs(), e.z.abs());

        for plane in &self.planes {
            let n = plane.normal;
            // Projected radius of the box onto the plane normal.
            let projected = n.x.abs() * half.x + n.y.abs() * half.y + n.z.abs() * half.z;
            if plane.signed_distance(center) + projected < 0.0 {
                return false;
            }
        }
        true
    }

    /// Conservative sphere-vs-frustum test: outside only if the center is more
    /// than `radius` behind some plane.
    ///
    /// `radius` must be non-negative (`Aabb::radius` always is); a negative value
    /// would tighten the test instead of loosening it and could cull a visible
    /// sphere.
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(center) >= -radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::radians;

    /// Reference camera: eye at +5z looking at the origin, 60deg vertical fov,
    /// square aspect, near 0.1, far 100. In world space that puts the near plane
    /// at z = 4.9, the far plane at z = -95, and the apex at the eye.
    fn test_view_projection() -> Mat4 {
        let view = Mat4::look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::UNIT_Y);
        let proj = Mat4::perspective(radians(60.0), 1.0, 0.1, 100.0);
        proj.mul(view)
    }

    fn test_frustum() -> Frustum {
        Frustum::from_view_projection(&test_view_projection())
    }

    fn box_at(center: Vec3, half: f32) -> Aabb {
        Aabb::new(center - Vec3::splat(half), center + Vec3::splat(half))
    }

    // --- Aabb ------------------------------------------------------------

    #[test]
    fn aabb_new_sorts_corners() {
        let a = Aabb::new(Vec3::new(1.0, 5.0, -2.0), Vec3::new(-3.0, 2.0, 4.0));
        assert_eq!(a.min, Vec3::new(-3.0, 2.0, -2.0));
        assert_eq!(a.max, Vec3::new(1.0, 5.0, 4.0));
    }

    #[test]
    fn aabb_from_points_empty_is_none() {
        let empty: Vec<Vec3> = Vec::new();
        assert!(Aabb::from_points(empty.into_iter()).is_none());
    }

    #[test]
    fn aabb_from_points_fits_all() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(-1.0, 4.0, 2.0),
            Vec3::new(3.0, -2.0, 1.0),
        ];
        let a = Aabb::from_points(points.iter().copied()).expect("non-empty");
        assert_eq!(a.min, Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(a.max, Vec3::new(3.0, 4.0, 2.0));
        for p in points {
            assert!(a.contains_point(p));
        }
    }

    #[test]
    fn aabb_from_points_single_point_is_degenerate_box() {
        let p = Vec3::new(2.0, -1.0, 7.0);
        let a = Aabb::from_points(std::iter::once(p)).expect("non-empty");
        assert_eq!(a.min, p);
        assert_eq!(a.max, p);
        assert_eq!(a.extents(), Vec3::ZERO);
        assert_eq!(a.radius(), 0.0);
        assert!(a.contains_point(p));
    }

    #[test]
    fn aabb_center_extents_radius() {
        let a = Aabb::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.center(), Vec3::ZERO);
        assert_eq!(a.extents(), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.half_extents(), Vec3::new(1.0, 2.0, 3.0));
        // sqrt(1 + 4 + 9)
        assert!((a.radius() - 14.0f32.sqrt()).abs() < 1e-6);
        // The bounding sphere must actually cover every corner.
        for c in a.corners() {
            assert!((c - a.center()).length() <= a.radius() + 1e-6);
        }
    }

    #[test]
    fn aabb_merge_covers_both() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let b = Aabb::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(4.0, 0.5, 0.5));
        let m = a.merge(&b);
        assert_eq!(m.min, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(m.max, Vec3::new(4.0, 1.0, 1.0));
        // merge is commutative and idempotent.
        assert_eq!(b.merge(&a), m);
        assert_eq!(a.merge(&a), a);
    }

    #[test]
    fn aabb_contains_point_boundaries() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(a.contains_point(Vec3::ZERO));
        assert!(a.contains_point(Vec3::splat(1.0))); // on the corner
        assert!(a.contains_point(Vec3::new(-1.0, 0.0, 0.0))); // on a face
        assert!(!a.contains_point(Vec3::new(1.001, 0.0, 0.0)));
        assert!(!a.contains_point(Vec3::new(0.0, 0.0, -5.0)));
    }

    #[test]
    fn aabb_corners_are_eight_distinct_points() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let corners = a.corners();
        for i in 0..8 {
            assert!(a.contains_point(corners[i]));
            for j in (i + 1)..8 {
                assert_ne!(corners[i], corners[j]);
            }
        }
        // Re-fitting the corners must reproduce the box exactly.
        let refit = Aabb::from_points(corners.into_iter()).expect("8 corners");
        assert_eq!(refit, a);
    }

    #[test]
    fn aabb_transformed_translation() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let t = a.transformed(&Mat4::translation(5.0, -2.0, 3.0));
        assert!((t.min - Vec3::new(4.0, -3.0, 2.0)).length() < 1e-6);
        assert!((t.max - Vec3::new(6.0, -1.0, 4.0)).length() < 1e-6);
        // Translation preserves size.
        assert!((t.extents() - a.extents()).length() < 1e-6);
    }

    #[test]
    fn aabb_transformed_rotation() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        // 45deg about Z: the square cross-section's diagonal becomes the x/y size.
        let r = a.transformed(&Mat4::rotation_z(radians(45.0)));
        let diag = 2.0f32.sqrt();
        assert!((r.max.x - diag).abs() < 1e-5, "max.x = {}", r.max.x);
        assert!((r.max.y - diag).abs() < 1e-5, "max.y = {}", r.max.y);
        assert!((r.min.x + diag).abs() < 1e-5);
        assert!((r.min.y + diag).abs() < 1e-5);
        // Z is the rotation axis and must be untouched.
        assert!((r.min.z + 1.0).abs() < 1e-6);
        assert!((r.max.z - 1.0).abs() < 1e-6);
        // A rotated box still contains every rotated corner.
        let m = Mat4::rotation_z(radians(45.0));
        for c in a.corners() {
            assert!(r.contains_point(m.transform_point(c)));
        }
    }

    #[test]
    fn aabb_transformed_scale_then_translate() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        // Mat4::mul is `self * other`, so translation is applied after scaling.
        let m = Mat4::translation(10.0, 0.0, 0.0).mul(Mat4::scaling(2.0, 3.0, 4.0));
        let t = a.transformed(&m);
        assert!((t.min - Vec3::new(8.0, -3.0, -4.0)).length() < 1e-5);
        assert!((t.max - Vec3::new(12.0, 3.0, 4.0)).length() < 1e-5);
    }

    #[test]
    fn aabb_transformed_identity_is_noop() {
        let a = Aabb::new(Vec3::new(-2.0, 0.5, 1.0), Vec3::new(3.0, 4.0, 7.0));
        assert_eq!(a.transformed(&Mat4::IDENTITY), a);
    }

    // --- Plane -----------------------------------------------------------

    #[test]
    fn plane_signed_distance_sign_convention() {
        // Plane y = 0, normal pointing up.
        let p = Plane::new(Vec3::UNIT_Y, 0.0);
        assert!((p.signed_distance(Vec3::new(0.0, 2.0, 0.0)) - 2.0).abs() < 1e-6);
        assert!((p.signed_distance(Vec3::new(0.0, -3.0, 0.0)) + 3.0).abs() < 1e-6);
        assert!(p.signed_distance(Vec3::new(7.0, 0.0, -4.0)).abs() < 1e-6);
    }

    #[test]
    fn plane_signed_distance_with_offset() {
        // distance = -1 with normal +Y puts the plane at y = 1.
        let p = Plane::new(Vec3::UNIT_Y, -1.0);
        assert!(p.signed_distance(Vec3::ZERO) < 0.0);
        assert!(p.signed_distance(Vec3::new(0.0, 1.0, 0.0)).abs() < 1e-6);
        assert!((p.signed_distance(Vec3::new(0.0, 4.0, 0.0)) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn plane_from_coefficients_normalizes() {
        // Same plane as (1,0,0,-5), scaled by 3.
        let p = Plane::from_coefficients(3.0, 0.0, 0.0, -15.0);
        assert!((p.normal.length() - 1.0).abs() < 1e-6);
        assert!((p.normal - Vec3::UNIT_X).length() < 1e-6);
        // Real euclidean distance, not 3x of it.
        assert!((p.signed_distance(Vec3::new(6.0, 0.0, 0.0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn plane_from_coefficients_degenerate_normal_does_not_nan() {
        let p = Plane::from_coefficients(0.0, 0.0, 0.0, 2.0);
        assert!(p.normal.length() < 1e-9);
        assert!(p.signed_distance(Vec3::ONE).is_finite());
    }

    // --- Frustum extraction ----------------------------------------------

    #[test]
    fn frustum_planes_are_normalized() {
        let f = test_frustum();
        for (i, plane) in f.planes().iter().enumerate() {
            assert!(
                (plane.normal.length() - 1.0).abs() < 1e-4,
                "plane {} not normalized: len = {}",
                i,
                plane.normal.length()
            );
        }
    }

    #[test]
    fn frustum_near_far_planes_match_camera_setup() {
        let f = test_frustum();
        let near = f.planes()[Frustum::NEAR];
        let far = f.planes()[Frustum::FAR];

        // Near plane: 0.1 in front of the eye at z = 5 → z = 4.9, facing -Z.
        assert!((near.normal - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4);
        assert!(near.signed_distance(Vec3::new(0.0, 0.0, 4.9)).abs() < 1e-3);
        // Far plane: 100 in front of the eye → z = -95, facing +Z.
        assert!((far.normal - Vec3::UNIT_Z).length() < 1e-3);
        // The far plane's coefficients come from `row3 - row2`, a subtraction of
        // two nearly equal numbers, so it carries visibly more f32 noise than the
        // other five planes — hence the looser tolerance here.
        assert!(
            far.signed_distance(Vec3::new(0.0, 0.0, -95.0)).abs() < 0.1,
            "far plane misplaced: distance = {}, signed = {}",
            far.distance,
            far.signed_distance(Vec3::new(0.0, 0.0, -95.0))
        );
    }

    #[test]
    fn frustum_side_planes_pass_through_the_eye() {
        // All four lateral planes of a perspective frustum meet at the apex.
        let f = test_frustum();
        let eye = Vec3::new(0.0, 0.0, 5.0);
        for i in [Frustum::LEFT, Frustum::RIGHT, Frustum::BOTTOM, Frustum::TOP] {
            assert!(
                f.planes()[i].signed_distance(eye).abs() < 1e-4,
                "plane {} does not contain the apex",
                i
            );
        }
    }

    #[test]
    fn frustum_side_plane_normals_point_inward() {
        // Every inward normal must have a positive dot with the direction from
        // the plane toward a known-interior point.
        let f = test_frustum();
        let inside = Vec3::ZERO;
        for (i, plane) in f.planes().iter().enumerate() {
            assert!(
                plane.signed_distance(inside) > 0.0,
                "plane {} reports the frustum center as outside",
                i
            );
        }
    }

    #[test]
    fn frustum_from_orthographic_matrix() {
        // Identity view + ortho box: planes must land exactly on the box faces.
        let ortho = Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
        let f = Frustum::from_view_projection(&ortho);
        let left = f.planes()[Frustum::LEFT];
        assert!((left.normal - Vec3::UNIT_X).length() < 1e-5);
        assert!(left.signed_distance(Vec3::new(-1.0, 0.0, 0.0)).abs() < 1e-4);

        // Ortho near plane sits at view-space z = -0.1 and faces -Z.
        let near = f.planes()[Frustum::NEAR];
        assert!((near.normal - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4);
        assert!(near.signed_distance(Vec3::new(0.0, 0.0, -0.1)).abs() < 1e-3);

        assert!(f.contains_point(Vec3::new(0.0, 0.0, -1.0)));
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, 0.5))); // behind the near plane
        assert!(!f.contains_point(Vec3::new(2.0, 0.0, -1.0))); // outside the left/right walls
    }

    // --- contains_point ---------------------------------------------------

    #[test]
    fn frustum_contains_origin_and_rejects_outside_points() {
        let f = test_frustum();
        assert!(f.contains_point(Vec3::ZERO));
        assert!(f.contains_point(Vec3::new(0.0, 0.0, 4.0)));
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, 10.0)), "behind the camera");
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, 4.95)), "in front of near");
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, -200.0)), "beyond far");
        assert!(!f.contains_point(Vec3::new(100.0, 0.0, 0.0)), "far right");
        assert!(!f.contains_point(Vec3::new(-100.0, 0.0, 0.0)), "far left");
    }

    #[test]
    fn frustum_fov_edge_is_where_the_math_says() {
        // At z = 0 the eye is 5 units away, so the half-height is 5*tan(30deg).
        let f = test_frustum();
        let half_height = 5.0 * radians(30.0).tan(); // ~2.8868
        assert!(f.contains_point(Vec3::new(0.0, half_height * 0.99, 0.0)));
        assert!(!f.contains_point(Vec3::new(0.0, half_height * 1.01, 0.0)));
        // aspect = 1, so the horizontal edge is at the same offset.
        assert!(f.contains_point(Vec3::new(half_height * 0.99, 0.0, 0.0)));
        assert!(!f.contains_point(Vec3::new(half_height * 1.01, 0.0, 0.0)));
    }

    // --- intersects_aabb --------------------------------------------------

    #[test]
    fn aabb_at_origin_is_visible() {
        let f = test_frustum();
        assert!(f.intersects_aabb(&box_at(Vec3::ZERO, 0.5)));
    }

    #[test]
    fn aabb_behind_camera_is_culled() {
        let f = test_frustum();
        assert!(!f.intersects_aabb(&Aabb::new(
            Vec3::new(-1.0, -1.0, 20.0),
            Vec3::new(1.0, 1.0, 22.0)
        )));
    }

    #[test]
    fn aabb_beyond_far_plane_is_culled() {
        let f = test_frustum();
        assert!(!f.intersects_aabb(&box_at(Vec3::new(0.0, 0.0, -200.0), 1.0)));
    }

    #[test]
    fn aabb_off_to_the_sides_is_culled() {
        let f = test_frustum();
        for offset in [
            Vec3::new(-100.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(0.0, 100.0, 0.0),
            Vec3::new(0.0, -100.0, 0.0),
        ] {
            assert!(
                !f.intersects_aabb(&box_at(offset, 1.0)),
                "box at {} should be culled",
                offset
            );
        }
    }

    #[test]
    fn aabb_straddling_near_plane_is_kept() {
        // Near plane is at z = 4.9; this box spans 4.5..5.2 so part of it is
        // genuinely visible and it must NOT be culled.
        let f = test_frustum();
        let a = Aabb::new(Vec3::new(-0.1, -0.1, 4.5), Vec3::new(0.1, 0.1, 5.2));
        assert!(f.intersects_aabb(&a));
        // ...while a box entirely inside the near plane is culled.
        let too_close = Aabb::new(Vec3::new(-0.1, -0.1, 4.95), Vec3::new(0.1, 0.1, 5.2));
        assert!(!f.intersects_aabb(&too_close));
    }

    #[test]
    fn aabb_straddling_far_plane_is_kept() {
        let f = test_frustum();
        let a = Aabb::new(Vec3::new(-1.0, -1.0, -96.0), Vec3::new(1.0, 1.0, -94.0));
        assert!(f.intersects_aabb(&a));
    }

    #[test]
    fn huge_aabb_enclosing_frustum_is_visible() {
        let f = test_frustum();
        assert!(f.intersects_aabb(&box_at(Vec3::ZERO, 1000.0)));
        // Also when the box is centered well outside but still swallows the frustum.
        assert!(f.intersects_aabb(&Aabb::new(
            Vec3::new(-500.0, -500.0, -500.0),
            Vec3::new(500.0, 500.0, 500.0)
        )));
    }

    #[test]
    fn aabb_test_never_culls_a_visible_point() {
        // Conservativeness sweep: any point the frustum accepts must also be
        // accepted as a degenerate box, and as a small box around it.
        let f = test_frustum();
        let mut checked = 0;
        for xi in -6..=6 {
            for yi in -6..=6 {
                for zi in -20..=4 {
                    let p = Vec3::new(xi as f32 * 2.0, yi as f32 * 2.0, zi as f32 * 5.0);
                    if !f.contains_point(p) {
                        continue;
                    }
                    checked += 1;
                    assert!(f.intersects_aabb(&box_at(p, 0.0)), "degenerate box at {}", p);
                    assert!(f.intersects_aabb(&box_at(p, 0.25)), "small box at {}", p);
                    assert!(f.intersects_sphere(p, 0.0), "point sphere at {}", p);
                }
            }
        }
        assert!(checked > 20, "sweep covered too few interior points: {checked}");
    }

    #[test]
    fn aabb_with_inverted_bounds_is_not_falsely_culled() {
        // `Aabb`'s fields are public, so callers can build min > max. The
        // resulting negative half extents must not turn into a false negative
        // (invisible geometry); the box is still treated as covering the origin.
        let f = test_frustum();
        let inverted = Aabb {
            min: Vec3::splat(1.0),
            max: Vec3::splat(-1.0),
        };
        assert!(f.intersects_aabb(&inverted));
        // Same corners in the correct order agree.
        assert!(f.intersects_aabb(&Aabb::new(inverted.min, inverted.max)));
    }

    // --- intersects_sphere ------------------------------------------------

    #[test]
    fn sphere_at_origin_is_visible() {
        let f = test_frustum();
        assert!(f.intersects_sphere(Vec3::ZERO, 0.5));
        assert!(f.intersects_sphere(Vec3::ZERO, 0.0));
    }

    #[test]
    fn sphere_behind_camera_is_culled() {
        let f = test_frustum();
        assert!(!f.intersects_sphere(Vec3::new(0.0, 0.0, 20.0), 1.0));
    }

    #[test]
    fn sphere_beyond_far_plane_is_culled() {
        let f = test_frustum();
        assert!(!f.intersects_sphere(Vec3::new(0.0, 0.0, -200.0), 10.0));
    }

    #[test]
    fn sphere_radius_is_accounted_for() {
        let f = test_frustum();
        // Center is 0.6 behind the near plane (z = 5.5 vs near at 4.9).
        let center = Vec3::new(0.0, 0.0, 5.5);
        assert!(!f.contains_point(center));
        assert!(!f.intersects_sphere(center, 0.5), "too small to reach the near plane");
        assert!(f.intersects_sphere(center, 1.0), "big enough to cross the near plane");
    }

    #[test]
    fn sphere_just_outside_side_plane_is_culled() {
        let f = test_frustum();
        // 100 units to the right, radius 1 cannot reach back into the frustum.
        assert!(!f.intersects_sphere(Vec3::new(100.0, 0.0, 0.0), 1.0));
        // A radius that swallows the whole frustum does.
        assert!(f.intersects_sphere(Vec3::new(100.0, 0.0, 0.0), 500.0));
    }

    #[test]
    fn sphere_and_aabb_agree_on_clear_cases() {
        // The two tests use different math; they must not disagree when the
        // answer is unambiguous (well inside / far outside).
        let f = test_frustum();
        for (center, radius, expected) in [
            (Vec3::ZERO, 1.0, true),
            (Vec3::new(0.0, 0.0, 3.0), 0.5, true),
            (Vec3::new(0.0, 0.0, 50.0), 1.0, false),
            (Vec3::new(0.0, 300.0, 0.0), 1.0, false),
            (Vec3::new(0.0, 0.0, -500.0), 1.0, false),
        ] {
            assert_eq!(f.intersects_sphere(center, radius), expected, "sphere {center}");
            assert_eq!(f.intersects_aabb(&box_at(center, radius)), expected, "box {center}");
        }
    }

    #[test]
    fn frustum_from_planes_round_trips() {
        let f = test_frustum();
        let rebuilt = Frustum::from_planes(*f.planes());
        assert_eq!(rebuilt, f);
        assert!(rebuilt.contains_point(Vec3::ZERO));
    }
}
