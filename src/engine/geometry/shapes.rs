// Analytic shapes: the exact forms behind the engine's triangle meshes.
//
// A mesh is an approximation. A sphere built from 24 rings and 32 segments is
// 1536 triangles that *almost* satisfy |p - c| = r, and both of this module's
// consumers want the equation rather than the approximation:
//
// - The CPU fallback tracer (Phase 4.4) intersects rays analytically, because
//   walking 1536 triangles per ray on a CPU is not viable at interactive rates
//   while solving one quadratic is.
// - Physics (Phase 5.1) needs collision volumes, and a sphere-sphere test is
//   one distance comparison where a mesh-mesh test is a research project.
//
// Keeping one definition for both is the point: a collider that disagrees with
// what the tracer intersects would mean objects visibly reflecting somewhere
// other than where they collide.

use crate::engine::math::{Mat4, Vec3};

/// An analytic volume or surface, in whatever space its owner keeps it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// A sphere of `radius` centred on the shape's origin.
    Sphere { radius: f32 },
    /// An axis-aligned box reaching `half_extents` from its centre in each axis.
    Box { half_extents: Vec3 },
    /// A finite patch of an infinite plane: `normal` is the outward unit normal,
    /// `half_size` the extent from the centre along the two in-plane axes.
    ///
    /// Finite rather than infinite because the engine's ground *is* finite (a
    /// 50-unit plane), and an infinite ground would put a horizon in every
    /// reflection where the real scene has an edge.
    Plane { normal: Vec3, half_size: f32 },
}

impl Shape {
    /// A conservative bounding radius around the shape's origin.
    ///
    /// Used to reject a ray or a collision pair cheaply before doing exact work.
    pub fn bounding_radius(&self) -> f32 {
        match self {
            Self::Sphere { radius } => radius.abs(),
            Self::Box { half_extents } => half_extents.length(),
            Self::Plane { half_size, .. } => half_size.abs() * std::f32::consts::SQRT_2,
        }
    }

    /// The shape placed in world space by `model`.
    ///
    /// Translation moves the origin, scale changes the extents, and rotation goes
    /// into the resulting shape's `basis` — so a rotated box really is an
    /// oriented box rather than a silently axis-aligned one.
    ///
    /// The one thing still approximated is non-uniform scale on a sphere: this
    /// shape set has no ellipsoid, so the largest axis is taken and the volume
    /// encloses the drawn mesh instead of matching it. Documented rather than
    /// hidden, because a collider that quietly disagrees with the geometry is
    /// how "the reflection is in the wrong place" bugs start.
    pub fn transformed(&self, model: &Mat4) -> WorldShape {
        let origin = model.transform_point(Vec3::ZERO);
        // Column lengths of the upper 3x3 are the per-axis scale factors.
        let sx = Vec3::new(model.m[0], model.m[1], model.m[2]).length();
        let sy = Vec3::new(model.m[4], model.m[5], model.m[6]).length();
        let sz = Vec3::new(model.m[8], model.m[9], model.m[10]).length();
        let shape = match *self {
            Self::Sphere { radius } => Self::Sphere {
                // Non-uniform scale would make an ellipsoid, which this shape
                // set cannot express; the largest axis keeps the volume
                // enclosing rather than intersecting the drawn mesh.
                radius: radius * sx.max(sy).max(sz),
            },
            Self::Box { half_extents } => Self::Box {
                half_extents: Vec3::new(
                    half_extents.x * sx,
                    half_extents.y * sy,
                    half_extents.z * sz,
                ),
            },
            Self::Plane { normal, half_size } => Self::Plane {
                normal: model.transform_dir(normal).normalize(),
                half_size: half_size * sx.max(sz),
            },
        };
        WorldShape {
            origin,
            shape,
            basis: Basis::from_matrix(model),
        }
    }
}

/// An orthonormal rotation, stored as its three column vectors.
///
/// A 3x3 rotation rather than a full `Mat4` because that is all a shape's
/// orientation is, and because the two operations that matter — rotating a
/// direction into the shape's local frame and back out — are a transpose and a
/// multiply on exactly these three vectors. Reusing `Mat4` would invite the
/// translation to come along, which for a shape is already in `origin`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Basis {
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
}

impl Basis {
    pub const IDENTITY: Self = Self {
        x: Vec3::UNIT_X,
        y: Vec3::UNIT_Y,
        z: Vec3::UNIT_Z,
    };

    /// The rotation part of a model matrix, with the scale divided out.
    ///
    /// Scale is deliberately dropped: it belongs to the shape's extents, which
    /// `Shape::transformed` handles. Leaving it in the basis would scale the
    /// geometry twice.
    pub fn from_matrix(m: &Mat4) -> Self {
        let col = |i: usize| Vec3::new(m.m[i * 4], m.m[i * 4 + 1], m.m[i * 4 + 2]);
        let normalize_or = |v: Vec3, fallback: Vec3| {
            if v.length_squared() > 1e-12 {
                v.normalize()
            } else {
                fallback
            }
        };
        Self {
            x: normalize_or(col(0), Vec3::UNIT_X),
            y: normalize_or(col(1), Vec3::UNIT_Y),
            z: normalize_or(col(2), Vec3::UNIT_Z),
        }
    }

    /// One of the three axes, by index. Out-of-range indices give Z rather than
    /// panicking, because the callers are loops over `0..3`.
    pub fn axis(&self, index: usize) -> Vec3 {
        match index {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    /// Rotate a world-space vector into the shape's local frame.
    pub fn to_local(&self, v: Vec3) -> Vec3 {
        // Transpose of an orthonormal matrix is its inverse, so this is a
        // dot product against each column.
        Vec3::new(v.dot(self.x), v.dot(self.y), v.dot(self.z))
    }

    /// Rotate a local-space vector into world space.
    pub fn to_world(&self, v: Vec3) -> Vec3 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// Whether this is (numerically) the identity, so callers can skip the
    /// rotation work entirely for the common unrotated case.
    pub fn is_identity(&self) -> bool {
        (self.x - Vec3::UNIT_X).length_squared() < 1e-12
            && (self.y - Vec3::UNIT_Y).length_squared() < 1e-12
            && (self.z - Vec3::UNIT_Z).length_squared() < 1e-12
    }
}

impl Default for Basis {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A shape positioned and oriented in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldShape {
    /// World-space origin: the centre for spheres and boxes, a point on the
    /// surface for planes.
    pub origin: Vec3,
    /// The shape itself, with world-space extents.
    pub shape: Shape,
    /// Orientation. A box with a non-identity basis is an oriented box (OBB);
    /// a sphere ignores it.
    pub basis: Basis,
}

impl WorldShape {
    /// An unrotated shape at `origin`.
    pub fn new(origin: Vec3, shape: Shape) -> Self {
        Self {
            origin,
            shape,
            basis: Basis::IDENTITY,
        }
    }

    /// A shape with an explicit orientation.
    pub fn oriented(origin: Vec3, shape: Shape, basis: Basis) -> Self {
        Self {
            origin,
            shape,
            basis,
        }
    }

    /// Conservative world-space bounding radius around `origin`. Rotation does
    /// not change it, which is exactly why it is the right early rejection test.
    pub fn bounding_radius(&self) -> f32 {
        self.shape.bounding_radius()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_bounding_radius_is_its_radius() {
        assert_eq!(Shape::Sphere { radius: 2.5 }.bounding_radius(), 2.5);
    }

    #[test]
    fn box_bounding_radius_reaches_the_corner() {
        let s = Shape::Box {
            half_extents: Vec3::new(1.0, 2.0, 2.0),
        };
        assert!((s.bounding_radius() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn translation_moves_the_origin_and_leaves_the_radius_alone() {
        let s = Shape::Sphere { radius: 1.0 };
        let w = s.transformed(&Mat4::translation(3.0, -1.0, 2.0));
        assert_eq!(w.origin, Vec3::new(3.0, -1.0, 2.0));
        assert_eq!(w.shape, Shape::Sphere { radius: 1.0 });
    }

    /// The demo scene scales its spheres by 1.5, so a scale that does not reach
    /// the analytic radius would put every traced reflection at the wrong size.
    #[test]
    fn uniform_scale_scales_the_radius() {
        let s = Shape::Sphere { radius: 1.0 };
        let w = s.transformed(&Mat4::scaling_uniform(1.5));
        assert_eq!(w.shape, Shape::Sphere { radius: 1.5 });
    }

    #[test]
    fn non_uniform_scale_takes_the_largest_axis_so_the_volume_encloses() {
        let s = Shape::Sphere { radius: 1.0 };
        let w = s.transformed(&Mat4::scaling(1.0, 3.0, 2.0));
        assert_eq!(w.shape, Shape::Sphere { radius: 3.0 });
    }

    #[test]
    fn box_scale_is_per_axis() {
        let s = Shape::Box {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        let w = s.transformed(&Mat4::scaling(2.0, 3.0, 4.0));
        assert_eq!(
            w.shape,
            Shape::Box {
                half_extents: Vec3::new(2.0, 3.0, 4.0)
            }
        );
    }

    #[test]
    fn plane_normal_follows_rotation() {
        let s = Shape::Plane {
            normal: Vec3::UNIT_Y,
            half_size: 10.0,
        };
        // Rotating about Y leaves the Y normal untouched...
        let w = s.transformed(&Mat4::rotation_y(1.0));
        match w.shape {
            Shape::Plane { normal, .. } => {
                assert!((normal - Vec3::UNIT_Y).length() < 1e-5, "got {normal}")
            }
            other => panic!("expected a plane, got {other:?}"),
        }
        // ...but a quarter turn about X must tip it onto -Z.
        let w = s.transformed(&Mat4::rotation_x(std::f32::consts::FRAC_PI_2));
        match w.shape {
            Shape::Plane { normal, .. } => {
                assert!(normal.y.abs() < 1e-5, "normal should have left +Y: {normal}");
                assert!(normal.z.abs() > 0.99, "normal should be along Z: {normal}");
            }
            other => panic!("expected a plane, got {other:?}"),
        }
    }

    /// A transform combining translation and scale must apply both — the demo
    /// scene uses exactly that, and getting only one of them would place traced
    /// geometry somewhere the rasteriser did not draw it.
    #[test]
    fn translation_and_scale_compose() {
        let model = Mat4::translation(2.2, -0.5, -2.0).mul(Mat4::scaling_uniform(1.5));
        let w = Shape::Sphere { radius: 1.0 }.transformed(&model);
        assert!((w.origin - Vec3::new(2.2, -0.5, -2.0)).length() < 1e-5);
        assert_eq!(w.shape, Shape::Sphere { radius: 1.5 });
    }
}
