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
    /// Only the parts of a transform an analytic shape can absorb are applied:
    /// translation moves the origin, and scale changes the extents. A rotation
    /// applied to a sphere is a no-op; applied to a box or plane it is *not*
    /// representable, so the box stays axis-aligned and the plane's normal is
    /// rotated while its extent is not sheared. The approximation is documented
    /// rather than hidden because the alternative — silently rotating extents —
    /// produces colliders that do not match the drawn geometry.
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
        WorldShape { origin, shape }
    }
}

/// A shape positioned in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldShape {
    /// World-space origin: the centre for spheres and boxes, a point on the
    /// surface for planes.
    pub origin: Vec3,
    /// The shape itself, with world-space extents.
    pub shape: Shape,
}

impl WorldShape {
    pub fn new(origin: Vec3, shape: Shape) -> Self {
        Self { origin, shape }
    }

    /// Conservative world-space bounding radius around `origin`.
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
