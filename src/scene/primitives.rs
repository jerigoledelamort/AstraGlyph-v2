// Primitive mesh generators: sphere, plane, etc.
// All generators produce MeshComponent with computed normals and vertex colors.

use crate::engine::geometry::Shape;
use crate::engine::math::Vec3;
use crate::scene::{MeshComponent, MeshVertex};

/// Standard constant for PI.
const PI: f32 = std::f32::consts::PI;

/// The analytic shape a `sphere` mesh approximates.
///
/// Paired with the generator rather than derived from the mesh afterwards: the
/// radius is known exactly here and only approximately from 1536 triangles, and
/// the CPU tracer and the physics collider both need the exact value. Deriving
/// it back out of the vertices would reproduce the tessellation error in every
/// consumer.
pub const fn sphere_shape(radius: f32) -> Shape {
    Shape::Sphere { radius }
}

/// The analytic shape a `plane` mesh of side `size` approximates.
pub const fn plane_shape(size: f32) -> Shape {
    Shape::Plane {
        normal: Vec3::UNIT_Y,
        half_size: size * 0.5,
    }
}

/// The analytic shape a box mesh of the given half-extents approximates.
pub const fn box_shape(half_extents: Vec3) -> Shape {
    Shape::Box { half_extents }
}

/// Generate a UV-sphere mesh centered at `center` with the given radius.
///
/// `lat_segments` controls vertical resolution (latitude bands).
/// `lon_segments` controls horizontal resolution (longitude slices).
/// `color` is the per-vertex color assigned to all vertices.
pub fn sphere(
    center: Vec3,
    radius: f32,
    color: Vec3,
    lat_segments: u32,
    lon_segments: u32,
) -> MeshComponent {
    // Minimum sane resolution.
    let lat_segments = lat_segments.max(3);
    let lon_segments = lon_segments.max(4);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Generate vertices: rings from top pole to bottom pole.
    for lat in 0..=lat_segments {
        let theta = PI * (lat as f32 / lat_segments as f32); // 0..PI (top..bottom)
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=lon_segments {
            let phi = 2.0 * PI * (lon as f32 / lon_segments as f32); // 0..2PI
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            // Normal direction (unit sphere).
            let nx = sin_theta * cos_phi;
            let ny = cos_theta;
            let nz = sin_theta * sin_phi;

            let position = Vec3::new(
                center.x + radius * nx,
                center.y + radius * ny,
                center.z + radius * nz,
            );

            vertices.push(MeshVertex {
                position,
                normal: Vec3::new(nx, ny, nz),
                color,
            });
        }
    }

    // Generate indices: connect rings with triangles.
    let stride = lon_segments + 1; // vertices per ring (including seam duplicate)
    for lat in 0..lat_segments {
        for lon in 0..lon_segments {
            let first = lat * stride + lon;
            let second = first + stride;

            // Two triangles per quad, wound so the cross-product normal points
            // OUTWARD (matching the per-vertex normals). The previous order was
            // inverted, so every sphere was rendered inside-out: back-face
            // culling kept only the far hemisphere and the lighting was computed
            // on surfaces facing away from the camera.
            indices.extend_from_slice(&[first, first + 1, second]);
            indices.extend_from_slice(&[second, first + 1, second + 1]);
        }
    }

    MeshComponent::new(vertices, indices)
}

/// Generate a horizontal plane mesh (on the XZ plane) centered at `center`.
///
/// `size` is the half-extent (total width = size * 2).
/// `color` is the per-vertex color.
/// Normal points up (+Y).
pub fn plane(center: Vec3, size: f32, color: Vec3) -> MeshComponent {
    let x = center.x;
    let y = center.y;
    let z = center.z;

    let vertices = vec![
        MeshVertex { position: Vec3::new(x - size, y, z - size), normal: Vec3::UNIT_Y, color },
        MeshVertex { position: Vec3::new(x + size, y, z - size), normal: Vec3::UNIT_Y, color },
        MeshVertex { position: Vec3::new(x + size, y, z + size), normal: Vec3::UNIT_Y, color },
        MeshVertex { position: Vec3::new(x - size, y, z + size), normal: Vec3::UNIT_Y, color },
    ];

    // Winding matters: the pipeline culls back faces with `front_face: Ccw`, so
    // the triangles must wind counter-clockwise when seen from +Y — i.e. their
    // cross-product normal has to agree with the +Y normal attribute above.
    // The naive 0,1,2 / 0,2,3 order produces a DOWNWARD geometric normal, which
    // made the ground plane invisible from any camera above it.
    let indices = vec![0, 2, 1, 0, 3, 2];

    MeshComponent::new(vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_vertex_count() {
        // (lat_segments + 1) rings * (lon_segments + 1) vertices per ring
        let m = sphere(Vec3::ZERO, 1.0, Vec3::ONE, 8, 12);
        let expected = (8 + 1) * (12 + 1);
        assert_eq!(m.vertices.len(), expected);
    }

    #[test]
    fn sphere_index_count() {
        // lat_segments * lon_segments quads * 2 triangles * 3 indices
        let m = sphere(Vec3::ZERO, 1.0, Vec3::ONE, 8, 12);
        let expected = 8 * 12 * 6;
        assert_eq!(m.indices.len(), expected);
    }

    #[test]
    fn sphere_normals_are_unit_length() {
        let m = sphere(Vec3::ZERO, 1.0, Vec3::ONE, 8, 12);
        for v in &m.vertices {
            let len = v.normal.length();
            assert!((len - 1.0).abs() < 1e-5, "normal not unit length: {len}");
        }
    }

    #[test]
    fn sphere_at_origin_has_expected_bounds() {
        let r = 2.5;
        let m = sphere(Vec3::ZERO, r, Vec3::ONE, 16, 32);
        for v in &m.vertices {
            let dist = v.position.length();
            assert!((dist - r).abs() < 1e-5, "vertex not on sphere surface: dist={dist}");
        }
    }

    #[test]
    fn sphere_offset_center() {
        let center = Vec3::new(10.0, 20.0, 30.0);
        let r = 1.0;
        let m = sphere(center, r, Vec3::ONE, 8, 12);
        for v in &m.vertices {
            let dist = (v.position - center).length();
            assert!((dist - r).abs() < 1e-5, "vertex not on offset sphere: dist={dist}");
        }
    }

    #[test]
    fn sphere_min_resolution_clamped() {
        let m = sphere(Vec3::ZERO, 1.0, Vec3::ONE, 1, 1);
        // Should still produce a valid mesh (clamped to 3 lat, 4 lon).
        assert!(m.vertices.len() > 0);
        assert!(m.indices.len() > 0);
    }

    #[test]
    fn plane_vertex_and_index_count() {
        let m = plane(Vec3::ZERO, 10.0, Vec3::ONE);
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.indices.len(), 6);
    }

    #[test]
    fn plane_normal_points_up() {
        let m = plane(Vec3::ZERO, 10.0, Vec3::ONE);
        for v in &m.vertices {
            assert_eq!(v.normal, Vec3::UNIT_Y);
        }
    }

    /// Geometric winding must agree with the normal attribute.
    ///
    /// This is the check that was missing: `plane_normal_points_up` only looks at
    /// the normal *attribute*, so a mesh wound the wrong way passed it happily
    /// while being culled away by the back-face test at render time.
    #[test]
    fn every_primitive_triangle_winds_to_match_its_normal() {
        for mesh in [
            plane(Vec3::ZERO, 10.0, Vec3::ONE),
            plane(Vec3::new(1.0, -2.0, 3.0), 4.0, Vec3::ONE),
            sphere(Vec3::ZERO, 1.0, Vec3::ONE, 6, 8),
            sphere(Vec3::new(2.0, 0.0, -1.0), 2.5, Vec3::ONE, 4, 5),
        ] {
            assert_eq!(mesh.indices.len() % 3, 0, "indices must form whole triangles");
            for tri in mesh.indices.chunks(3) {
                let a = mesh.vertices[tri[0] as usize];
                let b = mesh.vertices[tri[1] as usize];
                let c = mesh.vertices[tri[2] as usize];

                let geometric = (b.position - a.position).cross(c.position - a.position);
                if geometric.length_squared() < 1e-12 {
                    continue; // degenerate triangle (sphere poles) — no winding to check
                }
                let geometric = geometric.normalize();
                // Average the attribute normals of the triangle's corners.
                let attribute = (a.normal + b.normal + c.normal).normalize();

                assert!(
                    geometric.dot(attribute) > 0.0,
                    "triangle {tri:?} winds against its normal: geometric {geometric}, attribute {attribute}"
                );
            }
        }
    }
}
