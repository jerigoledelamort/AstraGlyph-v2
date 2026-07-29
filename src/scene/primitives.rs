// Primitive mesh generators: sphere, plane, etc.
// All generators produce MeshComponent with computed normals and vertex colors.

use crate::engine::math::Vec3;
use crate::scene::{MeshComponent, MeshVertex};

/// Standard constant for PI.
const PI: f32 = std::f32::consts::PI;

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

            // Two triangles per quad (CCW winding when viewed from outside).
            indices.extend_from_slice(&[first, second, first + 1]);
            indices.extend_from_slice(&[second, second + 1, first + 1]);
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

    let indices = vec![0, 1, 2, 0, 2, 3];

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
}
