// Primitive mesh generators: sphere, plane, etc.
// All generators produce MeshComponent with computed normals and vertex colors.

use crate::engine::geometry::Shape;
use crate::engine::math::{Vec2, Vec3};
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

            // Equirectangular unwrap: u sweeps longitude, v latitude. The seam
            // duplicate at lon == lon_segments gets u = 1.0 rather than wrapping
            // back to 0.0 — the whole reason the seam vertex is duplicated is so
            // interpolation runs 0.96 -> 1.0 there instead of 0.96 -> 0.0, which
            // would smear the entire texture backwards across one quad.
            let uv = Vec2::new(
                lon as f32 / lon_segments as f32,
                lat as f32 / lat_segments as f32,
            );

            vertices.push(MeshVertex {
                position,
                normal: Vec3::new(nx, ny, nz),
                color,
                uv,
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

    // Planar unwrap with tiling: one UV unit per two world units, so a texture
    // repeats rather than being stretched across the whole plane. A 50-unit
    // ground at one repeat total would smear any texture into unreadable blur —
    // and with a `Repeat` sampler, UVs past 1.0 are exactly how tiling works.
    let tile = |world: f32| world / 2.0;
    let vertices = vec![
        MeshVertex { position: Vec3::new(x - size, y, z - size), normal: Vec3::UNIT_Y, color,
                     uv: Vec2::new(tile(-size), tile(-size)) },
        MeshVertex { position: Vec3::new(x + size, y, z - size), normal: Vec3::UNIT_Y, color,
                     uv: Vec2::new(tile(size), tile(-size)) },
        MeshVertex { position: Vec3::new(x + size, y, z + size), normal: Vec3::UNIT_Y, color,
                     uv: Vec2::new(tile(size), tile(size)) },
        MeshVertex { position: Vec3::new(x - size, y, z + size), normal: Vec3::UNIT_Y, color,
                     uv: Vec2::new(tile(-size), tile(size)) },
    ];

    // Winding matters: the pipeline culls back faces with `front_face: Ccw`, so
    // the triangles must wind counter-clockwise when seen from +Y — i.e. their
    // cross-product normal has to agree with the +Y normal attribute above.
    // The naive 0,1,2 / 0,2,3 order produces a DOWNWARD geometric normal, which
    // made the ground plane invisible from any camera above it.
    let indices = vec![0, 2, 1, 0, 3, 2];

    MeshComponent::new(vertices, indices)
}

/// Generate a box mesh centred at the origin with half-extents `half`.
/// Placement belongs to the entity's transform.
///
/// 24 vertices (four per face, so each face keeps its flat normal) with a
/// per-face UV unwrap, oriented so v grows downward in texture space when the
/// face is viewed from outside. Per-face rather than a cross layout because a
/// repeating material (crate, brick) is what a box in a game actually wears;
/// a cross unwrap only pays off for hand-painted textures, which need an
/// artist's unwrap anyway.
///
/// UV density matches `plane`: one texture repeat per two world units, so the
/// same material tiles at the same scale on a floor and on a crate standing on
/// it. A face of a unit-half-extent box (2x2 units) gets exactly one repeat;
/// larger boxes tile via the Repeat sampler.
pub fn box_mesh(half: Vec3, color: Vec3) -> MeshComponent {
    let (hx, hy, hz) = (half.x, half.y, half.z);
    let p = Vec3::new;

    // Corner positions per face, wound CCW seen from outside (matching the
    // face normal), with UVs walking (0,v) -> (u,v) -> (u,0) -> (0,0) so the
    // texture is upright from outside; (u, v) are the face's world dimensions
    // over the 2-units-per-tile density.
    struct Face {
        corners: [Vec3; 4],
        normal: Vec3,
        /// Face size in world units along its UV axes.
        size: (f32, f32),
    }
    let faces = [
        // Front (z = -hz, normal -Z): seen from -Z, +X is to the *left*.
        Face { corners: [p(hx, -hy, -hz), p(-hx, -hy, -hz), p(-hx, hy, -hz), p(hx, hy, -hz)],
               normal: p(0.0, 0.0, -1.0), size: (2.0 * hx, 2.0 * hy) },
        // Back (z = +hz, normal +Z).
        Face { corners: [p(-hx, -hy, hz), p(hx, -hy, hz), p(hx, hy, hz), p(-hx, hy, hz)],
               normal: p(0.0, 0.0, 1.0), size: (2.0 * hx, 2.0 * hy) },
        // Left (x = -hx, normal -X).
        Face { corners: [p(-hx, -hy, -hz), p(-hx, -hy, hz), p(-hx, hy, hz), p(-hx, hy, -hz)],
               normal: p(-1.0, 0.0, 0.0), size: (2.0 * hz, 2.0 * hy) },
        // Right (x = +hx, normal +X).
        Face { corners: [p(hx, -hy, hz), p(hx, -hy, -hz), p(hx, hy, -hz), p(hx, hy, hz)],
               normal: p(1.0, 0.0, 0.0), size: (2.0 * hz, 2.0 * hy) },
        // Bottom (y = -hy, normal -Y).
        Face { corners: [p(-hx, -hy, -hz), p(hx, -hy, -hz), p(hx, -hy, hz), p(-hx, -hy, hz)],
               normal: p(0.0, -1.0, 0.0), size: (2.0 * hx, 2.0 * hz) },
        // Top (y = +hy, normal +Y).
        Face { corners: [p(-hx, hy, hz), p(hx, hy, hz), p(hx, hy, -hz), p(-hx, hy, -hz)],
               normal: p(0.0, 1.0, 0.0), size: (2.0 * hx, 2.0 * hz) },
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for face in &faces {
        let (u_max, v_max) = (face.size.0 / 2.0, face.size.1 / 2.0);
        let face_uvs = [
            Vec2::new(0.0, v_max),
            Vec2::new(u_max, v_max),
            Vec2::new(u_max, 0.0),
            Vec2::new(0.0, 0.0),
        ];
        let base = vertices.len() as u32;
        for (corner, uv) in face.corners.iter().zip(face_uvs.iter()) {
            vertices.push(MeshVertex {
                position: *corner,
                normal: face.normal,
                color,
                uv: *uv,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

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
    fn sphere_uv_is_an_equirectangular_unwrap() {
        let m = sphere(Vec3::ZERO, 1.0, Vec3::ONE, 8, 12);
        // Poles: v = 0 at the top ring, v = 1 at the bottom.
        assert_eq!(m.vertices.first().unwrap().uv.y, 0.0);
        assert_eq!(m.vertices.last().unwrap().uv.y, 1.0);
        // The seam duplicate must carry u = 1.0, not wrap back to 0.0 —
        // otherwise the last quad of every ring interpolates u from ~0.9 down
        // to 0 and smears the whole texture backwards across it.
        let stride = 12 + 1;
        for ring in 0..=8u32 {
            let first = m.vertices[(ring * stride) as usize].uv;
            let seam = m.vertices[(ring * stride + 12) as usize].uv;
            assert_eq!(first.x, 0.0);
            assert_eq!(seam.x, 1.0, "seam vertex must close the unwrap");
        }
        // All UVs in range.
        for v in &m.vertices {
            assert!((0.0..=1.0).contains(&v.uv.x) && (0.0..=1.0).contains(&v.uv.y));
        }
    }

    #[test]
    fn plane_uv_tiles_with_world_size() {
        // Half-extent 4 => 8 world units => 4 UV repeats at 2 units per tile.
        let m = plane(Vec3::ZERO, 4.0, Vec3::ONE);
        let us: Vec<f32> = m.vertices.iter().map(|v| v.uv.x).collect();
        let span = us.iter().fold(f32::MIN, |a, &b| a.max(b))
            - us.iter().fold(f32::MAX, |a, &b| a.min(b));
        assert!((span - 4.0).abs() < 1e-5, "expected 4 repeats, got {span}");
        // The centre offset must not shift the tiling density.
        let m2 = plane(Vec3::new(100.0, 0.0, -3.0), 4.0, Vec3::ONE);
        let us2: Vec<f32> = m2.vertices.iter().map(|v| v.uv.x).collect();
        let span2 = us2.iter().fold(f32::MIN, |a, &b| a.max(b))
            - us2.iter().fold(f32::MAX, |a, &b| a.min(b));
        assert!((span2 - 4.0).abs() < 1e-5);
    }

    #[test]
    fn box_mesh_has_a_full_per_face_unwrap() {
        let m = box_mesh(Vec3::new(1.0, 1.0, 1.0), Vec3::ONE);
        assert_eq!(m.vertices.len(), 24);
        assert_eq!(m.indices.len(), 36);
        // Every face must span the full 0..1 square.
        for face in m.vertices.chunks(4) {
            let mut us: Vec<f32> = face.iter().map(|v| v.uv.x).collect();
            let mut vs: Vec<f32> = face.iter().map(|v| v.uv.y).collect();
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!((us[0], us[3]), (0.0, 1.0), "face does not span u");
            assert_eq!((vs[0], vs[3]), (0.0, 1.0), "face does not span v");
        }
        // Each face's four corners are coplanar with its normal.
        for face in m.vertices.chunks(4) {
            let n = face[0].normal;
            let d = face[0].position.dot(n);
            for v in face {
                assert_eq!(v.normal, n);
                assert!((v.position.dot(n) - d).abs() < 1e-6);
            }
        }
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
            box_mesh(Vec3::new(1.0, 2.0, 0.5), Vec3::ONE),
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
