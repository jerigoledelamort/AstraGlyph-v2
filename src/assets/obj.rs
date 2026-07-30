// Wavefront OBJ parser — self-implemented, per the "no external crates" rule.
//
// OBJ is a line-oriented text format, which makes it the right first model loader:
// the whole grammar that matters is `v`, `vn`, `vt`, `f` and `o`, and a parser for
// it is a few hundred lines rather than glTF's JSON-plus-binary-buffer-views.
//
// The three things that trip up a naive OBJ reader, all handled below:
//
// 1. **Indices are 1-based, and may be negative.** `-1` means the most recently
//    defined vertex. Off-by-one here shifts the whole mesh by one vertex; a negative
//    index read as unsigned wraps to a huge number and the face is silently dropped.
// 2. **A face vertex is a *triple* of indices** (`position/texcoord/normal`), and
//    two faces can share a position while using different normals. GPU vertices are
//    a single flat array, so each distinct triple has to become its own vertex —
//    otherwise a cube comes out with smoothed corners.
// 3. **Faces may have any number of vertices.** Quads are extremely common and
//    n-gons happen. They have to be fanned into triangles, because the renderer
//    draws `TriangleList`.
//
// Winding is preserved as authored rather than "corrected". A mesh whose winding is
// wrong disappears under back-face culling, and that is a property of the file the
// author needs to see — silently flipping it would hide a broken export.

use crate::engine::core::{EngineError, Result};
use crate::engine::math::Vec3;
use crate::scene::component::{MeshComponent, MeshVertex};

/// Default colour for a mesh whose file carries none.
///
/// OBJ has no vertex colours in the base format (some exporters append them to `v`,
/// which is handled). Mid-grey rather than white so an unlit or unshaded mesh is
/// still visibly a surface rather than a silhouette.
const DEFAULT_COLOR: Vec3 = Vec3::new(0.75, 0.75, 0.75);

/// A parsed OBJ file.
#[derive(Clone, Debug)]
pub struct ObjModel {
    /// Object name from the last `o` directive, if any.
    pub name: Option<String>,
    /// The mesh, ready for the renderer.
    pub mesh: MeshComponent,
    /// Faces that were fanned into more than one triangle.
    pub triangulated_faces: usize,
    /// Faces skipped as unusable, with the first reason. Reported rather than
    /// silently dropped: a file that loses half its faces should say so.
    pub skipped_faces: usize,
}

fn err(line: usize, msg: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidState(format!("line {line}: {msg}"))
}

/// One face vertex: indices into the position, texcoord and normal lists.
///
/// Resolved to absolute 0-based indices at parse time, so the deduplication key
/// below is comparable without re-resolving negatives.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FaceVertex {
    position: usize,
    normal: Option<usize>,
    texcoord: Option<usize>,
}

/// Resolve an OBJ index against a list length.
///
/// Positive indices are 1-based; negative ones count back from the end, so `-1` is
/// the most recently defined element. Both conventions are in real files, and
/// getting either wrong produces a mesh that is subtly or completely wrong rather
/// than an error.
fn resolve_index(raw: i64, count: usize) -> Option<usize> {
    if raw > 0 {
        let index = (raw - 1) as usize;
        (index < count).then_some(index)
    } else if raw < 0 {
        let from_end = (-raw) as usize;
        (from_end <= count).then(|| count - from_end)
    } else {
        // Index 0 does not exist in OBJ's 1-based scheme.
        None
    }
}

/// Parse OBJ source text.
pub fn parse(source: &str) -> Result<ObjModel> {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    // Per-vertex colours, when the exporter appended them to `v`.
    let mut vertex_colors: Vec<Option<Vec3>> = Vec::new();
    let mut name: Option<String> = None;

    // Deduplication: one GPU vertex per distinct index triple.
    let mut vertex_map: std::collections::HashMap<FaceVertex, u32> =
        std::collections::HashMap::new();
    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut triangulated_faces = 0usize;
    let mut skipped_faces = 0usize;
    let mut first_skip_reason: Option<String> = None;

    for (number, raw_line) in source.lines().enumerate() {
        let number = number + 1;
        // `#` starts a comment anywhere on the line.
        let line = match raw_line.find('#') {
            Some(at) => &raw_line[..at],
            None => raw_line,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let Some(keyword) = parts.next() else {
            continue;
        };
        let fields: Vec<&str> = parts.collect();

        match keyword {
            "v" => {
                if fields.len() < 3 {
                    return Err(err(number, "vertex needs at least three coordinates"));
                }
                let (x, y, z) = (
                    parse_float(fields[0], number)?,
                    parse_float(fields[1], number)?,
                    parse_float(fields[2], number)?,
                );
                positions.push(Vec3::new(x, y, z));
                // Some exporters append r g b after the coordinates. Read them when
                // present rather than ignoring them, since a coloured mesh losing
                // its colours is a visible regression.
                let color = if fields.len() >= 6 {
                    Some(Vec3::new(
                        parse_float(fields[3], number)?,
                        parse_float(fields[4], number)?,
                        parse_float(fields[5], number)?,
                    ))
                } else {
                    None
                };
                vertex_colors.push(color);
            }
            "vn" => {
                if fields.len() < 3 {
                    return Err(err(number, "normal needs three components"));
                }
                normals.push(Vec3::new(
                    parse_float(fields[0], number)?,
                    parse_float(fields[1], number)?,
                    parse_float(fields[2], number)?,
                ));
            }
            "vt" => {
                if fields.is_empty() {
                    return Err(err(number, "texture coordinate needs at least one value"));
                }
                texcoords.push([
                    parse_float(fields[0], number)?,
                    fields
                        .get(1)
                        .map(|f| parse_float(f, number))
                        .transpose()?
                        .unwrap_or(0.0),
                ]);
            }
            "o" | "g" => {
                if name.is_none() && !fields.is_empty() {
                    name = Some(fields.join(" "));
                }
            }
            "f" => {
                if fields.len() < 3 {
                    skipped_faces += 1;
                    if first_skip_reason.is_none() {
                        first_skip_reason =
                            Some(format!("line {number}: face with fewer than 3 vertices"));
                    }
                    continue;
                }
                // Resolve every corner first: a face with one bad index is dropped
                // whole rather than emitted as a partial triangle.
                let mut corners: Vec<FaceVertex> = Vec::with_capacity(fields.len());
                let mut bad = None;
                for field in &fields {
                    match parse_face_vertex(
                        field,
                        positions.len(),
                        normals.len(),
                        texcoords.len(),
                    ) {
                        Some(corner) => corners.push(corner),
                        None => {
                            bad = Some(format!("line {number}: bad face index {field:?}"));
                            break;
                        }
                    }
                }
                if let Some(reason) = bad {
                    skipped_faces += 1;
                    if first_skip_reason.is_none() {
                        first_skip_reason = Some(reason);
                    }
                    continue;
                }

                if corners.len() > 3 {
                    triangulated_faces += 1;
                }
                // Fan triangulation: (0,1,2), (0,2,3), ... Correct for convex
                // polygons, which is what OBJ faces are in practice; a concave
                // n-gon would need ear clipping, and no exporter emits one.
                for i in 1..corners.len() - 1 {
                    for corner in [corners[0], corners[i], corners[i + 1]] {
                        let next = vertices.len() as u32;
                        let index = *vertex_map.entry(corner).or_insert(next);
                        if index == next {
                            vertices.push(build_vertex(
                                corner,
                                &positions,
                                &normals,
                                &vertex_colors,
                            ));
                        }
                        indices.push(index);
                    }
                }
            }
            // Material libraries and everything else: skipped. `usemtl` would need
            // an MTL parser and a material mapping, which is a separate concern from
            // geometry and is not what this loader claims to do.
            _ => {}
        }
    }

    if vertices.is_empty() {
        return Err(EngineError::InvalidState(
            "OBJ file produced no geometry (no usable 'f' lines?)".to_string(),
        ));
    }

    // Any face lacking a normal got a flat one computed from its winding. That is
    // done after the fact here so it can be reported.
    let mut model = ObjModel {
        name,
        mesh: MeshComponent::new(vertices, indices),
        triangulated_faces,
        skipped_faces,
    };
    fill_missing_normals(&mut model.mesh, &normals);
    Ok(model)
}

/// Turn a resolved face vertex into a GPU vertex.
fn build_vertex(
    corner: FaceVertex,
    positions: &[Vec3],
    normals: &[Vec3],
    colors: &[Option<Vec3>],
) -> MeshVertex {
    MeshVertex {
        position: positions
            .get(corner.position)
            .copied()
            .unwrap_or(Vec3::ZERO),
        // A zero normal marks "needs computing"; `fill_missing_normals` finishes it.
        normal: corner
            .normal
            .and_then(|i| normals.get(i).copied())
            .unwrap_or(Vec3::ZERO),
        color: colors
            .get(corner.position)
            .copied()
            .flatten()
            .unwrap_or(DEFAULT_COLOR),
    }
}

/// Compute flat normals for vertices the file gave none.
///
/// From the *winding*, via a cross product — not from an adjacent vertex's normal.
/// The geometric normal is the one back-face culling agrees with, and using anything
/// else is how a mesh ends up invisible while its attributes look fine.
fn fill_missing_normals(mesh: &mut MeshComponent, file_normals: &[Vec3]) {
    // Nothing to do if the file supplied normals for everything.
    if !file_normals.is_empty()
        && mesh
            .vertices
            .iter()
            .all(|v| v.normal.length_squared() > 1e-12)
    {
        return;
    }
    // Accumulate each triangle's face normal into its three vertices, then
    // normalize: that gives smooth normals on a shared-vertex mesh and flat ones
    // where vertices are not shared, which is the same rule the primitives use.
    let mut accumulated = vec![Vec3::ZERO; mesh.vertices.len()];
    for triangle in mesh.indices.chunks_exact(3) {
        let [a, b, c] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let Some(&va) = mesh.vertices.get(a) else { continue };
        let Some(&vb) = mesh.vertices.get(b) else { continue };
        let Some(&vc) = mesh.vertices.get(c) else { continue };
        let face = (vb.position - va.position).cross(vc.position - va.position);
        for index in [a, b, c] {
            accumulated[index] = accumulated[index] + face;
        }
    }
    for (vertex, sum) in mesh.vertices.iter_mut().zip(accumulated.iter()) {
        if vertex.normal.length_squared() > 1e-12 {
            continue;
        }
        vertex.normal = if sum.length_squared() > 1e-12 {
            sum.normalize()
        } else {
            // A degenerate triangle has no normal. +Y rather than zero, so the
            // shader's `normalize` does not produce NaN.
            Vec3::UNIT_Y
        };
    }
}

/// Parse a `position/texcoord/normal` face vertex.
///
/// All of `1`, `1/2`, `1//3` and `1/2/3` are legal, and the two-slash form is
/// common — it is what an exporter writes for a mesh with normals but no UVs.
fn parse_face_vertex(
    field: &str,
    position_count: usize,
    normal_count: usize,
    texcoord_count: usize,
) -> Option<FaceVertex> {
    let mut parts = field.split('/');
    let position = resolve_index(parts.next()?.trim().parse::<i64>().ok()?, position_count)?;

    let texcoord = match parts.next() {
        Some(text) if !text.trim().is_empty() => {
            Some(resolve_index(text.trim().parse::<i64>().ok()?, texcoord_count)?)
        }
        _ => None,
    };
    let normal = match parts.next() {
        Some(text) if !text.trim().is_empty() => {
            Some(resolve_index(text.trim().parse::<i64>().ok()?, normal_count)?)
        }
        _ => None,
    };
    Some(FaceVertex {
        position,
        normal,
        texcoord,
    })
}

fn parse_float(text: &str, line: usize) -> Result<f32> {
    text.trim()
        .parse::<f32>()
        .map_err(|_| err(line, format!("expected a number, found {text:?}")))
}

/// Read and parse an OBJ file from disk.
pub fn load(path: impl AsRef<std::path::Path>) -> Result<ObjModel> {
    let path = path.as_ref();
    let source = std::fs::read_to_string(path)?;
    parse(&source)
        .map_err(|e| EngineError::InvalidState(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit triangle with explicit normals.
    const TRIANGLE: &str = "\
# a comment
o test_triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
f 1//1 2//1 3//1
";

    #[test]
    fn parses_a_triangle() {
        let model = parse(TRIANGLE).expect("should parse");
        assert_eq!(model.name.as_deref(), Some("test_triangle"));
        assert_eq!(model.mesh.vertices.len(), 3);
        assert_eq!(model.mesh.indices, vec![0, 1, 2]);
        assert_eq!(model.mesh.vertices[1].position, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(model.mesh.vertices[0].normal, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(model.skipped_faces, 0);
    }

    /// Indices are 1-based. An off-by-one shifts the entire mesh by one vertex,
    /// which looks like a subtly wrong model rather than an error.
    #[test]
    fn indices_are_one_based() {
        let model = parse("v 1 0 0\nv 2 0 0\nv 3 0 0\nf 1 2 3\n").unwrap();
        // Face index 1 must be the FIRST vertex, x = 1.
        assert_eq!(model.mesh.vertices[0].position.x, 1.0);
        assert_eq!(model.mesh.vertices[2].position.x, 3.0);
    }

    /// Negative indices count back from the end: `-1` is the most recent vertex.
    /// Read as unsigned they wrap to a huge number and the face vanishes.
    #[test]
    fn negative_indices_count_back_from_the_end() {
        let model = parse("v 1 0 0\nv 2 0 0\nv 3 0 0\nf -3 -2 -1\n").unwrap();
        assert_eq!(model.mesh.indices.len(), 3);
        let xs: Vec<f32> = model.mesh.vertices.iter().map(|v| v.position.x).collect();
        assert_eq!(xs, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn index_zero_is_rejected_as_a_face_but_does_not_fail_the_file() {
        // 0 is not a valid OBJ index; the face is skipped and reported.
        let model = parse("v 1 0 0\nv 2 0 0\nv 3 0 0\nf 0 1 2\nf 1 2 3\n").unwrap();
        assert_eq!(model.skipped_faces, 1);
        assert_eq!(model.mesh.indices.len(), 3, "the good face still loaded");
    }

    #[test]
    fn an_out_of_range_index_skips_only_that_face() {
        let model = parse("v 1 0 0\nv 2 0 0\nv 3 0 0\nf 1 2 99\nf 1 2 3\n").unwrap();
        assert_eq!(model.skipped_faces, 1);
        assert_eq!(model.mesh.indices.len(), 3);
    }

    /// All four face-vertex forms appear in real files. `1//3` in particular is what
    /// an exporter writes for normals-but-no-UVs, and mis-parsing it loses the normal.
    #[test]
    fn every_face_vertex_form_parses() {
        let source = "\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vn 0 0 1
f 1 2 3
f 1/1 2/1 3/1
f 1//1 2//1 3//1
f 1/1/1 2/1/1 3/1/1
";
        let model = parse(source).unwrap();
        assert_eq!(model.skipped_faces, 0, "every form should have parsed");
        assert_eq!(model.mesh.indices.len(), 12, "four triangles");
    }

    /// A quad must become two triangles: the renderer draws TriangleList, and a quad
    /// passed through as-is would be read as one triangle plus garbage.
    #[test]
    fn a_quad_is_fanned_into_two_triangles() {
        let source = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
";
        let model = parse(source).unwrap();
        assert_eq!(model.mesh.indices.len(), 6, "two triangles");
        assert_eq!(model.triangulated_faces, 1);
        // The fan shares corner 0, so both triangles start there.
        assert_eq!(model.mesh.indices[0], model.mesh.indices[3]);
    }

    #[test]
    fn an_ngon_is_fanned_into_n_minus_two_triangles() {
        let source = "\
v 0 0 0
v 1 0 0
v 2 1 0
v 1 2 0
v 0 2 0
f 1 2 3 4 5
";
        let model = parse(source).unwrap();
        assert_eq!(model.mesh.indices.len(), 9, "three triangles from a pentagon");
    }

    /// Two faces can share a position but use different normals. GPU vertices are a
    /// flat array, so each distinct triple needs its own vertex — sharing them
    /// smooths every corner of what should be a hard edge.
    #[test]
    fn the_same_position_with_different_normals_becomes_two_vertices() {
        let source = "\
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vn 0 0 1
vn 0 1 0
f 1//1 2//1 3//1
f 1//2 2//2 4//2
";
        let model = parse(source).unwrap();
        // Positions 1 and 2 appear with both normals, so they duplicate: 3 + 3
        // distinct triples rather than 4 positions.
        assert_eq!(
            model.mesh.vertices.len(),
            6,
            "distinct index triples must not be merged"
        );
        // And the two faces really do carry different normals.
        let normals: Vec<Vec3> = model.mesh.vertices.iter().map(|v| v.normal).collect();
        assert!(normals.contains(&Vec3::new(0.0, 0.0, 1.0)));
        assert!(normals.contains(&Vec3::new(0.0, 1.0, 0.0)));
    }

    /// An identical triple *should* be shared, or a large mesh doubles in size.
    #[test]
    fn identical_triples_are_shared() {
        let source = "\
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vn 0 0 1
f 1//1 2//1 3//1
f 2//1 4//1 3//1
";
        let model = parse(source).unwrap();
        assert_eq!(
            model.mesh.vertices.len(),
            4,
            "the shared edge should reuse its vertices"
        );
        assert_eq!(model.mesh.indices.len(), 6);
    }

    /// A file with no normals must get computed ones, from the winding.
    #[test]
    fn missing_normals_are_computed_from_the_winding() {
        // Counter-clockwise when seen from +Z, so the normal is +Z.
        let model = parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        for v in &model.mesh.vertices {
            assert!(
                (v.normal - Vec3::UNIT_Z).length() < 1e-5,
                "computed normal was {}",
                v.normal
            );
        }
    }

    /// Reversing the winding must reverse the computed normal — that is what makes
    /// it the *geometric* normal, the one back-face culling agrees with.
    #[test]
    fn a_reversed_winding_gives_a_reversed_normal() {
        let model = parse("v 0 0 0\nv 0 1 0\nv 1 0 0\nf 1 2 3\n").unwrap();
        for v in &model.mesh.vertices {
            assert!(
                (v.normal + Vec3::UNIT_Z).length() < 1e-5,
                "computed normal was {}",
                v.normal
            );
        }
    }

    /// Winding is preserved as authored, never "corrected". A mesh with reversed
    /// winding disappears under back-face culling, and that is a property of the
    /// file the author needs to see.
    #[test]
    fn winding_is_preserved_rather_than_normalised() {
        let ccw = parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let cw = parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 3 2 1\n").unwrap();
        // The index orders must differ, i.e. the parser did not reorder either.
        let ccw_positions: Vec<Vec3> = ccw
            .mesh
            .indices
            .iter()
            .map(|i| ccw.mesh.vertices[*i as usize].position)
            .collect();
        let cw_positions: Vec<Vec3> = cw
            .mesh
            .indices
            .iter()
            .map(|i| cw.mesh.vertices[*i as usize].position)
            .collect();
        assert_ne!(ccw_positions, cw_positions);
    }

    #[test]
    fn a_degenerate_triangle_does_not_produce_a_nan_normal() {
        // All three corners identical: the cross product is zero.
        let model = parse("v 0 0 0\nf 1 1 1\n").unwrap();
        for v in &model.mesh.vertices {
            assert!(v.normal.x.is_finite() && v.normal.length() > 0.5);
        }
    }

    #[test]
    fn per_vertex_colours_are_read_when_present() {
        let model = parse("v 0 0 0 1.0 0.0 0.0\nv 1 0 0 0.0 1.0 0.0\nv 0 1 0 0.0 0.0 1.0\nf 1 2 3\n")
            .unwrap();
        assert_eq!(model.mesh.vertices[0].color, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(model.mesh.vertices[2].color, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn a_file_without_colours_gets_the_default() {
        let model = parse(TRIANGLE).unwrap();
        assert_eq!(model.mesh.vertices[0].color, DEFAULT_COLOR);
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let source = "\
# leading comment

v 0 0 0   # trailing comment
v 1 0 0
v 0 1 0

f 1 2 3
# trailing comment at EOF
";
        let model = parse(source).unwrap();
        assert_eq!(model.mesh.vertices.len(), 3);
        assert_eq!(model.mesh.indices.len(), 3);
    }

    #[test]
    fn unknown_directives_are_skipped_without_failing() {
        let source = "\
mtllib scene.mtl
usemtl red
s off
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
";
        assert!(parse(source).is_ok());
    }

    #[test]
    fn a_file_with_no_geometry_is_an_error() {
        assert!(parse("").is_err());
        assert!(parse("# only a comment").is_err());
        assert!(
            parse("v 0 0 0\nv 1 0 0\nv 0 1 0\n").is_err(),
            "vertices with no faces produce no drawable geometry"
        );
    }

    #[test]
    fn a_malformed_number_is_an_error_naming_its_line() {
        let e = parse("v 0 0 0\nv not-a-number 0 0\n").unwrap_err();
        assert!(e.to_string().contains("line 2"), "{e}");
    }

    #[test]
    fn a_short_vertex_line_is_an_error() {
        assert!(parse("v 0 0\nf 1 1 1\n").is_err());
        assert!(parse("vn 0 1\n").is_err());
    }

    #[test]
    fn a_face_with_too_few_vertices_is_skipped_and_counted() {
        let model = parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\nf 1 2 3\n").unwrap();
        assert_eq!(model.skipped_faces, 1);
        assert_eq!(model.mesh.indices.len(), 3);
    }

    /// Whatever a malformed file contains, the parser must return rather than panic.
    #[test]
    fn malformed_input_never_panics() {
        for source in [
            "f",
            "f 1",
            "f 1 2",
            "f //",
            "f 1// 2// 3//",
            "f a b c",
            "v",
            "v 1",
            "vn",
            "vt",
            "f -99 -98 -97",
            "f 1/2/3/4/5 1 1",
            "o",
            "\0\0\0",
        ] {
            let _ = parse(source);
        }
    }

    /// The phase criterion: a known OBJ must load to the same geometry as building
    /// it by hand. Checked on a full cube, where a winding or triple-sharing error
    /// would show up as the wrong triangle or vertex count.
    #[test]
    fn a_cube_loads_to_the_geometry_it_describes() {
        // 8 corners, 6 quad faces, each with its own normal.
        let source = "\
o cube
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
vn  0  0 -1
vn  0  0  1
vn -1  0  0
vn  1  0  0
vn  0 -1  0
vn  0  1  0
f 1//1 3//1 2//1
f 1//1 4//1 3//1
f 5//2 6//2 7//2
f 5//2 7//2 8//2
f 1//3 5//3 8//3
f 1//3 8//3 4//3
f 2//4 3//4 7//4
f 2//4 7//4 6//4
f 1//5 2//5 6//5
f 1//5 6//5 5//5
f 4//6 8//6 7//6
f 4//6 7//6 3//6
";
        let model = parse(source).expect("cube should parse");
        assert_eq!(model.name.as_deref(), Some("cube"));
        assert_eq!(model.mesh.indices.len(), 36, "12 triangles");
        assert_eq!(model.skipped_faces, 0);
        // Each corner appears with three different face normals, so the 8 positions
        // become 24 vertices — the hallmark of a hard-edged cube.
        assert_eq!(
            model.mesh.vertices.len(),
            24,
            "8 corners x 3 face normals; fewer means normals were merged"
        );
        // Every vertex sits on the cube's surface.
        for v in &model.mesh.vertices {
            assert!(
                (v.position.x.abs() - 1.0).abs() < 1e-6
                    && (v.position.y.abs() - 1.0).abs() < 1e-6
                    && (v.position.z.abs() - 1.0).abs() < 1e-6,
                "vertex off the cube: {}",
                v.position
            );
        }
        // Every index is in range, which a shifted 1-based conversion would break.
        assert!(model
            .mesh
            .indices
            .iter()
            .all(|i| (*i as usize) < model.mesh.vertices.len()));
        // All six axis-aligned normals are present.
        for expected in [
            Vec3::UNIT_X,
            -Vec3::UNIT_X,
            Vec3::UNIT_Y,
            -Vec3::UNIT_Y,
            Vec3::UNIT_Z,
            -Vec3::UNIT_Z,
        ] {
            assert!(
                model
                    .mesh
                    .vertices
                    .iter()
                    .any(|v| (v.normal - expected).length() < 1e-5),
                "no vertex has normal {expected}"
            );
        }
    }

    /// The other half of the criterion: the loaded quad-based cube must describe the
    /// same surface as one written with explicit triangles.
    #[test]
    fn a_quad_face_and_its_two_triangles_describe_the_same_surface() {
        let quad = parse("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n").unwrap();
        let tris =
            parse("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n").unwrap();
        assert_eq!(quad.mesh.indices.len(), tris.mesh.indices.len());
        // Same triangle corner positions, in the same order.
        let corners = |m: &ObjModel| -> Vec<Vec3> {
            m.mesh
                .indices
                .iter()
                .map(|i| m.mesh.vertices[*i as usize].position)
                .collect()
        };
        assert_eq!(corners(&quad), corners(&tris));
    }
}
