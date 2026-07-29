// Scene loader: builds a Scene + Camera + lights from a JSON description.
//
// Uses the hand-written parser in engine/core/json.rs (no serde). The schema is
// intentionally small and explicit — see assets/scenes/material_spheres.json for
// a complete example:
//
//   {
//     "name": "...",
//     "camera": { "position": [x,y,z], "target": [x,y,z], "up": [x,y,z],
//                 "projection": { "type": "perspective",
//                                 "fov_y_degrees": f, "aspect": f, "near": f, "far": f } },
//     "lights": [ { "type": "point", "position": [x,y,z], "color": [r,g,b],
//                   "ambient": f, "intensity": f },
//                 { "type": "directional", "direction": [x,y,z], ... } ],
//     "entities": [ { "name": "...",
//                     "transform": { "position": [..], "rotation_degrees": [..], "scale": [..] },
//                     "mesh": { "type": "sphere", "radius": f, "color": [r,g,b],
//                               "rings": u, "segments": u },
//                     "material": { "type": "matte", "color": [r,g,b,a],
//                                   "ambient": f, "diffuse": f } } ]
//   }
//
// Missing optional fields fall back to documented defaults; missing *required*
// fields are an error rather than a silent default, so a typo in a scene file
// surfaces immediately instead of producing an empty or black scene.

use std::path::Path;

use crate::engine::core::{json, EngineError, Result};
use crate::engine::math::{radians, Transform, Vec3, Vec4};
use crate::renderer::LightUniform;
use crate::engine::geometry::Shape;
use crate::scene::{
    plane, plane_shape, sphere, sphere_shape, Camera, ColliderComponent, Entity, Hierarchy,
    MaterialComponent, MeshComponent, Projection, Scene, TransformComponent,
};

/// Everything a scene file describes: the entities, the camera to view them
/// with, the lights that illuminate them, and the parent/child links between
/// entities (from nested `"children"` arrays).
pub struct LoadedScene {
    pub name: String,
    pub scene: Scene,
    pub camera: Camera,
    pub lights: Vec<LightUniform>,
    pub hierarchy: Hierarchy,
}

fn err(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidState(msg.into())
}

/// Parse a scene from a JSON string.
pub fn parse_scene(source: &str) -> Result<LoadedScene> {
    let root = json::parse(source)?;

    let name = root.get_str("name").unwrap_or("unnamed").to_string();

    let camera_value = root
        .get("camera")
        .ok_or_else(|| err("scene: missing required \"camera\" object"))?;
    let camera = parse_camera(camera_value)?;

    let mut lights = Vec::new();
    if let Some(entries) = root.get_array("lights") {
        for (i, entry) in entries.iter().enumerate() {
            lights.push(
                parse_light(entry).map_err(|e| err(format!("scene: lights[{i}]: {e}")))?,
            );
        }
    }

    let mut scene = Scene::new();
    let mut hierarchy = Hierarchy::new();
    if let Some(entries) = root.get_array("entities") {
        for (i, entry) in entries.iter().enumerate() {
            add_entity(&mut scene, &mut hierarchy, entry, None)
                .map_err(|e| err(format!("scene: entities[{i}]: {e}")))?;
        }
    }

    Ok(LoadedScene { name, scene, camera, lights, hierarchy })
}

/// Read and parse a scene file from disk.
pub fn load_scene_file(path: impl AsRef<Path>) -> Result<LoadedScene> {
    let path = path.as_ref();
    let source = std::fs::read_to_string(path)?;
    parse_scene(&source).map_err(|e| err(format!("{}: {e}", path.display())))
}

fn parse_camera(value: &json::JsonValue) -> Result<Camera> {
    let position = vec3_field(value, "position")
        .ok_or_else(|| err("camera: missing required \"position\" [x,y,z]"))?;
    let target = vec3_field(value, "target")
        .ok_or_else(|| err("camera: missing required \"target\" [x,y,z]"))?;
    let up = vec3_field(value, "up").unwrap_or(Vec3::UNIT_Y);

    let projection = match value.get("projection") {
        Some(p) => parse_projection(p)?,
        // A sensible default beats failing outright: most scenes want a plain
        // 60-degree perspective camera.
        None => Projection::perspective(radians(60.0), 16.0 / 9.0, 0.1, 200.0),
    };

    Ok(Camera::new(position, target, up, projection))
}

fn parse_projection(value: &json::JsonValue) -> Result<Projection> {
    let kind = value
        .get_str("type")
        .ok_or_else(|| err("projection: missing \"type\" (\"perspective\" or \"orthographic\")"))?;
    let near = value.get_f32("near").unwrap_or(0.1);
    let far = value.get_f32("far").unwrap_or(200.0);
    let aspect = value.get_f32("aspect").unwrap_or(16.0 / 9.0);

    match kind {
        "perspective" => {
            let fov_y_degrees = value.get_f32("fov_y_degrees").unwrap_or(60.0);
            Ok(Projection::perspective(radians(fov_y_degrees), aspect, near, far))
        }
        "orthographic" => {
            // Either explicit bounds, or the ergonomic height + aspect form.
            match (
                value.get_f32("left"),
                value.get_f32("right"),
                value.get_f32("bottom"),
                value.get_f32("top"),
            ) {
                (Some(left), Some(right), Some(bottom), Some(top)) => {
                    Ok(Projection::orthographic(left, right, bottom, top, near, far))
                }
                _ => {
                    let height = value.get_f32("height").ok_or_else(|| {
                        err("orthographic projection: needs either left/right/bottom/top or \"height\"")
                    })?;
                    Ok(Projection::orthographic_sized(height, aspect, near, far))
                }
            }
        }
        other => Err(err(format!(
            "projection: unknown type {other:?} (expected \"perspective\" or \"orthographic\")"
        ))),
    }
}

fn parse_light(value: &json::JsonValue) -> Result<LightUniform> {
    let kind = value
        .get_str("type")
        .ok_or_else(|| err("light: missing \"type\" (\"point\" or \"directional\")"))?;
    let color = vec3_field(value, "color").unwrap_or(Vec3::ONE);
    let ambient = value.get_f32("ambient").unwrap_or(0.05);
    let intensity = value.get_f32("intensity").unwrap_or(1.0);

    match kind {
        "point" => {
            let position = vec3_field(value, "position")
                .ok_or_else(|| err("point light: missing \"position\" [x,y,z]"))?;
            Ok(LightUniform::point(position, color, ambient, intensity))
        }
        "directional" => {
            let direction = vec3_field(value, "direction")
                .ok_or_else(|| err("directional light: missing \"direction\" [x,y,z]"))?;
            if direction.length_squared() <= 0.0 {
                return Err(err("directional light: \"direction\" must be non-zero"));
            }
            Ok(LightUniform::directional(direction, color, ambient, intensity))
        }
        other => Err(err(format!(
            "light: unknown type {other:?} (expected \"point\" or \"directional\")"
        ))),
    }
}

/// Add one entity (and, recursively, its `"children"`) to the scene.
///
/// A child's transform is relative to its parent — the composition happens at
/// render time via `Hierarchy::world_matrices`, not here, so moving a parent at
/// runtime moves its whole subtree.
fn add_entity(
    scene: &mut Scene,
    hierarchy: &mut Hierarchy,
    value: &json::JsonValue,
    parent: Option<Entity>,
) -> Result<()> {
    let mesh_value = value
        .get("mesh")
        .ok_or_else(|| err("entity: missing required \"mesh\" object"))?;
    let (mesh, shape) = parse_mesh(mesh_value)?;

    let material = match value.get("material") {
        Some(m) => parse_material(m)?,
        None => MaterialComponent::default(),
    };

    let transform = match value.get("transform") {
        Some(t) => parse_transform(t),
        None => Transform::identity(),
    };

    let entity = scene.create_entity();
    scene.add_component(entity, mesh);
    scene.add_component(entity, ColliderComponent::new(shape));
    scene.add_component(entity, material);
    scene.add_component(entity, TransformComponent { local: transform });

    if let Some(parent) = parent {
        // Entities are created parent-before-child and each id is fresh, so this
        // cannot cycle; surface an error rather than ignoring it if that ever changes.
        hierarchy.set_parent(entity, parent)?;
    }

    if let Some(children) = value.get_array("children") {
        for (i, child) in children.iter().enumerate() {
            add_entity(scene, hierarchy, child, Some(entity))
                .map_err(|e| err(format!("children[{i}]: {e}")))?;
        }
    }

    Ok(())
}

/// Parse a mesh and, alongside it, the analytic shape it approximates.
///
/// The two are produced together on purpose: the exact radius or size is right
/// here in the scene file, and recovering it from the generated triangles later
/// would bake the tessellation error into every consumer (the CPU tracer and the
/// physics collider both need the equation, not the approximation).
fn parse_mesh(value: &json::JsonValue) -> Result<(MeshComponent, Shape)> {
    let kind = value
        .get_str("type")
        .ok_or_else(|| err("mesh: missing \"type\" (\"plane\" or \"sphere\")"))?;
    let color = vec3_field(value, "color").unwrap_or(Vec3::ONE);
    // Geometry is authored at the origin; placement belongs to the transform.
    let center = Vec3::ZERO;

    match kind {
        "plane" => {
            let size = value.get_f32("size").unwrap_or(1.0);
            if size <= 0.0 {
                return Err(err("plane mesh: \"size\" must be positive"));
            }
            Ok((plane(center, size, color), plane_shape(size)))
        }
        "sphere" => {
            let radius = value.get_f32("radius").unwrap_or(1.0);
            if radius <= 0.0 {
                return Err(err("sphere mesh: \"radius\" must be positive"));
            }
            let rings = value.get_u32("rings").unwrap_or(16);
            let segments = value.get_u32("segments").unwrap_or(24);
            Ok((
                sphere(center, radius, color, rings, segments),
                sphere_shape(radius),
            ))
        }
        other => Err(err(format!(
            "mesh: unknown type {other:?} (expected \"plane\" or \"sphere\")"
        ))),
    }
}

fn parse_material(value: &json::JsonValue) -> Result<MaterialComponent> {
    let kind = value.get_str("type").unwrap_or("matte");
    let color = value
        .get_vec4("color")
        .map(|c| Vec4::new(c[0], c[1], c[2], c[3]))
        .or_else(|| value.get_vec3("color").map(|c| Vec4::new(c[0], c[1], c[2], 1.0)))
        .unwrap_or(Vec4::new(1.0, 1.0, 1.0, 1.0));

    match kind {
        "matte" => {
            let ambient = value.get_f32("ambient").unwrap_or(0.1);
            let diffuse = value.get_f32("diffuse").unwrap_or(0.9);
            Ok(MaterialComponent::matte(color, ambient, diffuse))
        }
        "mirror" => {
            let reflectivity = value.get_f32("reflectivity").unwrap_or(0.8);
            Ok(MaterialComponent::mirror(color, reflectivity))
        }
        "glass" => {
            let ior = value.get_f32("ior").unwrap_or(1.5);
            let transparency = value.get_f32("transparency").unwrap_or(0.8);
            Ok(MaterialComponent::glass(color, ior, transparency))
        }
        other => Err(err(format!(
            "material: unknown type {other:?} (expected \"matte\", \"mirror\" or \"glass\")"
        ))),
    }
}

/// Transforms are fully optional: any missing channel keeps its identity value.
/// Rotation is authored in degrees (`rotation_degrees`) since hand-written
/// scene files are far more readable that way; radians are accepted too.
fn parse_transform(value: &json::JsonValue) -> Transform {
    let position = vec3_field(value, "position").unwrap_or(Vec3::ZERO);
    let rotation = match vec3_field(value, "rotation_degrees") {
        Some(deg) => Vec3::new(radians(deg.x), radians(deg.y), radians(deg.z)),
        None => vec3_field(value, "rotation").unwrap_or(Vec3::ZERO),
    };
    let scale = vec3_field(value, "scale").unwrap_or(Vec3::ONE);
    Transform::new(position, rotation, scale)
}

fn vec3_field(value: &json::JsonValue, key: &str) -> Option<Vec3> {
    value.get_vec3(key).map(|v| Vec3::new(v[0], v[1], v[2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{MaterialComponent, MeshComponent};

    const MINIMAL: &str = r#"{
        "camera": { "position": [0, 1, 5], "target": [0, 0, 0] },
        "entities": [ { "mesh": { "type": "sphere" } } ]
    }"#;

    /// `Result::unwrap_err` needs `T: Debug`, and `Scene` (owning boxed
    /// components) deliberately isn't `Debug` — so unwrap the error side by hand.
    fn expect_err(result: Result<LoadedScene>) -> EngineError {
        match result {
            Ok(loaded) => panic!("expected an error, got scene {:?}", loaded.name),
            Err(e) => e,
        }
    }

    #[test]
    fn parses_minimal_scene_with_defaults() {
        let loaded = parse_scene(MINIMAL).expect("minimal scene should parse");
        assert_eq!(loaded.name, "unnamed");
        assert_eq!(loaded.camera.position, Vec3::new(0.0, 1.0, 5.0));
        assert_eq!(loaded.camera.up, Vec3::UNIT_Y, "up defaults to +Y");
        assert!(loaded.lights.is_empty());
        assert_eq!(loaded.scene.entities_with::<MeshComponent>().len(), 1);
    }

    #[test]
    fn entity_gets_mesh_material_and_transform_components() {
        let loaded = parse_scene(MINIMAL).unwrap();
        let entity = loaded.scene.entities_with::<MeshComponent>()[0];
        assert!(loaded.scene.get_component::<MeshComponent>(entity).is_some());
        assert!(loaded.scene.get_component::<MaterialComponent>(entity).is_some());
        assert!(loaded.scene.get_component::<TransformComponent>(entity).is_some());
    }

    #[test]
    fn parses_transform_position_scale_and_degrees_rotation() {
        let src = r#"{
            "camera": { "position": [0,0,1], "target": [0,0,0] },
            "entities": [ {
                "mesh": { "type": "plane", "size": 2.0 },
                "transform": {
                    "position": [1, 2, 3],
                    "rotation_degrees": [0, 90, 0],
                    "scale": [2, 2, 2]
                }
            } ]
        }"#;
        let loaded = parse_scene(src).unwrap();
        let entity = loaded.scene.entities_with::<MeshComponent>()[0];
        let t = loaded.scene.get_component::<TransformComponent>(entity).unwrap();
        assert_eq!(t.local.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.local.scale, Vec3::splat(2.0));
        assert!((t.local.rotation.y - radians(90.0)).abs() < 1e-6);
    }

    #[test]
    fn transform_defaults_to_identity_when_absent() {
        let loaded = parse_scene(MINIMAL).unwrap();
        let entity = loaded.scene.entities_with::<MeshComponent>()[0];
        let t = loaded.scene.get_component::<TransformComponent>(entity).unwrap();
        assert_eq!(t.local.position, Vec3::ZERO);
        assert_eq!(t.local.scale, Vec3::ONE);
        assert_eq!(t.local.rotation, Vec3::ZERO);
    }

    #[test]
    fn nested_children_become_hierarchy_links() {
        let src = r#"{
            "camera": { "position": [0,0,1], "target": [0,0,0] },
            "entities": [ {
                "mesh": { "type": "sphere" },
                "transform": { "position": [10, 0, 0] },
                "children": [ {
                    "mesh": { "type": "sphere" },
                    "transform": { "position": [0, 2, 0] },
                    "children": [ { "mesh": { "type": "sphere" },
                                    "transform": { "position": [0, 0, 3] } } ]
                } ]
            } ]
        }"#;
        let loaded = parse_scene(src).unwrap();
        let entities = loaded.scene.all_entities().to_vec();
        assert_eq!(entities.len(), 3);

        let (root, child, grandchild) = (entities[0], entities[1], entities[2]);
        assert_eq!(loaded.hierarchy.parent(root), None, "first entity is a root");
        assert_eq!(loaded.hierarchy.parent(child), Some(root));
        assert_eq!(loaded.hierarchy.parent(grandchild), Some(child));

        // Child transforms are relative: the grandchild's world position is the
        // sum of the whole chain (all three are pure translations).
        let local_of = |e: crate::scene::Entity| {
            loaded
                .scene
                .get_component::<TransformComponent>(e)
                .map(|t| t.world_matrix())
                .unwrap_or(crate::engine::math::Mat4::IDENTITY)
        };
        let world = loaded.hierarchy.world_matrix(grandchild, &local_of);
        let p = world.transform_point(Vec3::ZERO);
        assert!(
            (p - Vec3::new(10.0, 2.0, 3.0)).length() < 1e-5,
            "expected (10,2,3), got {p}"
        );
    }

    #[test]
    fn flat_entities_have_no_parents() {
        let loaded = parse_scene(MINIMAL).unwrap();
        let entity = loaded.scene.all_entities()[0];
        assert_eq!(loaded.hierarchy.parent(entity), None);
    }

    #[test]
    fn error_inside_a_child_is_reported_with_its_path() {
        let e = expect_err(parse_scene(
            r#"{ "camera": { "position": [0,0,1], "target": [0,0,0] },
                 "entities": [ { "mesh": { "type": "sphere" },
                                 "children": [ { "mesh": { "type": "torus" } } ] } ] }"#,
        ));
        let msg = format!("{e}");
        assert!(msg.contains("children[0]"), "got: {msg}");
    }

    #[test]
    fn parses_both_light_types() {
        let src = r#"{
            "camera": { "position": [0,0,1], "target": [0,0,0] },
            "lights": [
                { "type": "point", "position": [0, 5, 0], "color": [1, 1, 1], "intensity": 2.0 },
                { "type": "directional", "direction": [0, -1, 0], "ambient": 0.25 }
            ]
        }"#;
        let loaded = parse_scene(src).unwrap();
        assert_eq!(loaded.lights.len(), 2);
        // position.w encodes the light type: 1.0 = point, 0.0 = directional.
        assert_eq!(loaded.lights[0].position[3], 1.0);
        assert_eq!(loaded.lights[0].position[1], 5.0);
        assert_eq!(loaded.lights[0].diffuse, 2.0);
        assert_eq!(loaded.lights[1].position[3], 0.0);
        assert_eq!(loaded.lights[1].direction[1], -1.0);
        assert_eq!(loaded.lights[1].ambient, 0.25);
    }

    #[test]
    fn parses_all_material_types() {
        let src = r#"{
            "camera": { "position": [0,0,1], "target": [0,0,0] },
            "entities": [
                { "mesh": { "type": "sphere" },
                  "material": { "type": "matte", "color": [1, 0, 0, 1], "ambient": 0.2, "diffuse": 0.7 } },
                { "mesh": { "type": "sphere" },
                  "material": { "type": "mirror", "color": [0, 1, 0, 1], "reflectivity": 0.9 } },
                { "mesh": { "type": "sphere" },
                  "material": { "type": "glass", "color": [0, 0, 1, 1], "ior": 1.7, "transparency": 0.6 } }
            ]
        }"#;
        let loaded = parse_scene(src).unwrap();
        let mut kinds: Vec<u32> = loaded
            .scene
            .entities_with::<MaterialComponent>()
            .iter()
            .map(|e| loaded.scene.get_component::<MaterialComponent>(*e).unwrap().material_type)
            .collect();
        kinds.sort_unstable();
        assert_eq!(kinds, vec![0, 1, 2], "matte, mirror and glass must all appear");
    }

    #[test]
    fn material_color_accepts_rgb_as_well_as_rgba() {
        let src = r#"{
            "camera": { "position": [0,0,1], "target": [0,0,0] },
            "entities": [ { "mesh": { "type": "sphere" },
                            "material": { "type": "matte", "color": [0.25, 0.5, 0.75] } } ]
        }"#;
        let loaded = parse_scene(src).unwrap();
        let entity = loaded.scene.entities_with::<MaterialComponent>()[0];
        let m = loaded.scene.get_component::<MaterialComponent>(entity).unwrap();
        assert_eq!(m.color, Vec4::new(0.25, 0.5, 0.75, 1.0), "alpha defaults to 1.0");
    }

    #[test]
    fn parses_orthographic_projection_both_forms() {
        let explicit = r#"{ "camera": { "position": [0,0,1], "target": [0,0,0],
            "projection": { "type": "orthographic", "left": -4, "right": 4,
                            "bottom": -2, "top": 2, "near": 0.5, "far": 50 } } }"#;
        let loaded = parse_scene(explicit).unwrap();
        assert!(loaded.camera.projection.is_orthographic());

        let sized = r#"{ "camera": { "position": [0,0,1], "target": [0,0,0],
            "projection": { "type": "orthographic", "height": 10, "aspect": 2 } } }"#;
        let loaded = parse_scene(sized).unwrap();
        match loaded.camera.projection {
            Projection::Orthographic { left, right, top, bottom, .. } => {
                assert!((right - left - 20.0).abs() < 1e-5);
                assert!((top - bottom - 10.0).abs() < 1e-5);
            }
            _ => panic!("expected orthographic"),
        }
    }

    #[test]
    fn perspective_fov_is_read_in_degrees() {
        let src = r#"{ "camera": { "position": [0,0,1], "target": [0,0,0],
            "projection": { "type": "perspective", "fov_y_degrees": 90, "aspect": 1,
                            "near": 0.1, "far": 10 } } }"#;
        let loaded = parse_scene(src).unwrap();
        match loaded.camera.projection {
            Projection::Perspective { fov_y, .. } => {
                assert!((fov_y - radians(90.0)).abs() < 1e-6);
            }
            _ => panic!("expected perspective"),
        }
    }

    #[test]
    fn missing_camera_is_an_error() {
        let e = expect_err(parse_scene(r#"{ "entities": [] }"#));
        assert!(format!("{e}").contains("camera"), "error should name the missing field: {e}");
    }

    #[test]
    fn missing_required_camera_fields_are_errors() {
        assert!(parse_scene(r#"{ "camera": { "target": [0,0,0] } }"#).is_err());
        assert!(parse_scene(r#"{ "camera": { "position": [0,0,1] } }"#).is_err());
    }

    #[test]
    fn entity_without_mesh_is_an_error() {
        let e = expect_err(parse_scene(
            r#"{ "camera": { "position": [0,0,1], "target": [0,0,0] },
                 "entities": [ { "material": { "type": "matte" } } ] }"#,
        ));
        let msg = format!("{e}");
        assert!(msg.contains("entities[0]"), "error should point at the entity: {msg}");
        assert!(msg.contains("mesh"), "error should name the missing field: {msg}");
    }

    #[test]
    fn unknown_type_names_are_errors() {
        let base = r#"{ "camera": { "position": [0,0,1], "target": [0,0,0] }, "#;
        assert!(parse_scene(&format!(
            r#"{base}"entities": [ {{ "mesh": {{ "type": "torus" }} }} ] }}"#
        ))
        .is_err());
        assert!(parse_scene(&format!(
            r#"{base}"entities": [ {{ "mesh": {{ "type": "sphere" }},
                 "material": {{ "type": "velvet" }} }} ] }}"#
        ))
        .is_err());
        assert!(parse_scene(&format!(
            r#"{base}"lights": [ {{ "type": "spot" }} ] }}"#
        ))
        .is_err());
    }

    #[test]
    fn degenerate_geometry_is_rejected() {
        let base = r#"{ "camera": { "position": [0,0,1], "target": [0,0,0] }, "#;
        assert!(parse_scene(&format!(
            r#"{base}"entities": [ {{ "mesh": {{ "type": "sphere", "radius": 0 }} }} ] }}"#
        ))
        .is_err());
        assert!(parse_scene(&format!(
            r#"{base}"entities": [ {{ "mesh": {{ "type": "plane", "size": -1 }} }} ] }}"#
        ))
        .is_err());
        assert!(parse_scene(&format!(
            r#"{base}"lights": [ {{ "type": "directional", "direction": [0,0,0] }} ] }}"#
        ))
        .is_err());
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(parse_scene("").is_err());
        assert!(parse_scene("{").is_err());
        assert!(parse_scene("not json at all").is_err());
        assert!(parse_scene(r#"{ "camera": }"#).is_err());
    }

    #[test]
    fn light_error_message_includes_the_index() {
        let e = expect_err(parse_scene(
            r#"{ "camera": { "position": [0,0,1], "target": [0,0,0] },
                 "lights": [ { "type": "point", "position": [0,1,0] },
                             { "type": "point" } ] }"#,
        ));
        assert!(format!("{e}").contains("lights[1]"), "got: {e}");
    }

    #[test]
    fn loads_the_repository_scene_file_from_disk() {
        // Exercises the real file path, not just in-memory strings: this is the
        // scene the demo ships with, so a schema drift breaks this test.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/scenes/material_spheres.json");
        let loaded = load_scene_file(&path)
            .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
        assert_eq!(loaded.name, "material_spheres");
        assert_eq!(loaded.lights.len(), 2);
        assert_eq!(
            loaded.scene.entities_with::<MeshComponent>().len(),
            5,
            "ground, three spheres, and the blue sphere's satellite child"
        );
        // The satellite is nested, so exactly one entity must have a parent.
        let parented = loaded
            .scene
            .all_entities()
            .iter()
            .filter(|e| loaded.hierarchy.parent(**e).is_some())
            .count();
        assert_eq!(parented, 1, "the scene file's nested child must be linked");
    }

    #[test]
    fn identical_materials_across_entities_collapse_to_one_registry_slot() {
        // The renderer feeds every drawn entity's material through
        // MaterialRegistry; this checks the loader + registry pair actually
        // deduplicates, which is what keeps the 256-slot buffer from filling up
        // on a scene full of same-looking objects.
        let src = r#"{
            "camera": { "position": [0,0,1], "target": [0,0,0] },
            "entities": [
                { "mesh": { "type": "sphere" },
                  "material": { "type": "matte", "color": [1, 0, 0, 1], "ambient": 0.1, "diffuse": 0.9 } },
                { "mesh": { "type": "sphere" },
                  "material": { "type": "matte", "color": [1, 0, 0, 1], "ambient": 0.1, "diffuse": 0.9 } },
                { "mesh": { "type": "sphere" },
                  "material": { "type": "matte", "color": [0, 1, 0, 1], "ambient": 0.1, "diffuse": 0.9 } }
            ]
        }"#;
        let loaded = parse_scene(src).unwrap();
        let mut registry = crate::scene::MaterialRegistry::new();
        let indices: Vec<u32> = loaded
            .scene
            .all_entities()
            .iter()
            .filter_map(|e| loaded.scene.get_component::<MaterialComponent>(*e))
            .map(|m| registry.register(m))
            .collect();

        assert_eq!(registry.len(), 2, "the two identical red materials must share a slot");
        assert_eq!(indices[0], indices[1], "identical materials get the same index");
        assert_ne!(indices[0], indices[2], "a different colour gets its own index");
    }

    #[test]
    fn scene_world_bounds_reflect_hierarchy_and_transforms() {
        // Mirrors what the render loop does to size the shadow frustum: local
        // AABB -> world matrix (through the parent chain) -> merged bounds.
        let src = r#"{
            "camera": { "position": [0,0,1], "target": [0,0,0] },
            "entities": [ {
                "mesh": { "type": "sphere", "radius": 1.0 },
                "transform": { "position": [10, 0, 0] },
                "children": [ { "mesh": { "type": "sphere", "radius": 1.0 },
                                "transform": { "position": [0, 5, 0] } } ]
            } ]
        }"#;
        let loaded = parse_scene(src).unwrap();
        let local_of = |e: Entity| {
            loaded
                .scene
                .get_component::<TransformComponent>(e)
                .map(|t| t.world_matrix())
                .unwrap_or(crate::engine::math::Mat4::IDENTITY)
        };
        let entities = loaded.scene.entities_with::<MeshComponent>();
        let mut bounds: Option<crate::scene::Aabb> = None;
        for (entity, model) in loaded.hierarchy.world_matrices(&entities, &local_of) {
            let mesh = loaded.scene.get_component::<MeshComponent>(entity).unwrap();
            let local = crate::scene::Aabb::from_points(mesh.vertices.iter().map(|v| v.position))
                .expect("sphere has vertices");
            let world = local.transformed(&model);
            bounds = Some(match bounds {
                Some(acc) => acc.merge(&world),
                None => world,
            });
        }
        let bounds = bounds.expect("scene has meshes");
        // Parent sphere spans y in [-1, 1] at x=10; the child sits 5 units above it.
        assert!(bounds.max.y > 5.0, "child must extend the bounds upward: {bounds:?}");
        assert!(bounds.min.y < 0.0, "parent's lower half must be included: {bounds:?}");
        assert!((bounds.center().x - 10.0).abs() < 1e-4, "both sit at x=10: {bounds:?}");
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let e = expect_err(load_scene_file("definitely/not/here.json"));
        assert!(matches!(e, EngineError::Io(_)), "expected Io error, got {e:?}");
    }
}
