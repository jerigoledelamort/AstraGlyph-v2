// Scene serialiser: a `Scene` back out to the JSON the loader reads.
//
// The other half of `scene::loader`, and the piece an editor cannot do without —
// "move an entity and save it" is only useful if the result loads back identically.
// That round trip is the property this module is tested on, not the byte-for-byte
// output: JSON has many valid spellings of the same document, and asserting on the
// text would break on a formatting change while missing an actual data loss.
//
// Written by hand, like the parser, for the same reason. It also means the output is
// formatted for a human to read and diff — a scene file is something the author
// edits, so a minified single line would be worse than useless.

use std::collections::HashMap;

use crate::engine::core::Result;
use crate::engine::geometry::Shape;
use crate::engine::math::{degrees, Vec3};
use crate::renderer::LightUniform;
use crate::scene::{
    Camera, ColliderComponent, Entity, Hierarchy, MaterialComponent, MaterialType, MeshComponent,
    Projection, Scene, TransformComponent,
};

/// How a mesh should be written back out.
///
/// A `MeshComponent` is a triangle soup: the fact that it *was* a sphere of radius 1
/// with 24 rings is not recoverable from its vertices. So the descriptor is recorded
/// when the scene is built and carried alongside, rather than reverse-engineered —
/// guessing "this looks spherical" would turn a round trip into a lossy re-mesh.
#[derive(Clone, Debug, PartialEq)]
pub enum MeshSource {
    Plane { size: f32 },
    Sphere { radius: f32, rings: u32, segments: u32 },
    Box { half_extents: Vec3 },
    Obj { path: String },
}

impl MeshSource {
    /// Infer a source from a collider, for an entity the editor created rather than
    /// loaded. Loses the tessellation detail, which is why loaded entities keep their
    /// original descriptor instead.
    pub fn from_shape(shape: &Shape) -> Self {
        match shape {
            Shape::Sphere { radius } => Self::Sphere {
                radius: *radius,
                rings: 16,
                segments: 24,
            },
            Shape::Plane { half_size, .. } => Self::Plane {
                size: half_size * 2.0,
            },
            Shape::Box { half_extents } => Self::Box {
                half_extents: *half_extents,
            },
        }
    }
}

/// Everything needed to write a scene back out.
///
/// The mesh sources are separate from the `Scene` because they are authoring
/// metadata, not runtime state: the renderer never needs to know a sphere's ring
/// count, so putting it in a component would make every frame carry it.
pub struct SceneDocument<'a> {
    pub name: &'a str,
    pub scene: &'a Scene,
    pub camera: &'a Camera,
    pub lights: &'a [LightUniform],
    pub hierarchy: &'a Hierarchy,
    /// Mesh descriptors by entity id, for entities whose original form is known.
    pub mesh_sources: &'a HashMap<u64, MeshSource>,
    /// Texture paths by `texture_index`, from `LoadedScene::textures`. A material
    /// whose index has no entry here is written without its texture — the index
    /// alone is meaningless in a file that a fresh load will re-number anyway.
    pub textures: &'a [String],
}

/// Serialise a scene to the loader's JSON format.
pub fn to_json(document: &SceneDocument<'_>) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"name\": {},\n", quote(document.name)));

    // Camera.
    out.push_str("  \"camera\": {\n");
    out.push_str(&format!(
        "    \"position\": {},\n",
        vec3_json(document.camera.position)
    ));
    out.push_str(&format!(
        "    \"target\": {},\n",
        vec3_json(document.camera.target)
    ));
    out.push_str(&format!("    \"up\": {},\n", vec3_json(document.camera.up)));
    out.push_str("    \"projection\": {\n");
    match document.camera.projection {
        Projection::Perspective {
            fov_y,
            aspect,
            near,
            far,
        } => {
            out.push_str("      \"type\": \"perspective\",\n");
            // Degrees, because that is what the loader reads and what a human edits.
            out.push_str(&format!(
                "      \"fov_y_degrees\": {},\n",
                number(degrees(fov_y))
            ));
            out.push_str(&format!("      \"aspect\": {},\n", number(aspect)));
            out.push_str(&format!("      \"near\": {},\n", number(near)));
            out.push_str(&format!("      \"far\": {}\n", number(far)));
        }
        Projection::Orthographic {
            left,
            right,
            bottom,
            top,
            near,
            far,
        } => {
            out.push_str("      \"type\": \"orthographic\",\n");
            out.push_str(&format!("      \"left\": {},\n", number(left)));
            out.push_str(&format!("      \"right\": {},\n", number(right)));
            out.push_str(&format!("      \"bottom\": {},\n", number(bottom)));
            out.push_str(&format!("      \"top\": {},\n", number(top)));
            out.push_str(&format!("      \"near\": {},\n", number(near)));
            out.push_str(&format!("      \"far\": {}\n", number(far)));
        }
    }
    out.push_str("    }\n");
    out.push_str("  },\n");

    // Lights.
    out.push_str("  \"lights\": [\n");
    for (i, light) in document.lights.iter().enumerate() {
        let is_point = light.position[3] > 0.5;
        out.push_str("    {\n");
        if is_point {
            out.push_str("      \"type\": \"point\",\n");
            out.push_str(&format!(
                "      \"position\": {},\n",
                vec3_json(Vec3::new(
                    light.position[0],
                    light.position[1],
                    light.position[2]
                ))
            ));
        } else {
            out.push_str("      \"type\": \"directional\",\n");
            out.push_str(&format!(
                "      \"direction\": {},\n",
                vec3_json(Vec3::new(
                    light.direction[0],
                    light.direction[1],
                    light.direction[2]
                ))
            ));
        }
        out.push_str(&format!(
            "      \"color\": {},\n",
            vec3_json(Vec3::new(light.color[0], light.color[1], light.color[2]))
        ));
        out.push_str(&format!("      \"ambient\": {},\n", number(light.ambient)));
        out.push_str(&format!("      \"intensity\": {}\n", number(light.diffuse)));
        out.push_str("    }");
        if i + 1 < document.lights.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ],\n");

    // Entities. Only roots at the top level; children nest inside their parent, which
    // is how the loader expects the hierarchy and the only way it round-trips.
    out.push_str("  \"entities\": [\n");
    let roots: Vec<Entity> = document
        .scene
        .entities_with::<MeshComponent>()
        .into_iter()
        .filter(|e| document.hierarchy.parent(*e).is_none())
        .collect();
    for (i, entity) in roots.iter().enumerate() {
        write_entity(&mut out, document, *entity, 2);
        if i + 1 < roots.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Write one entity and, recursively, its children.
fn write_entity(out: &mut String, document: &SceneDocument<'_>, entity: Entity, depth: usize) {
    let pad = "  ".repeat(depth);
    let inner = "  ".repeat(depth + 1);
    out.push_str(&format!("{pad}{{\n"));
    out.push_str(&format!(
        "{inner}\"name\": {},\n",
        quote(&format!("entity_{}", entity.id()))
    ));

    if let Some(transform) = document.scene.get_component::<TransformComponent>(entity) {
        let t = &transform.local;
        out.push_str(&format!("{inner}\"transform\": {{\n"));
        out.push_str(&format!(
            "{inner}  \"position\": {},\n",
            vec3_json(t.position)
        ));
        out.push_str(&format!(
            "{inner}  \"rotation_degrees\": {},\n",
            vec3_json(Vec3::new(
                degrees(t.rotation.x),
                degrees(t.rotation.y),
                degrees(t.rotation.z)
            ))
        ));
        out.push_str(&format!("{inner}  \"scale\": {}\n", vec3_json(t.scale)));
        out.push_str(&format!("{inner}}},\n"));
    }

    // Mesh. The recorded descriptor if there is one, otherwise inferred from the
    // collider — which is what an editor-created entity has.
    let source = document
        .mesh_sources
        .get(&entity.id())
        .cloned()
        .or_else(|| {
            document
                .scene
                .get_component::<ColliderComponent>(entity)
                .map(|c| MeshSource::from_shape(&c.shape))
        });
    let color = document
        .scene
        .get_component::<MeshComponent>(entity)
        .and_then(|m| m.vertices.first())
        .map(|v| v.color)
        .unwrap_or(Vec3::ONE);
    match source {
        Some(MeshSource::Plane { size }) => {
            out.push_str(&format!("{inner}\"mesh\": {{\n"));
            out.push_str(&format!("{inner}  \"type\": \"plane\",\n"));
            out.push_str(&format!("{inner}  \"size\": {},\n", number(size)));
            out.push_str(&format!("{inner}  \"color\": {}\n", vec3_json(color)));
            out.push_str(&format!("{inner}}},\n"));
        }
        Some(MeshSource::Sphere {
            radius,
            rings,
            segments,
        }) => {
            out.push_str(&format!("{inner}\"mesh\": {{\n"));
            out.push_str(&format!("{inner}  \"type\": \"sphere\",\n"));
            out.push_str(&format!("{inner}  \"radius\": {},\n", number(radius)));
            out.push_str(&format!("{inner}  \"rings\": {rings},\n"));
            out.push_str(&format!("{inner}  \"segments\": {segments},\n"));
            out.push_str(&format!("{inner}  \"color\": {}\n", vec3_json(color)));
            out.push_str(&format!("{inner}}},\n"));
        }
        Some(MeshSource::Box { half_extents }) => {
            out.push_str(&format!("{inner}\"mesh\": {{\n"));
            out.push_str(&format!("{inner}  \"type\": \"box\",\n"));
            out.push_str(&format!(
                "{inner}  \"half_extents\": {},\n",
                vec3_json(half_extents)
            ));
            out.push_str(&format!("{inner}  \"color\": {}\n", vec3_json(color)));
            out.push_str(&format!("{inner}}},\n"));
        }
        Some(MeshSource::Obj { path }) => {
            out.push_str(&format!("{inner}\"mesh\": {{\n"));
            out.push_str(&format!("{inner}  \"type\": \"obj\",\n"));
            out.push_str(&format!("{inner}  \"path\": {},\n", quote(&path)));
            out.push_str(&format!("{inner}  \"color\": {}\n", vec3_json(color)));
            out.push_str(&format!("{inner}}},\n"));
        }
        None => {
            // No descriptor and no collider: fall back to a unit sphere rather than
            // writing an entity the loader would reject for a missing mesh.
            out.push_str(&format!("{inner}\"mesh\": {{\n"));
            out.push_str(&format!("{inner}  \"type\": \"sphere\",\n"));
            out.push_str(&format!("{inner}  \"radius\": 1.0,\n"));
            out.push_str(&format!("{inner}  \"color\": {}\n", vec3_json(color)));
            out.push_str(&format!("{inner}}},\n"));
        }
    }

    // Material.
    let material = document
        .scene
        .get_component::<MaterialComponent>(entity)
        .copied()
        .unwrap_or_default();
    out.push_str(&format!("{inner}\"material\": {{\n"));
    let kind = if material.material_type == MaterialType::Mirror as u32 {
        "mirror"
    } else if material.material_type == MaterialType::Glass as u32 {
        "glass"
    } else {
        "matte"
    };
    out.push_str(&format!("{inner}  \"type\": \"{kind}\",\n"));
    out.push_str(&format!(
        "{inner}  \"color\": [{}, {}, {}, {}],\n",
        number(material.color.x),
        number(material.color.y),
        number(material.color.z),
        number(material.color.w)
    ));
    out.push_str(&format!(
        "{inner}  \"ambient\": {},\n",
        number(material.ambient)
    ));
    out.push_str(&format!(
        "{inner}  \"diffuse\": {}",
        number(material.diffuse)
    ));
    // Only the fields each material type actually uses, so a matte surface does not
    // carry a meaningless `ior`.
    if kind == "mirror" {
        out.push_str(&format!(
            ",\n{inner}  \"reflectivity\": {}",
            number(material.reflectivity)
        ));
    } else if kind == "glass" {
        out.push_str(&format!(",\n{inner}  \"ior\": {}", number(material.ior)));
        out.push_str(&format!(
            ",\n{inner}  \"transparency\": {}",
            number(material.transparency)
        ));
    }
    // The texture is written as its path, resolved through the document's
    // texture table — indices are load-order artifacts and do not survive a
    // round trip on their own.
    if material.has_texture() {
        if let Some(path) = document.textures.get(material.texture_index as usize) {
            out.push_str(&format!(",\n{inner}  \"texture\": {}", quote(path)));
        }
    }
    if material.flags & crate::scene::component::material_flags::ALPHA_TEST != 0 {
        out.push_str(&format!(",\n{inner}  \"alpha_test\": true"));
    }
    out.push('\n');
    out.push_str(&format!("{inner}}}"));

    // Children, nested.
    let children = document.hierarchy.children(entity);
    if !children.is_empty() {
        out.push_str(",\n");
        out.push_str(&format!("{inner}\"children\": [\n"));
        for (i, child) in children.iter().enumerate() {
            write_entity(out, document, *child, depth + 2);
            if i + 1 < children.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str(&format!("{inner}]\n"));
    } else {
        out.push('\n');
    }
    out.push_str(&format!("{pad}}}"));
}

/// Write a scene to disk.
///
/// Writes to a temporary file and renames, so an interrupted save cannot leave a
/// half-written scene where the original was. The scene file is the authored asset;
/// losing it to a crash mid-write would be the worst possible failure of an editor.
pub fn save(path: impl AsRef<std::path::Path>, document: &SceneDocument<'_>) -> Result<()> {
    let path = path.as_ref();
    let json = to_json(document);
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, json.as_bytes())?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

/// Format a float for JSON.
///
/// Always with a decimal point, so `1` is written `1.0`: the loader accepts both, but
/// a human diffing the file should see that these are floats, and a value that
/// alternates between `1` and `1.0` across saves makes for noisy diffs.
fn number(value: f32) -> String {
    if !value.is_finite() {
        // JSON has no infinity or NaN. Zero rather than an invalid document, which
        // would fail to reload and lose the whole scene.
        return "0.0".to_string();
    }
    if value == value.trunc() && value.abs() < 1e7 {
        format!("{value:.1}")
    } else {
        // Trim trailing zeros so 0.5 is "0.5" rather than "0.50000000".
        let mut text = format!("{value:.6}");
        while text.ends_with('0') && !text.ends_with(".0") {
            text.pop();
        }
        text
    }
}

fn vec3_json(v: Vec3) -> String {
    format!("[{}, {}, {}]", number(v.x), number(v.y), number(v.z))
}

/// Quote and escape a JSON string.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // Control characters must be escaped or the document is invalid.
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::{radians, Transform, Vec4};
    use crate::scene::{parse_scene, plane_shape, sphere, sphere_shape, LoadedScene};

    /// Build a small scene by hand, with the mesh descriptors an editor would record.
    fn sample() -> (Scene, Camera, Vec<LightUniform>, Hierarchy, HashMap<u64, MeshSource>) {
        let mut scene = Scene::new();
        let mut hierarchy = Hierarchy::new();
        let mut sources = HashMap::new();

        let parent = scene.create_entity();
        scene.add_component(parent, sphere(Vec3::ZERO, 1.0, Vec3::new(0.8, 0.2, 0.2), 16, 24));
        scene.add_component(parent, ColliderComponent::new(sphere_shape(1.0)));
        scene.add_component(
            parent,
            MaterialComponent::mirror(Vec4::new(0.8, 0.2, 0.2, 1.0), 0.7),
        );
        scene.add_component(
            parent,
            TransformComponent {
                local: Transform {
                    position: Vec3::new(1.5, 2.0, -3.0),
                    rotation: Vec3::new(radians(30.0), 0.0, 0.0),
                    scale: Vec3::splat(2.0),
                },
            },
        );
        sources.insert(
            parent.id(),
            MeshSource::Sphere {
                radius: 1.0,
                rings: 16,
                segments: 24,
            },
        );

        let child = scene.create_entity();
        scene.add_component(child, sphere(Vec3::ZERO, 1.0, Vec3::new(0.2, 0.8, 0.2), 8, 12));
        scene.add_component(child, ColliderComponent::new(sphere_shape(1.0)));
        scene.add_component(
            child,
            MaterialComponent::glass(Vec4::new(0.2, 0.8, 0.2, 1.0), 1.5, 0.8),
        );
        scene.add_component(
            child,
            TransformComponent {
                local: Transform {
                    position: Vec3::new(0.0, 1.6, 0.0),
                    rotation: Vec3::ZERO,
                    scale: Vec3::splat(0.3),
                },
            },
        );
        sources.insert(
            child.id(),
            MeshSource::Sphere {
                radius: 1.0,
                rings: 8,
                segments: 12,
            },
        );
        hierarchy.set_parent(child, parent).expect("no cycle");

        let ground = scene.create_entity();
        scene.add_component(
            ground,
            crate::scene::plane(Vec3::ZERO, 50.0, Vec3::new(0.7, 0.7, 0.7)),
        );
        scene.add_component(ground, ColliderComponent::new(plane_shape(50.0)));
        scene.add_component(
            ground,
            MaterialComponent::matte(Vec4::new(0.7, 0.7, 0.7, 1.0), 0.15, 0.85),
        );
        scene.add_component(
            ground,
            TransformComponent {
                local: Transform {
                    position: Vec3::new(0.0, -2.0, 0.0),
                    ..Transform::identity()
                },
            },
        );
        sources.insert(ground.id(), MeshSource::Plane { size: 50.0 });

        let camera = Camera::new(
            Vec3::new(0.0, 3.0, 8.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 16.0 / 9.0, 0.1, 200.0),
        );
        let lights = vec![
            LightUniform::point(Vec3::new(0.0, 10.0, 0.0), Vec3::ONE, 0.1, 1.0),
            LightUniform::directional(
                Vec3::new(0.4, -0.3, 0.6),
                Vec3::new(0.4, 0.45, 0.6),
                0.05,
                0.35,
            ),
        ];
        (scene, camera, lights, hierarchy, sources)
    }

    fn round_trip(
        scene: &Scene,
        camera: &Camera,
        lights: &[LightUniform],
        hierarchy: &Hierarchy,
        sources: &HashMap<u64, MeshSource>,
    ) -> LoadedScene {
        let json = to_json(&SceneDocument {
            name: "round_trip",
            scene,
            camera,
            lights,
            hierarchy,
            mesh_sources: sources,
            textures: &[],
        });
        parse_scene(&json).unwrap_or_else(|e| panic!("saved scene failed to reload: {e}\n{json}"))
    }

    /// The property the whole module exists for, and Phase 6.1's completion
    /// criterion: what is saved must load back without loss.
    #[test]
    fn a_saved_scene_reloads_with_the_same_structure() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let loaded = round_trip(&scene, &camera, &lights, &hierarchy, &sources);

        assert_eq!(loaded.name, "round_trip");
        assert_eq!(loaded.lights.len(), 2);
        assert_eq!(
            loaded.scene.entities_with::<MeshComponent>().len(),
            scene.entities_with::<MeshComponent>().len(),
            "entity count changed across the round trip"
        );
        // The nested child must still be nested.
        let parented = loaded
            .scene
            .all_entities()
            .iter()
            .filter(|e| loaded.hierarchy.parent(**e).is_some())
            .count();
        assert_eq!(parented, 1, "the hierarchy was lost");
    }

    /// A textured, alpha-tested material must keep its texture path and flag
    /// across save/load — the стадия-1 addition to the round-trip property.
    #[test]
    fn textured_materials_survive_the_round_trip() {
        let mut scene = Scene::new();
        let hierarchy = Hierarchy::new();
        let mut sources = HashMap::new();

        let e = scene.create_entity();
        scene.add_component(e, sphere(Vec3::ZERO, 1.0, Vec3::ONE, 8, 12));
        scene.add_component(e, ColliderComponent::new(sphere_shape(1.0)));
        scene.add_component(
            e,
            MaterialComponent::matte(Vec4::new(1.0, 1.0, 1.0, 1.0), 0.1, 0.9)
                .with_texture(0)
                .with_alpha_test(),
        );
        scene.add_component(e, TransformComponent { local: Transform::identity() });
        sources.insert(e.id(), MeshSource::Sphere { radius: 1.0, rings: 8, segments: 12 });

        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 100.0),
        );
        let textures = vec!["assets/textures/checker.png".to_string()];
        let json = to_json(&SceneDocument {
            name: "textured",
            scene: &scene,
            camera: &camera,
            lights: &[],
            hierarchy: &hierarchy,
            mesh_sources: &sources,
            textures: &textures,
        });
        let loaded = parse_scene(&json)
            .unwrap_or_else(|e| panic!("textured scene failed to reload: {e}\n{json}"));

        assert_eq!(loaded.textures, textures, "the texture path must survive");
        let entity = loaded.scene.entities_with::<MaterialComponent>()[0];
        let m = loaded.scene.get_component::<MaterialComponent>(entity).unwrap();
        assert_eq!(m.texture_index, 0);
        assert!(
            m.flags & crate::scene::component::material_flags::ALPHA_TEST != 0,
            "alpha_test must survive"
        );
    }

    /// Transforms are what an editor changes, so losing one would defeat the point.
    #[test]
    fn transforms_survive_the_round_trip() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let loaded = round_trip(&scene, &camera, &lights, &hierarchy, &sources);

        let originals: Vec<Vec3> = scene
            .entities_with::<MeshComponent>()
            .into_iter()
            .filter_map(|e| scene.get_component::<TransformComponent>(e))
            .map(|t| t.local.position)
            .collect();
        let reloaded: Vec<Vec3> = loaded
            .scene
            .entities_with::<MeshComponent>()
            .into_iter()
            .filter_map(|e| loaded.scene.get_component::<TransformComponent>(e))
            .map(|t| t.local.position)
            .collect();
        assert_eq!(originals.len(), reloaded.len());
        for (a, b) in originals.iter().zip(reloaded.iter()) {
            assert!((*a - *b).length() < 1e-4, "position drifted: {a} -> {b}");
        }
    }

    /// Rotation is stored in radians and written in degrees, so a missing conversion
    /// would scale every rotation by 57.3 — visible, but only if something checks.
    #[test]
    fn rotation_round_trips_through_degrees() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let loaded = round_trip(&scene, &camera, &lights, &hierarchy, &sources);
        let rotations: Vec<Vec3> = loaded
            .scene
            .entities_with::<MeshComponent>()
            .into_iter()
            .filter_map(|e| loaded.scene.get_component::<TransformComponent>(e))
            .map(|t| t.local.rotation)
            .collect();
        // The sample's parent has a 30-degree X rotation, i.e. ~0.5236 radians.
        assert!(
            rotations
                .iter()
                .any(|r| (r.x - radians(30.0)).abs() < 1e-3),
            "no entity came back with the 30-degree rotation: {rotations:?}"
        );
    }

    #[test]
    fn scale_survives_the_round_trip() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let loaded = round_trip(&scene, &camera, &lights, &hierarchy, &sources);
        let scales: Vec<f32> = loaded
            .scene
            .entities_with::<MeshComponent>()
            .into_iter()
            .filter_map(|e| loaded.scene.get_component::<TransformComponent>(e))
            .map(|t| t.local.scale.x)
            .collect();
        assert!(scales.iter().any(|s| (*s - 2.0).abs() < 1e-4), "{scales:?}");
        assert!(scales.iter().any(|s| (*s - 0.3).abs() < 1e-4), "{scales:?}");
    }

    /// Material *type* has to survive, not just its colour: a mirror reloading as
    /// matte would silently change how the scene looks.
    #[test]
    fn material_types_and_their_parameters_survive() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let loaded = round_trip(&scene, &camera, &lights, &hierarchy, &sources);

        let materials: Vec<MaterialComponent> = loaded
            .scene
            .entities_with::<MeshComponent>()
            .into_iter()
            .filter_map(|e| loaded.scene.get_component::<MaterialComponent>(e))
            .copied()
            .collect();
        let mirror = materials
            .iter()
            .find(|m| m.material_type == MaterialType::Mirror as u32)
            .expect("the mirror material was lost");
        assert!((mirror.reflectivity - 0.7).abs() < 1e-4);
        let glass = materials
            .iter()
            .find(|m| m.material_type == MaterialType::Glass as u32)
            .expect("the glass material was lost");
        assert!((glass.ior - 1.5).abs() < 1e-4);
        assert!((glass.transparency - 0.8).abs() < 1e-4);
        assert!(
            materials
                .iter()
                .any(|m| m.material_type == MaterialType::Matte as u32),
            "the matte material was lost"
        );
    }

    #[test]
    fn the_camera_survives_the_round_trip() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let loaded = round_trip(&scene, &camera, &lights, &hierarchy, &sources);
        assert!((loaded.camera.position - camera.position).length() < 1e-4);
        assert!((loaded.camera.target - camera.target).length() < 1e-4);
        match (loaded.camera.projection, camera.projection) {
            (
                Projection::Perspective { fov_y: a, .. },
                Projection::Perspective { fov_y: b, .. },
            ) => assert!((a - b).abs() < 1e-3, "field of view drifted: {a} vs {b}"),
            other => panic!("projection type changed: {other:?}"),
        }
    }

    /// An orthographic camera takes a different branch, and one that silently wrote a
    /// perspective block would change the whole view.
    #[test]
    fn an_orthographic_camera_survives_too() {
        let (scene, _, lights, hierarchy, sources) = sample();
        let camera = Camera::new(
            Vec3::new(0.0, 5.0, 5.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::orthographic(-10.0, 10.0, -6.0, 6.0, 0.1, 100.0),
        );
        let loaded = round_trip(&scene, &camera, &lights, &hierarchy, &sources);
        assert!(
            loaded.camera.projection.is_orthographic(),
            "an orthographic camera came back as {:?}",
            loaded.camera.projection
        );
    }

    /// Both light types take different branches, and a point light written as
    /// directional would lose its position entirely.
    #[test]
    fn both_light_types_survive_with_their_parameters() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let loaded = round_trip(&scene, &camera, &lights, &hierarchy, &sources);

        let point = loaded
            .lights
            .iter()
            .find(|l| l.position[3] > 0.5)
            .expect("the point light was lost");
        assert!((point.position[1] - 10.0).abs() < 1e-4, "its position moved");
        let directional = loaded
            .lights
            .iter()
            .find(|l| l.position[3] < 0.5)
            .expect("the directional light was lost");
        assert!(
            (directional.direction[0] - 0.4).abs() < 1e-4,
            "its direction changed"
        );
        assert!((directional.diffuse - 0.35).abs() < 1e-4);
    }

    /// Saving twice must produce identical text. A writer that reordered its output
    /// would make every save a large diff regardless of what changed, which for a
    /// version-controlled asset is a real cost.
    #[test]
    fn saving_is_deterministic() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let document = SceneDocument {
            name: "stable",
            scene: &scene,
            camera: &camera,
            lights: &lights,
            hierarchy: &hierarchy,
            mesh_sources: &sources,
            textures: &[],
        };
        assert_eq!(to_json(&document), to_json(&document));
    }

    /// A save-load-save cycle must reach a fixed point, or repeated editing would
    /// drift the file even with no changes.
    #[test]
    fn a_second_round_trip_reaches_a_fixed_point() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let first = to_json(&SceneDocument {
            name: "fixed",
            scene: &scene,
            camera: &camera,
            lights: &lights,
            hierarchy: &hierarchy,
            mesh_sources: &sources,
            textures: &[],
        });
        let loaded = parse_scene(&first).expect("first save should reload");
        // The reloaded scene has no recorded descriptors, so they come from colliders
        // — which is exactly the editor-created-entity path.
        let second = to_json(&SceneDocument {
            name: "fixed",
            scene: &loaded.scene,
            camera: &loaded.camera,
            lights: &loaded.lights,
            hierarchy: &loaded.hierarchy,
            mesh_sources: &HashMap::new(),
            textures: &[],
        });
        let third_load = parse_scene(&second).expect("second save should reload");
        let third = to_json(&SceneDocument {
            name: "fixed",
            scene: &third_load.scene,
            camera: &third_load.camera,
            lights: &third_load.lights,
            hierarchy: &third_load.hierarchy,
            mesh_sources: &HashMap::new(),
            textures: &[],
        });
        assert_eq!(
            second, third,
            "repeated save/load cycles must converge rather than drifting"
        );
    }

    // --- formatting ---

    /// Floats are written with a decimal point so a human diffing the file can see
    /// they are floats, and so a value does not alternate between `1` and `1.0`.
    #[test]
    fn integral_floats_keep_a_decimal_point() {
        assert_eq!(number(1.0), "1.0");
        assert_eq!(number(-3.0), "-3.0");
        assert_eq!(number(0.0), "0.0");
    }

    #[test]
    fn fractional_floats_are_trimmed_but_not_truncated() {
        assert_eq!(number(0.5), "0.5");
        assert_eq!(number(1.25), "1.25");
        assert_eq!(number(-0.125), "-0.125");
    }

    /// JSON has no infinity or NaN. Writing one would produce a document that fails
    /// to reload, losing the whole scene rather than one field.
    #[test]
    fn non_finite_floats_become_zero_rather_than_invalid_json() {
        assert_eq!(number(f32::NAN), "0.0");
        assert_eq!(number(f32::INFINITY), "0.0");
        assert_eq!(number(f32::NEG_INFINITY), "0.0");
    }

    #[test]
    fn strings_are_escaped() {
        assert_eq!(quote("plain"), "\"plain\"");
        assert_eq!(quote("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(quote("back\\slash"), "\"back\\\\slash\"");
        assert_eq!(quote("line\nbreak"), "\"line\\nbreak\"");
        assert_eq!(quote("tab\there"), "\"tab\\there\"");
    }

    /// A control character in a name would make the document invalid.
    #[test]
    fn control_characters_are_escaped_as_unicode() {
        assert_eq!(quote("\u{0}"), "\"\\u0000\"");
        assert_eq!(quote("\u{1f}"), "\"\\u001f\"");
    }

    /// A name with a quote in it must not break the document — an editor lets the
    /// user type names.
    #[test]
    fn a_scene_name_with_a_quote_still_reloads() {
        let (scene, camera, lights, hierarchy, sources) = sample();
        let json = to_json(&SceneDocument {
            name: "he said \"hello\"",
            scene: &scene,
            camera: &camera,
            lights: &lights,
            hierarchy: &hierarchy,
            mesh_sources: &sources,
            textures: &[],
        });
        let loaded = parse_scene(&json).expect("a quoted name must not break the document");
        assert_eq!(loaded.name, "he said \"hello\"");
    }

    // --- edge cases ---

    #[test]
    fn an_empty_scene_produces_a_loadable_document() {
        let scene = Scene::new();
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 100.0),
        );
        let json = to_json(&SceneDocument {
            name: "empty",
            scene: &scene,
            camera: &camera,
            lights: &[],
            hierarchy: &Hierarchy::new(),
            mesh_sources: &HashMap::new(),
            textures: &[],
        });
        let loaded = parse_scene(&json).expect("an empty scene must still be valid JSON");
        assert!(loaded.scene.all_entities().is_empty());
        assert!(loaded.lights.is_empty());
    }

    /// An entity with no recorded descriptor takes the collider-inference path, which
    /// is what every editor-created entity does.
    #[test]
    fn an_entity_without_a_recorded_descriptor_is_inferred_from_its_collider() {
        let mut scene = Scene::new();
        let e = scene.create_entity();
        scene.add_component(e, sphere(Vec3::ZERO, 2.5, Vec3::ONE, 8, 8));
        scene.add_component(e, ColliderComponent::new(sphere_shape(2.5)));
        scene.add_component(
            e,
            TransformComponent {
                local: Transform::identity(),
            },
        );
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 100.0),
        );
        let json = to_json(&SceneDocument {
            name: "inferred",
            scene: &scene,
            camera: &camera,
            lights: &[],
            hierarchy: &Hierarchy::new(),
            mesh_sources: &HashMap::new(),
            textures: &[],
        });
        let loaded = parse_scene(&json).expect("should reload");
        assert_eq!(loaded.scene.entities_with::<MeshComponent>().len(), 1);
        // The radius came from the collider, so the reloaded sphere is the right size.
        let collider = loaded
            .scene
            .entities_with::<MeshComponent>()
            .into_iter()
            .filter_map(|e| loaded.scene.get_component::<ColliderComponent>(e))
            .next()
            .expect("the reloaded entity should have a collider");
        match collider.shape {
            Shape::Sphere { radius } => assert!((radius - 2.5).abs() < 1e-4, "got {radius}"),
            other => panic!("expected a sphere, got {other:?}"),
        }
    }

    /// Boxes gained a scene-format spelling with texturing (стадия 1), so a
    /// box collider now round-trips as a box rather than its enclosing sphere.
    #[test]
    fn a_box_collider_is_written_as_a_box() {
        let half = Vec3::new(1.0, 2.0, 2.0);
        let source = MeshSource::from_shape(&Shape::Box { half_extents: half });
        assert_eq!(source, MeshSource::Box { half_extents: half });
    }

    /// And the full round trip: a box mesh entity saves and reloads as a box.
    #[test]
    fn a_box_mesh_survives_the_round_trip() {
        let mut scene = Scene::new();
        let half = Vec3::new(0.5, 1.0, 0.25);
        let e = scene.create_entity();
        scene.add_component(e, crate::scene::box_mesh(half, Vec3::ONE));
        scene.add_component(e, ColliderComponent::new(Shape::Box { half_extents: half }));
        scene.add_component(e, MaterialComponent::default());
        scene.add_component(e, TransformComponent { local: Transform::identity() });
        let mut sources = HashMap::new();
        sources.insert(e.id(), MeshSource::Box { half_extents: half });

        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 100.0),
        );
        let json = to_json(&SceneDocument {
            name: "boxed",
            scene: &scene,
            camera: &camera,
            lights: &[],
            hierarchy: &Hierarchy::new(),
            mesh_sources: &sources,
            textures: &[],
        });
        let loaded = parse_scene(&json)
            .unwrap_or_else(|err| panic!("box scene failed to reload: {err}\n{json}"));
        let entity = loaded.scene.entities_with::<MeshComponent>()[0];
        let collider = loaded.scene.get_component::<ColliderComponent>(entity).unwrap();
        match collider.shape {
            Shape::Box { half_extents } => {
                assert!((half_extents - half).length() < 1e-5, "got {half_extents}")
            }
            other => panic!("expected a box, got {other:?}"),
        }
        let mesh = loaded.scene.get_component::<MeshComponent>(entity).unwrap();
        assert_eq!(mesh.vertices.len(), 24, "the box mesh regenerated per-face");
    }

    /// Writing to disk uses a temporary file and a rename, so an interrupted save
    /// cannot leave a half-written scene where the original was.
    #[test]
    fn saving_to_disk_leaves_no_temporary_behind() {
        let dir = std::env::temp_dir().join(format!(
            "astraglyph_writer_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("out.json");

        let (scene, camera, lights, hierarchy, sources) = sample();
        save(
            &path,
            &SceneDocument {
                name: "saved",
                scene: &scene,
                camera: &camera,
                lights: &lights,
                hierarchy: &hierarchy,
                mesh_sources: &sources,
                textures: &[],
            },
        )
        .expect("save should succeed");

        assert!(path.exists(), "the scene file should exist");
        assert!(
            !path.with_extension("json.tmp").exists(),
            "the temporary file should have been renamed away"
        );
        // And it loads.
        let loaded = crate::scene::load_scene_file(&path).expect("the saved file should load");
        assert_eq!(loaded.name, "saved");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
