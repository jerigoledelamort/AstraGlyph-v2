// Cornell Box demo scene geometry.
// Classic test scene: a box with colored walls, two boxes inside, and a light source.
//
// Dimensions follow the original Cornell Box specification:
// - Box: 555 x 555 x 555 units
// - Left wall: red, Right wall: green, Back wall: white, Floor/Ceiling: white
// - Two boxes inside
//
// All geometry is authored at the origin and placed with a `TransformComponent`,
// like every other scene in the engine. This used to bake world positions into
// the vertices instead, which had two costs: entities without local-space
// geometry cannot carry analytic colliders (a collider at the origin is not
// where the wall is), and the scene exercised none of the model-matrix path it
// was supposed to be a regression reference for. The world-space result is
// vertex-for-vertex identical — pinned by a test below, because "visually the
// same" is exactly the kind of claim that silently rots.

use crate::engine::math::{radians, Transform, Vec3, Vec4};
use crate::engine::geometry::Shape;
use crate::renderer::LightUniform;
use crate::scene::{
    box_mesh, box_shape, plane, Camera, ColliderComponent, Entity, MaterialComponent,
    MeshComponent, Projection, Scene, TransformComponent,
};

/// Box half-size (the box is centred at the origin; 555 / 2).
const HALF: f32 = 277.5;

/// Build the Cornell Box scene.
///
/// Returns the scene and the camera positioned inside the box.
pub fn build_scene() -> (Scene, Camera) {
    let mut scene = Scene::new();
    let s = HALF;

    let white = Vec3::new(0.73, 0.73, 0.73);
    let red = Vec3::new(0.63, 0.065, 0.065);
    let green = Vec3::new(0.12, 0.45, 0.15);
    let matte = |c: Vec3| MaterialComponent::new(Vec4::new(c.x, c.y, c.z, 1.0), 0.1, 0.9);

    // Each wall is the same origin-authored quad — a `plane` of half-extent `s`
    // facing +Y — rotated so its normal points into the box and translated onto
    // its face. Rotations are single-axis, so the Euler composition order in
    // `Transform::to_matrix` cannot reorder anything.
    let wall = |color: Vec3, rotation_deg: Vec3, position: Vec3| -> (MeshComponent, Transform) {
        (
            plane(Vec3::ZERO, s, color),
            Transform::new(
                position,
                Vec3::new(
                    radians(rotation_deg.x),
                    radians(rotation_deg.y),
                    radians(rotation_deg.z),
                ),
                Vec3::ONE,
            ),
        )
    };
    // The collider matches the quad: a bounded plane, +Y in local space like the
    // mesh, rotated into place by the same transform (`Shape::transformed`
    // rotates the normal by the model matrix).
    let wall_shape = Shape::Plane {
        normal: Vec3::UNIT_Y,
        half_size: s,
    };

    // Floor (white), normal +Y.
    let (mesh, transform) = wall(white, Vec3::ZERO, Vec3::new(0.0, -s, 0.0));
    add_entity(&mut scene, mesh, wall_shape, matte(white), transform);

    // Ceiling (white), normal -Y: flipped over about X.
    let (mesh, transform) = wall(white, Vec3::new(180.0, 0.0, 0.0), Vec3::new(0.0, s, 0.0));
    add_entity(&mut scene, mesh, wall_shape, matte(white), transform);

    // Back wall (white), normal +Z: +Y rotated 90 degrees about X.
    let (mesh, transform) = wall(white, Vec3::new(90.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -s));
    add_entity(&mut scene, mesh, wall_shape, matte(white), transform);

    // Left wall (red), normal +X: +Y rotated -90 degrees about Z.
    let (mesh, transform) = wall(red, Vec3::new(0.0, 0.0, -90.0), Vec3::new(-s, 0.0, 0.0));
    add_entity(&mut scene, mesh, wall_shape, matte(red), transform);

    // Right wall (green), normal -X: +Y rotated +90 degrees about Z.
    let (mesh, transform) = wall(green, Vec3::new(0.0, 0.0, 90.0), Vec3::new(s, 0.0, 0.0));
    add_entity(&mut scene, mesh, wall_shape, matte(green), transform);

    // --- Tall box (left side) ---
    let tall_half = Vec3::new(80.0, 300.0, 80.0);
    add_entity(
        &mut scene,
        box_mesh(tall_half, white),
        box_shape(tall_half),
        matte(white),
        Transform {
            position: Vec3::new(-150.0, -s, -100.0),
            ..Transform::identity()
        },
    );

    // --- Short box (right side) ---
    let short_half = Vec3::new(80.0, 150.0, 80.0);
    add_entity(
        &mut scene,
        box_mesh(short_half, white),
        box_shape(short_half),
        matte(white),
        Transform {
            position: Vec3::new(150.0, -s, 50.0),
            ..Transform::identity()
        },
    );

    // Camera positioned inside the box looking toward the back wall.
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, s - 50.0),
        Vec3::new(0.0, 0.0, -s),
        Vec3::UNIT_Y,
        Projection::perspective(radians(60.0), 16.0 / 9.0, 1.0, 1000.0),
    );

    (scene, camera)
}

/// Add a mesh + collider + material + transform entity to the scene.
fn add_entity(
    scene: &mut Scene,
    mesh: MeshComponent,
    shape: Shape,
    material: MaterialComponent,
    transform: Transform,
) -> Entity {
    let entity = scene.create_entity();
    scene.add_component(entity, mesh);
    scene.add_component(entity, ColliderComponent::new(shape));
    scene.add_component(entity, material);
    scene.add_component(entity, TransformComponent { local: transform });
    entity
}

/// Default light direction for the Cornell box (pointing down and slightly forward).
pub fn default_light() -> Vec3 {
    Vec3::new(0.0, -1.0, -0.3).normalize()
}

/// Lights for the Cornell box: a ceiling point light (the classic Cornell
/// box light panel) plus a very weak directional fill so the shadowed
/// (unlit) walls aren't pure black.
pub fn lights() -> Vec<LightUniform> {
    vec![
        LightUniform::point(Vec3::new(0.0, 250.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 0.1, 1.0),
        LightUniform::directional(default_light(), Vec3::new(1.0, 1.0, 1.0), 0.02, 0.15),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::geometry::WorldShape;

    /// World-space triangles of every mesh entity, transformed by its own
    /// transform — the same thing the renderer computes per frame.
    fn world_triangles(scene: &Scene) -> Vec<[Vec3; 3]> {
        let mut out = Vec::new();
        for entity in scene.entities_with::<MeshComponent>() {
            let mesh = scene.get_component::<MeshComponent>(entity).unwrap();
            let model = scene
                .get_component::<TransformComponent>(entity)
                .map(|t| t.world_matrix())
                .unwrap_or(crate::engine::math::Mat4::IDENTITY);
            for tri in mesh.indices.chunks(3) {
                out.push([
                    model.transform_point(mesh.vertices[tri[0] as usize].position),
                    model.transform_point(mesh.vertices[tri[1] as usize].position),
                    model.transform_point(mesh.vertices[tri[2] as usize].position),
                ]);
            }
        }
        out
    }

    /// The transform rewrite must not move a single vertex: the box interior
    /// spans exactly [-HALF, HALF] on every axis, and each wall's corners land
    /// on the corners of its face.
    #[test]
    fn walls_land_exactly_where_the_baked_geometry_was() {
        let (scene, _) = build_scene();
        let s = HALF;
        // Collect every wall vertex in world space (walls are the five quads,
        // i.e. the meshes with 4 vertices).
        let mut wall_corners: Vec<Vec3> = Vec::new();
        for entity in scene.entities_with::<MeshComponent>() {
            let mesh = scene.get_component::<MeshComponent>(entity).unwrap();
            if mesh.vertices.len() != 4 {
                continue;
            }
            let model = scene
                .get_component::<TransformComponent>(entity)
                .unwrap()
                .world_matrix();
            for v in &mesh.vertices {
                wall_corners.push(model.transform_point(v.position));
            }
        }
        assert_eq!(wall_corners.len(), 20, "five walls of four corners each");
        // Every corner must be a corner of the box on the wall's two axes.
        for corner in &wall_corners {
            for value in [corner.x, corner.y, corner.z] {
                assert!(
                    (value.abs() - s).abs() < 1e-3,
                    "wall corner {corner} is not on the box surface"
                );
            }
        }
        // Each cube corner is shared by at least two walls (floor/ceiling meet
        // the side walls there), so all 8 corners must appear.
        for x in [-s, s] {
            for y in [-s, s] {
                for z in [-s, s] {
                    let expected = Vec3::new(x, y, z);
                    assert!(
                        wall_corners.iter().any(|c| (*c - expected).length() < 1e-3),
                        "no wall touches the box corner {expected}"
                    );
                }
            }
        }
    }

    /// Wall normals must point into the box after the transform — outward-facing
    /// walls would be back-face culled and the scene would render as five holes.
    #[test]
    fn wall_normals_point_into_the_box() {
        let (scene, _) = build_scene();
        for entity in scene.entities_with::<MeshComponent>() {
            let mesh = scene.get_component::<MeshComponent>(entity).unwrap();
            if mesh.vertices.len() != 4 {
                continue;
            }
            let model = scene
                .get_component::<TransformComponent>(entity)
                .unwrap()
                .world_matrix();
            let world_normal = model.transform_dir(mesh.vertices[0].normal).normalize();
            let world_center = model.transform_point(Vec3::ZERO);
            // Pointing into the box = pointing from the wall toward the origin.
            let inward = (-world_center).normalize();
            assert!(
                world_normal.dot(inward) > 0.99,
                "wall at {world_center} has normal {world_normal}, expected {inward}"
            );
            // Geometric winding must agree, or the back-face cull eats the wall.
            let a = model.transform_point(mesh.vertices[0].position);
            let b = model.transform_point(mesh.vertices[1].position);
            let c = model.transform_point(mesh.vertices[2].position);
            let geometric = (b - a).cross(c - a).normalize();
            assert!(
                geometric.dot(world_normal) > 0.99,
                "winding disagrees with normal on the wall at {world_center}"
            );
        }
    }

    /// The whole point of 0.3: every entity now carries a collider, and the
    /// collider agrees with where the geometry actually is.
    #[test]
    fn every_entity_has_a_collider_in_the_right_place() {
        let (scene, _) = build_scene();
        let entities = scene.entities_with::<MeshComponent>();
        assert_eq!(entities.len(), 7, "five walls and two boxes");
        for entity in entities {
            let collider = scene
                .get_component::<ColliderComponent>(entity)
                .expect("every Cornell entity carries a collider");
            let model = scene
                .get_component::<TransformComponent>(entity)
                .expect("every Cornell entity carries a transform")
                .world_matrix();
            let world: WorldShape = collider.shape.transformed(&model);
            // The collider's world origin must sit inside (or on) the box.
            for value in [world.origin.x, world.origin.y, world.origin.z] {
                assert!(
                    value.abs() <= HALF + 1e-3,
                    "collider origin {} escaped the box",
                    world.origin
                );
            }
        }
    }

    /// The two inner boxes keep their old world-space bounds.
    #[test]
    fn inner_boxes_keep_their_baked_bounds() {
        let (scene, _) = build_scene();
        let mut box_bounds: Vec<(Vec3, Vec3)> = Vec::new();
        for entity in scene.entities_with::<MeshComponent>() {
            let mesh = scene.get_component::<MeshComponent>(entity).unwrap();
            if mesh.vertices.len() != 24 {
                continue;
            }
            let model = scene
                .get_component::<TransformComponent>(entity)
                .unwrap()
                .world_matrix();
            let mut min = Vec3::splat(f32::INFINITY);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for v in &mesh.vertices {
                let p = model.transform_point(v.position);
                min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
                max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
            }
            box_bounds.push((min, max));
        }
        assert_eq!(box_bounds.len(), 2, "the tall box and the short box");
        // The baked originals: center (-150, -HALF, -100) half (80, 300, 80)
        // and center (150, -HALF, 50) half (80, 150, 80).
        let expected = [
            (
                Vec3::new(-230.0, -HALF - 300.0, -180.0),
                Vec3::new(-70.0, -HALF + 300.0, -20.0),
            ),
            (
                Vec3::new(70.0, -HALF - 150.0, -30.0),
                Vec3::new(230.0, -HALF + 150.0, 130.0),
            ),
        ];
        for (want_min, want_max) in expected {
            assert!(
                box_bounds
                    .iter()
                    .any(|(min, max)| (*min - want_min).length() < 1e-3
                        && (*max - want_max).length() < 1e-3),
                "no box matches bounds {want_min}..{want_max}; got {box_bounds:?}"
            );
        }
    }

    #[test]
    fn triangle_count_is_unchanged_by_the_rewrite() {
        let (scene, _) = build_scene();
        // 5 walls * 2 triangles + 2 boxes * 12 triangles = 34.
        assert_eq!(world_triangles(&scene).len(), 34);
    }
}
