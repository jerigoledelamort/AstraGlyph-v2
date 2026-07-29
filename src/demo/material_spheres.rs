// Material spheres demo scene.
//
// A minimal scene for testing materials and lighting:
// - White horizontal plane (ground)
// - 3 spheres: red (matte), blue (mirror), green (glass)
//   arranged in a triangle, close enough for mutual influence,
//   far enough to view from all angles.
// - A point key light from above plus a weak directional fill light.
//
// The spheres' geometry is built once at the origin and placed via
// TransformComponent, so this scene also exercises the model-matrix path
// (Phase 2.1) rather than baking world positions into vertex data.

use crate::engine::geometry::Shape;
use crate::engine::math::{radians, Transform, Vec3, Vec4};
use crate::renderer::LightUniform;
use crate::scene::{
    plane, plane_shape, sphere, sphere_shape, Camera, ColliderComponent, Entity, MaterialComponent,
    MeshComponent, Projection, Scene, TransformComponent,
};

/// Build the material spheres demo scene.
///
/// Returns the scene and a camera positioned to view all three spheres.
pub fn build_scene() -> (Scene, Camera) {
    let mut scene = Scene::new();

    // --- Ground plane (white matte) ---
    // Built at the origin and lowered by its transform.
    let ground = plane(Vec3::ZERO, 50.0, Vec3::new(0.8, 0.8, 0.8));
    add_mesh_entity(
        &mut scene,
        ground,
        plane_shape(50.0),
        MaterialComponent::matte(Vec4::new(0.8, 0.8, 0.8, 1.0), 0.15, 0.85),
        Transform {
            position: Vec3::new(0.0, -2.0, 0.0),
            ..Transform::identity()
        },
    );

    // --- Three spheres ---
    // All three share one unit-radius mesh built at the origin; position and
    // size come from their transforms.
    let unit_sphere = |color: Vec3| sphere(Vec3::ZERO, 1.0, color, 24, 32);
    let placed = |position: Vec3| Transform {
        position,
        scale: Vec3::splat(1.5),
        ..Transform::identity()
    };

    // Red — matte (high diffuse, low specular, low shininess)
    add_mesh_entity(
        &mut scene,
        unit_sphere(Vec3::new(0.85, 0.1, 0.1)),
        sphere_shape(1.0),
        MaterialComponent::matte(Vec4::new(0.85, 0.1, 0.1, 1.0), 0.1, 0.9),
        placed(Vec3::new(-2.5, -0.5, 0.0)),
    );

    // Blue — mirror (high specular, high shininess, high reflectivity)
    add_mesh_entity(
        &mut scene,
        unit_sphere(Vec3::new(0.1, 0.2, 0.85)),
        sphere_shape(1.0),
        MaterialComponent::mirror(Vec4::new(0.1, 0.2, 0.85, 1.0), 0.8),
        placed(Vec3::new(2.5, -0.5, 0.0)),
    );

    // Green — glass (transparent, refractive, IOR=1.5)
    add_mesh_entity(
        &mut scene,
        unit_sphere(Vec3::new(0.1, 0.75, 0.2)),
        sphere_shape(1.0),
        MaterialComponent::glass(Vec4::new(0.1, 0.75, 0.2, 1.0), 1.5, 0.8),
        placed(Vec3::new(0.0, -0.5, -2.5)),
    );

    // --- Camera ---
    // Positioned at an angle to see all three spheres.
    let camera = Camera::new(
        Vec3::new(0.0, 3.0, 8.0),
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::UNIT_Y,
        Projection::perspective(radians(60.0), 16.0 / 9.0, 0.1, 200.0),
    );

    (scene, camera)
}

/// Default light direction for the material spheres scene.
/// Point light from directly above (pointing straight down).
pub fn default_light() -> Vec3 {
    Vec3::new(0.0, -1.0, 0.0).normalize()
}

/// Lights for the material spheres scene: a bright point key light from
/// above, plus a weak directional fill light so shadowed sides aren't
/// pitch black. Demonstrates summing multiple, differently-typed lights.
pub fn lights() -> Vec<LightUniform> {
    vec![
        LightUniform::point(Vec3::new(0.0, 10.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 0.1, 1.0),
        LightUniform::directional(Vec3::new(0.4, -0.3, 0.6), Vec3::new(0.4, 0.45, 0.6), 0.05, 0.35),
    ]
}

/// Add a mesh + material + transform entity to the scene.
fn add_mesh_entity(
    scene: &mut Scene,
    mesh: MeshComponent,
    shape: Shape,
    material: MaterialComponent,
    transform: Transform,
) -> Entity {
    let entity = scene.create_entity();
    scene.add_component(entity, mesh);
    // The analytic form of the same geometry, for the CPU tracer and physics.
    // Attached here rather than derived later so it cannot disagree with the mesh.
    scene.add_component(entity, ColliderComponent::new(shape));
    scene.add_component(entity, material);
    scene.add_component(entity, TransformComponent { local: transform });
    entity
}
