// Material spheres demo scene.
//
// A minimal scene for testing materials and lighting:
// - White horizontal plane (ground)
// - 3 spheres: red (matte), blue (mirror), green (glass)
//   arranged in a triangle, close enough for mutual influence,
//   far enough to view from all angles.
// - Single point light from above (straight down).

use crate::engine::math::{radians, Vec3, Vec4};
use crate::renderer::LightUniform;
use crate::scene::{Camera, Entity, MaterialComponent, MeshComponent, Projection, Scene, plane, sphere};

/// Build the material spheres demo scene.
///
/// Returns the scene and a camera positioned to view all three spheres.
pub fn build_scene() -> (Scene, Camera) {
    let mut scene = Scene::new();

    // --- Ground plane (white matte) ---
    let ground = plane(Vec3::new(0.0, -2.0, 0.0), 50.0, Vec3::new(0.8, 0.8, 0.8));
    add_mesh_entity(
        &mut scene,
        ground,
        MaterialComponent::matte(Vec4::new(0.8, 0.8, 0.8, 1.0), 0.15, 0.85),
    );

    // --- Three spheres ---
    // Red — matte (high diffuse, low specular, low shininess)
    let red_sphere = sphere(
        Vec3::new(-2.5, -0.5, 0.0),
        1.5,
        Vec3::new(0.85, 0.1, 0.1),
        24,
        32,
    );
    add_mesh_entity(
        &mut scene,
        red_sphere,
        MaterialComponent::matte(Vec4::new(0.85, 0.1, 0.1, 1.0), 0.1, 0.9),
    );

    // Blue — mirror (high specular, high shininess, high reflectivity)
    let blue_sphere = sphere(
        Vec3::new(2.5, -0.5, 0.0),
        1.5,
        Vec3::new(0.1, 0.2, 0.85),
        24,
        32,
    );
    add_mesh_entity(
        &mut scene,
        blue_sphere,
        MaterialComponent::mirror(Vec4::new(0.1, 0.2, 0.85, 1.0), 0.8),
    );

    // Green — glass (transparent, refractive, IOR=1.5)
    let green_sphere = sphere(
        Vec3::new(0.0, -0.5, -2.5),
        1.5,
        Vec3::new(0.1, 0.75, 0.2),
        24,
        32,
    );
    add_mesh_entity(
        &mut scene,
        green_sphere,
        MaterialComponent::glass(Vec4::new(0.1, 0.75, 0.2, 1.0), 1.5, 0.8),
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

/// Add a mesh + material entity to the scene.
fn add_mesh_entity(scene: &mut Scene, mesh: MeshComponent, material: MaterialComponent) -> Entity {
    let entity = scene.create_entity();
    scene.add_component(entity, mesh);
    scene.add_component(entity, material);
    entity
}
