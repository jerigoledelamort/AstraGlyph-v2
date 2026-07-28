// Cornell Box demo scene geometry.
// Classic test scene: a box with colored walls, two boxes inside, and a light source.
//
// Dimensions follow the original Cornell Box specification:
// - Box: 555 x 555 x 555 units
// - Left wall: red, Right wall: green, Back wall: white, Floor/Ceiling: white
// - Two boxes inside

use crate::engine::math::{Vec3, Vec4};
use crate::scene::{
    Camera, Entity, MaterialComponent, MeshComponent, MeshVertex, Projection, Scene,
};
use crate::engine::math::radians;

/// Build the Cornell Box scene.
///
/// Returns the scene and the camera positioned inside the box.
pub fn build_scene() -> (Scene, Camera) {
    let mut scene = Scene::new();

    // Box half-size (we center the box at origin for simplicity).
    let s = 277.5; // half of 555

    // --- Walls ---
    // Each wall is a pair of triangles (4 vertices, 6 indices).

    let white = Vec3::new(0.73, 0.73, 0.73);
    let red = Vec3::new(0.63, 0.065, 0.065);
    let green = Vec3::new(0.12, 0.45, 0.15);

    // Floor (white)
    let floor = create_box_wall(
        Vec3::new(-s, -s, -s),
        Vec3::new(s, -s, -s),
        Vec3::new(s, -s, s),
        Vec3::new(-s, -s, s),
        Vec3::UNIT_Y,
        white,
    );
    add_mesh_entity(&mut scene, floor, MaterialComponent::new(
        Vec4::new(0.73, 0.73, 0.73, 1.0), 0.1, 0.9,
    ));

    // Ceiling (white)
    let ceiling = create_box_wall(
        Vec3::new(-s, s, -s),
        Vec3::new(-s, s, s),
        Vec3::new(s, s, s),
        Vec3::new(s, s, -s),
        Vec3::new(0.0, -1.0, 0.0),
        white,
    );
    add_mesh_entity(&mut scene, ceiling, MaterialComponent::new(
        Vec4::new(0.73, 0.73, 0.73, 1.0), 0.1, 0.9,
    ));

    // Back wall (white)
    let back = create_box_wall(
        Vec3::new(-s, -s, -s),
        Vec3::new(-s, s, -s),
        Vec3::new(s, s, -s),
        Vec3::new(s, -s, -s),
        Vec3::UNIT_Z,
        white,
    );
    add_mesh_entity(&mut scene, back, MaterialComponent::new(
        Vec4::new(0.73, 0.73, 0.73, 1.0), 0.1, 0.9,
    ));

    // Left wall (red)
    let left = create_box_wall(
        Vec3::new(-s, -s, -s),
        Vec3::new(-s, -s, s),
        Vec3::new(-s, s, s),
        Vec3::new(-s, s, -s),
        Vec3::UNIT_X,
        red,
    );
    add_mesh_entity(&mut scene, left, MaterialComponent::new(
        Vec4::new(0.63, 0.065, 0.065, 1.0), 0.1, 0.9,
    ));

    // Right wall (green)
    let right = create_box_wall(
        Vec3::new(s, -s, -s),
        Vec3::new(s, s, -s),
        Vec3::new(s, s, s),
        Vec3::new(s, -s, s),
        Vec3::new(-1.0, 0.0, 0.0),
        green,
    );
    add_mesh_entity(&mut scene, right, MaterialComponent::new(
        Vec4::new(0.12, 0.45, 0.15, 1.0), 0.1, 0.9,
    ));

    // --- Tall box (left side) ---
    let tall_box = create_box(
        Vec3::new(-150.0, -s, -100.0),
        Vec3::new(80.0, 300.0, 80.0),
        white,
    );
    add_mesh_entity(&mut scene, tall_box, MaterialComponent::new(
        Vec4::new(0.73, 0.73, 0.73, 1.0), 0.1, 0.9,
    ));

    // --- Short box (right side) ---
    let short_box = create_box(
        Vec3::new(150.0, -s, 50.0),
        Vec3::new(80.0, 150.0, 80.0),
        white,
    );
    add_mesh_entity(&mut scene, short_box, MaterialComponent::new(
        Vec4::new(0.73, 0.73, 0.73, 1.0), 0.1, 0.9,
    ));

    // Camera positioned inside the box looking toward the back wall.
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, s - 50.0),
        Vec3::new(0.0, 0.0, -s),
        Vec3::UNIT_Y,
        Projection::perspective(radians(60.0), 16.0 / 9.0, 1.0, 1000.0),
    );

    (scene, camera)
}

/// Create a rectangular wall from 4 corner vertices (CCW from outside).
fn create_box_wall(
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    normal: Vec3,
    color: Vec3,
) -> MeshComponent {
    let vertices = vec![
        MeshVertex { position: v0, normal, color },
        MeshVertex { position: v1, normal, color },
        MeshVertex { position: v2, normal, color },
        MeshVertex { position: v3, normal, color },
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    MeshComponent::new(vertices, indices)
}

/// Create a 3D box mesh centered at `center` with half-extents `half`.
fn create_box(center: Vec3, half: Vec3, color: Vec3) -> MeshComponent {
    let cx = center.x;
    let cy = center.y;
    let cz = center.z;
    let hx = half.x;
    let hy = half.y;
    let hz = half.z;

    // 8 corners
    let p = |x: f32, y: f32, z: f32| Vec3::new(cx + x, cy + y, cz + z);

    let v = [
        // Front face (z = -hz, normal = -Z)
        p(-hx, -hy, -hz), p(hx, -hy, -hz), p(hx, hy, -hz), p(-hx, hy, -hz),
        // Back face (z = +hz, normal = +Z)
        p(-hx, -hy, hz), p(hx, -hy, hz), p(hx, hy, hz), p(-hx, hy, hz),
        // Left face (x = -hx, normal = -X)
        p(-hx, -hy, -hz), p(-hx, hy, -hz), p(-hx, hy, hz), p(-hx, -hy, hz),
        // Right face (x = +hx, normal = +X)
        p(hx, -hy, -hz), p(hx, -hy, hz), p(hx, hy, hz), p(hx, hy, -hz),
        // Bottom face (y = -hy, normal = -Y)
        p(-hx, -hy, -hz), p(-hx, -hy, hz), p(hx, -hy, hz), p(hx, -hy, -hz),
        // Top face (y = +hy, normal = +Y)
        p(-hx, hy, -hz), p(hx, hy, -hz), p(hx, hy, hz), p(-hx, hy, hz),
    ];

    let normals = [
        Vec3::new(0.0, 0.0, -1.0),  // front
        Vec3::new(0.0, 0.0, 1.0),   // back
        Vec3::new(-1.0, 0.0, 0.0),  // left
        Vec3::new(1.0, 0.0, 0.0),   // right
        Vec3::new(0.0, -1.0, 0.0),  // bottom
        Vec3::new(0.0, 1.0, 0.0),   // top
    ];

    let mut vertices = Vec::with_capacity(24);
    for face in 0..6 {
        for i in 0..4 {
            vertices.push(MeshVertex {
                position: v[face * 4 + i],
                normal: normals[face],
                color,
            });
        }
    }

    let indices: Vec<u32> = (0..6u32)
        .flat_map(|f| {
            let base = f * 4;
            [base, base + 1, base + 2, base, base + 2, base + 3]
        })
        .collect();

    MeshComponent::new(vertices, indices)
}

/// Add a mesh + material entity to the scene.
fn add_mesh_entity(scene: &mut Scene, mesh: MeshComponent, material: MaterialComponent) -> Entity {
    let entity = scene.create_entity();
    scene.add_component(entity, mesh);
    scene.add_component(entity, material);
    entity
}

/// Default light direction for the Cornell box (pointing down and slightly forward).
pub fn default_light() -> Vec3 {
    Vec3::new(0.0, -1.0, -0.3).normalize()
}