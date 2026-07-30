// Scene module: entity, component, scene, camera, primitives, hierarchy,
// frustum culling, material registry.

pub mod archetype;
pub mod camera;
pub mod camera_rig;
pub mod component;
pub mod entity;
pub mod frustum;
pub mod hierarchy;
pub mod loader;
pub mod material_registry;
pub mod primitives;
pub mod scene;

pub use camera::{Camera, Projection};
pub use camera_rig::{CameraMode, CameraRig};
pub use hierarchy::Hierarchy;
pub use material_registry::MaterialRegistry;
// Public API of these modules that the engine itself doesn't call yet — they are
// exported for scene authors and covered by their own tests.
#[allow(unused_imports)]
pub use frustum::{Aabb, Frustum, Plane};
#[allow(unused_imports)]
pub use loader::{load_scene_file, parse_scene, LoadedScene};
#[allow(unused_imports)]
pub use component::{
    ColliderComponent, MaterialComponent, MaterialType, MaterialUniform, MeshComponent, MeshVertex,
    TransformComponent,
};
pub use entity::Entity;
// `box_shape` is exported from `primitives` but not re-exported here: the only
// box geometry in the engine is the Cornell Box demo, which bakes world
// positions into its vertices instead of using transforms, so a local-space
// collider would sit at the origin rather than at the wall.
pub use primitives::{plane, plane_shape, sphere, sphere_shape};
pub use scene::Scene;