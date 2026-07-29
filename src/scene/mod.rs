// Scene module: entity, component, scene, camera, primitives.

pub mod camera;
pub mod component;
pub mod entity;
pub mod primitives;
pub mod scene;

pub use camera::{Camera, Projection};
#[allow(unused_imports)]
pub use component::{MaterialComponent, MaterialType, MaterialUniform, MeshComponent, MeshVertex};
pub use entity::Entity;
pub use primitives::{plane, sphere};
pub use scene::Scene;