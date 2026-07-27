// Scene module: entity, component, scene, camera.

pub mod camera;
pub mod component;
pub mod entity;
pub mod scene;

pub use camera::{Camera, Projection};
pub use component::{MaterialComponent, MeshComponent, MeshVertex, TransformComponent};
pub use entity::Entity;
pub use scene::Scene;