// Component trait and built-in component types.
// Components are plain data structs; systems interpret them.

use crate::engine::math::{Mat4, Transform, Vec3, Vec4};

/// Marker trait for data that can be used as a component.
pub trait Component: Send + 'static {}

// --- Built-in components ---

/// Local transform (position, rotation, scale).
#[derive(Clone, Debug)]
pub struct TransformComponent {
    pub local: Transform,
}

impl TransformComponent {
    pub fn new(position: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self {
            local: Transform::new(position, rotation, scale),
        }
    }

    pub fn world_matrix(&self) -> Mat4 {
        self.local.to_matrix()
    }
}

impl Component for TransformComponent {}

/// Vertex/index data for a mesh.
#[derive(Clone, Debug)]
pub struct MeshComponent {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl MeshComponent {
    pub fn new(vertices: Vec<MeshVertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }
}

impl Component for MeshComponent {}

/// A single vertex for 3D rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
}

unsafe impl crate::engine::core::Pod for MeshVertex {}

/// Material describing surface properties.
#[derive(Clone, Copy, Debug)]
pub struct MaterialComponent {
    /// Diffuse (albedo) color.
    pub color: Vec4,
    /// Ambient coefficient.
    pub ambient: f32,
    /// Diffuse coefficient.
    pub diffuse: f32,
}

impl MaterialComponent {
    pub const fn new(color: Vec4, ambient: f32, diffuse: f32) -> Self {
        Self { color, ambient, diffuse }
    }
}

impl Component for MaterialComponent {}

impl Default for MaterialComponent {
    fn default() -> Self {
        Self {
            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            ambient: 0.1,
            diffuse: 0.9,
        }
    }
}