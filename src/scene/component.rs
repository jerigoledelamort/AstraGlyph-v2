// Component trait and built-in component types.
// Components are plain data structs; systems interpret them.

use crate::engine::geometry::Shape;
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

/// The analytic shape an entity's mesh approximates, in the entity's local space.
///
/// Attached alongside `MeshComponent` by every scene source (the loader and the
/// code-built demos), because two subsystems need the equation rather than the
/// triangles:
///
/// - The CPU fallback tracer solves one quadratic per sphere instead of walking
///   1536 triangles per ray, which is the difference between an interactive
///   fallback and a slideshow.
/// - Physics needs a collision volume, and mesh-mesh collision is a different
///   project entirely.
///
/// One component serves both so a collider can never disagree with what the
/// tracer reflects.
#[derive(Clone, Copy, Debug)]
pub struct ColliderComponent {
    pub shape: Shape,
}

impl ColliderComponent {
    pub const fn new(shape: Shape) -> Self {
        Self { shape }
    }
}

impl Component for ColliderComponent {}

/// A single vertex for 3D rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
}

unsafe impl crate::engine::core::Pod for MeshVertex {}

/// Material type determining the shading model.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialType {
    /// Matte / diffuse surface (Lambertian).
    Matte = 0,
    /// Mirror / reflective surface.
    Mirror = 1,
    /// Glass / transparent refractive surface.
    Glass = 2,
}

impl Default for MaterialType {
    fn default() -> Self {
        Self::Matte
    }
}

/// Material describing surface properties.
/// Mirrors the WGSL `Material` struct layout (48 bytes, std430-compatible).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MaterialComponent {
    /// Diffuse (albedo) color. The .w component is padding for alignment.
    pub color: Vec4,
    /// Material type: 0=Matte, 1=Mirror, 2=Glass.
    pub material_type: u32,
    /// Ambient coefficient.
    pub ambient: f32,
    /// Diffuse coefficient.
    pub diffuse: f32,
    /// Specular coefficient.
    pub specular: f32,
    /// Shininess exponent (Phong specular power).
    pub shininess: f32,
    /// Index of refraction (for glass, e.g. 1.5 for typical glass).
    pub ior: f32,
    /// Reflectivity at normal incidence (0..1).
    pub reflectivity: f32,
    /// Transparency (0=opaque, 1=fully transparent).
    pub transparency: f32,
}

impl MaterialComponent {
    pub const fn new(color: Vec4, ambient: f32, diffuse: f32) -> Self {
        Self {
            color,
            material_type: MaterialType::Matte as u32,
            ambient,
            diffuse,
            specular: 0.0,
            shininess: 1.0,
            ior: 1.0,
            reflectivity: 0.0,
            transparency: 0.0,
        }
    }

    /// Create a matte material with the given albedo and diffuse coefficient.
    pub const fn matte(color: Vec4, ambient: f32, diffuse: f32) -> Self {
        Self {
            color,
            material_type: MaterialType::Matte as u32,
            ambient,
            diffuse,
            specular: 0.0,
            shininess: 1.0,
            ior: 1.0,
            reflectivity: 0.0,
            transparency: 0.0,
        }
    }

    /// Create a mirror material with high specular and reflectivity.
    pub const fn mirror(color: Vec4, reflectivity: f32) -> Self {
        Self {
            color,
            material_type: MaterialType::Mirror as u32,
            ambient: 0.05,
            diffuse: 0.1,
            specular: 0.8,
            shininess: 128.0,
            ior: 1.0,
            reflectivity,
            transparency: 0.0,
        }
    }

    /// Create a glass material with refraction and transparency.
    pub const fn glass(color: Vec4, ior: f32, transparency: f32) -> Self {
        Self {
            color,
            material_type: MaterialType::Glass as u32,
            ambient: 0.05,
            diffuse: 0.1,
            specular: 0.5,
            shininess: 64.0,
            ior,
            reflectivity: 0.1,
            transparency,
        }
    }
}

impl Component for MaterialComponent {}

impl Default for MaterialComponent {
    fn default() -> Self {
        Self::matte(Vec4::new(1.0, 1.0, 1.0, 1.0), 0.1, 0.9)
    }
}

/// GPU-ready material uniform, matching the WGSL `Material` struct.
/// 48 bytes, std430-compatible.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialUniform {
    /// Albedo color (xyz) + padding (w).
    pub albedo: [f32; 4],
    /// Material type: 0=Matte, 1=Mirror, 2=Glass.
    pub material_type: u32,
    /// Ambient coefficient.
    pub ambient: f32,
    /// Diffuse coefficient.
    pub diffuse: f32,
    /// Specular coefficient.
    pub specular: f32,
    /// Shininess exponent.
    pub shininess: f32,
    /// Index of refraction.
    pub ior: f32,
    /// Reflectivity at normal incidence.
    pub reflectivity: f32,
    /// Transparency.
    pub transparency: f32,
}

unsafe impl crate::engine::core::Pod for MaterialUniform {}

impl From<&MaterialComponent> for MaterialUniform {
    fn from(m: &MaterialComponent) -> Self {
        Self {
            albedo: [m.color.x, m.color.y, m.color.z, m.color.w],
            material_type: m.material_type,
            ambient: m.ambient,
            diffuse: m.diffuse,
            specular: m.specular,
            shininess: m.shininess,
            ior: m.ior,
            reflectivity: m.reflectivity,
            transparency: m.transparency,
        }
    }
}