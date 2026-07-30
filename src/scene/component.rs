// Component trait and built-in component types.
// Components are plain data structs; systems interpret them.

use crate::engine::geometry::Shape;
use crate::engine::math::{Mat4, Transform, Vec2, Vec3, Vec4};

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

/// A single vertex for 3D rendering: position + normal + colour + uv,
/// 11 f32 / 44 bytes.
///
/// The vertex colour stays alongside the UV rather than being replaced by it:
/// untextured materials shade with the colour exactly as before, and textured
/// ones can still use it as a per-vertex tint. Growing this struct changes the
/// GPU vertex stride *and* the ray tracer's geometry heap stride — see
/// `renderer::scene_pass::VERTEX_ATTRS` and
/// `renderer::raytrace::HEAP_FLOATS_PER_VERTEX` (with its WGSL mirror
/// `HEAP_STRIDE` in `scene_traced_fragment.wgsl`); all of them are pinned to
/// this layout by tests.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
    /// Texture coordinates. (0,0) for meshes that predate texturing — with
    /// `texture_index == NO_TEXTURE` they never reach a sampler.
    pub uv: Vec2,
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

/// Bit flags inside a material's `flags` field. Mirrored in WGSL as
/// `MATERIAL_FLAG_*` constants in `scene_shading.wgsl`.
pub mod material_flags {
    /// Binary alpha cutout: discard fragments (or continue rays) where the
    /// sampled texture alpha is below 0.5. Distinct from glass transparency,
    /// which blends; a fence or a leaf is either there or it is not.
    pub const ALPHA_TEST: u32 = 1 << 0;
}

/// Material describing surface properties.
/// Mirrors the WGSL `Material` struct layout (64 bytes, std430-compatible).
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
    /// Texture layer in the scene's texture array, or `NO_TEXTURE`
    /// (`graphics::texture_array::NO_TEXTURE`) for an untextured material.
    /// The index doubles as an index into `LoadedScene::textures`, which is
    /// where the *path* lives — kept out of this struct so it stays `Copy`
    /// and a byte-for-byte mirror of the GPU layout.
    pub texture_index: u32,
    /// Bit flags, see `material_flags`.
    pub flags: u32,
    /// UV multiplier compensating for the texture's padding inside its array
    /// layer (1.0 when the texture matches the array size). Resolved when the
    /// texture array is built, not authored.
    pub uv_scale: [f32; 2],
}

/// Sentinel for "no texture", mirrored from `graphics::texture_array::NO_TEXTURE`.
/// Duplicated as a const here (with an equality test in `texture_array`) so this
/// GPU-layout module does not depend on the graphics module.
pub const NO_TEXTURE: u32 = 0xFFFF_FFFF;

impl MaterialComponent {
    pub const fn new(color: Vec4, ambient: f32, diffuse: f32) -> Self {
        Self::matte(color, ambient, diffuse)
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
            texture_index: NO_TEXTURE,
            flags: 0,
            uv_scale: [1.0, 1.0],
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
            texture_index: NO_TEXTURE,
            flags: 0,
            uv_scale: [1.0, 1.0],
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
            texture_index: NO_TEXTURE,
            flags: 0,
            uv_scale: [1.0, 1.0],
        }
    }

    /// This material with a texture attached. Albedo becomes a multiplier over
    /// the sampled texel (white = the texture as-is).
    pub const fn with_texture(mut self, texture_index: u32) -> Self {
        self.texture_index = texture_index;
        self
    }

    /// This material with binary alpha-test cutout enabled.
    pub const fn with_alpha_test(mut self) -> Self {
        self.flags |= material_flags::ALPHA_TEST;
        self
    }

    /// Whether this material carries a texture.
    pub const fn has_texture(&self) -> bool {
        self.texture_index != NO_TEXTURE
    }
}

impl Component for MaterialComponent {}

impl Default for MaterialComponent {
    fn default() -> Self {
        Self::matte(Vec4::new(1.0, 1.0, 1.0, 1.0), 0.1, 0.9)
    }
}

/// GPU-ready material uniform, matching the WGSL `Material` struct.
/// 64 bytes, std430-compatible.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
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
    /// Texture array layer, or `NO_TEXTURE`.
    pub texture_index: u32,
    /// Bit flags, see `material_flags`.
    pub flags: u32,
    /// UV multiplier for the texture's padded array layer.
    pub uv_scale: [f32; 2],
}

unsafe impl crate::engine::core::Pod for MaterialUniform {}

impl Default for MaterialUniform {
    /// The zero material *except* for `texture_index`: a zeroed index would be
    /// layer 0 — a real texture — so an accidentally-defaulted material would
    /// silently wear whatever texture happened to be loaded first.
    fn default() -> Self {
        Self::from(&MaterialComponent::default())
    }
}

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
            texture_index: m.texture_index,
            flags: m.flags,
            uv_scale: m.uv_scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_vertex_is_11_floats() {
        // Pinned because three separate consumers depend on it: the vertex
        // buffer layout, the geometry heap stride, and the WGSL HEAP_STRIDE.
        assert_eq!(std::mem::size_of::<MeshVertex>(), 44);
    }

    #[test]
    fn material_uniform_matches_the_wgsl_struct_size() {
        // vec4 (16) + 8 scalars (32) + u32 + u32 + vec2 (16) = 64, a multiple
        // of the 16-byte alignment the vec4 member forces in std430.
        assert_eq!(std::mem::size_of::<MaterialUniform>(), 64);
        assert_eq!(std::mem::size_of::<MaterialUniform>() % 16, 0);
    }

    #[test]
    fn default_material_carries_no_texture() {
        // A zeroed texture_index would be layer 0 — a real texture. The default
        // must be the sentinel, or every untextured mesh wears the first PNG.
        let u = MaterialUniform::default();
        assert_eq!(u.texture_index, NO_TEXTURE);
        assert_eq!(u.flags, 0);
        assert_eq!(u.uv_scale, [1.0, 1.0]);
    }

    #[test]
    fn texture_and_flag_builders_compose() {
        let m = MaterialComponent::matte(Vec4::new(1.0, 1.0, 1.0, 1.0), 0.1, 0.9)
            .with_texture(3)
            .with_alpha_test();
        assert!(m.has_texture());
        assert_eq!(m.texture_index, 3);
        assert!(m.flags & material_flags::ALPHA_TEST != 0);
        assert!(!MaterialComponent::default().has_texture());
    }
}