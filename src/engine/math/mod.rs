// Self-implemented math library: Vec2, Vec3, Vec4, Mat4, Transform.

pub mod mat4;
pub mod transform;
pub mod vec2;
pub mod vec3;
pub mod vec4;

pub use mat4::Mat4;
pub use transform::Transform;
pub use vec2::Vec2;
pub use vec3::Vec3;
pub use vec4::Vec4;

/// Convert degrees to radians.
pub fn radians(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

/// Convert radians to degrees.
pub fn degrees(radians: f32) -> f32 {
    radians * 180.0 / std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radians_degrees_conversion() {
        assert!((radians(0.0) - 0.0).abs() < 1e-6);
        assert!((radians(180.0) - std::f32::consts::PI).abs() < 1e-6);
        assert!((degrees(std::f32::consts::PI) - 180.0).abs() < 1e-6);
        assert!((degrees(0.0) - 0.0).abs() < 1e-6);
    }
}
