// Transform: position + rotation + scale, decomposed for easy manipulation.

use std::ops::Neg;
use crate::engine::math::{Mat4, Vec3};

/// Decomposed affine transform.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Vec3, // Euler angles in radians (pitch, yaw, roll)
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transform {
    pub const fn identity() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
        }
    }

    pub fn new(position: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self { position, rotation, scale }
    }

    /// Compose into a single 4x4 model matrix (T * R * S).
    pub fn to_matrix(self) -> Mat4 {
        let t = Mat4::translation_vec3(self.position);
        let rx = Mat4::rotation_x(self.rotation.x);
        let ry = Mat4::rotation_y(self.rotation.y);
        let rz = Mat4::rotation_z(self.rotation.z);
        let s = Mat4::scaling_vec3(self.scale);
        // Apply order: scale first, then rotate (Z * Y * X), then translate.
        t.mul(ry).mul(rx).mul(rz).mul(s)
    }

    /// Get the forward direction (-Z in local space).
    pub fn forward(self) -> Vec3 {
        let m = self.to_matrix();
        m.transform_dir(Vec3::UNIT_Z.neg())
    }

    /// Get the right direction (+X in local space).
    pub fn right(self) -> Vec3 {
        let m = self.to_matrix();
        m.transform_dir(Vec3::UNIT_X)
    }

    /// Get the up direction (+Y in local space).
    pub fn up(self) -> Vec3 {
        let m = self.to_matrix();
        m.transform_dir(Vec3::UNIT_Y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_identity_matrix() {
        let t = Transform::identity();
        assert_eq!(t.to_matrix(), Mat4::IDENTITY);
    }

    #[test]
    fn transform_translation_only() {
        let t = Transform {
            position: Vec3::new(5.0, 10.0, 15.0),
            ..Transform::identity()
        };
        let m = t.to_matrix();
        assert_eq!(m.transform_point(Vec3::ZERO), Vec3::new(5.0, 10.0, 15.0));
    }

    #[test]
    fn transform_scale_only() {
        let t = Transform {
            scale: Vec3::new(2.0, 3.0, 4.0),
            ..Transform::identity()
        };
        let m = t.to_matrix();
        assert_eq!(m.transform_point(Vec3::new(1.0, 1.0, 1.0)), Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn transform_forward_default() {
        let t = Transform::identity();
        let f = t.forward();
        assert!((f - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5);
    }
}
