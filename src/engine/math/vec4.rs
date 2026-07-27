// 4D vector — self-implemented, no external math crates.
// Used for homogeneous coordinates and RGBA colors.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };

    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v, w: v }
    }

    /// Create from a Vec3 + w component (homogeneous coordinate).
    pub const fn from_vec3(v: crate::engine::math::Vec3, w: f32) -> Self {
        Self { x: v.x, y: v.y, z: v.z, w }
    }

    /// Drop the w component and return the Vec3 part.
    pub fn xyz(self) -> crate::engine::math::Vec3 {
        crate::engine::math::Vec3::new(self.x, self.y, self.z)
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }
}

impl Add for Vec4 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl Sub for Vec4 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
            w: self.w * s,
        }
    }
}

impl Neg for Vec4 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::Vec3;

    #[test]
    fn vec4_new_and_splat() {
        assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0), Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 });
        assert_eq!(Vec4::splat(5.0), Vec4 { x: 5.0, y: 5.0, z: 5.0, w: 5.0 });
    }

    #[test]
    fn vec4_from_vec3() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let v4 = Vec4::from_vec3(v, 1.0);
        assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 1.0));
        assert_eq!(v4.xyz(), v);
    }

    #[test]
    fn vec4_add_sub_mul() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a + b, Vec4::new(6.0, 8.0, 10.0, 12.0));
        assert_eq!(b - a, Vec4::new(4.0, 4.0, 4.0, 4.0));
        assert_eq!(a * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn vec4_dot() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a.dot(b), 70.0);
    }

    #[test]
    fn vec4_lerp() {
        let a = Vec4::ZERO;
        let b = Vec4::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(a.lerp(b, 0.5), Vec4::new(5.0, 10.0, 15.0, 20.0));
    }
}
