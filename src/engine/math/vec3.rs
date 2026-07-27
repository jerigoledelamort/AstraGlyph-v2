// 3D vector — self-implemented, no external math crates.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const UNIT_Z: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::ZERO;
        }
        self * (1.0 / len)
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    /// Reflect `self` around the given surface normal.
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, s: f32) -> Self {
        Self {
            x: self.x / s,
            y: self.y / s,
            z: self.z / s,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_new_and_splat() {
        assert_eq!(Vec3::new(1.0, 2.0, 3.0), Vec3 { x: 1.0, y: 2.0, z: 3.0 });
        assert_eq!(Vec3::splat(5.0), Vec3 { x: 5.0, y: 5.0, z: 5.0 });
    }

    #[test]
    fn vec3_add_sub() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b - a, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0);
    }

    #[test]
    fn vec3_cross() {
        let a = Vec3::UNIT_X;
        let b = Vec3::UNIT_Y;
        assert_eq!(a.cross(b), Vec3::UNIT_Z);
        assert_eq!(b.cross(a), -Vec3::UNIT_Z);
    }

    #[test]
    fn vec3_length_and_normalize() {
        let v = Vec3::new(0.0, 3.0, 4.0);
        assert_eq!(v.length(), 5.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_lerp() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 20.0, 30.0);
        assert_eq!(a.lerp(b, 0.5), Vec3::new(5.0, 10.0, 15.0));
    }

    #[test]
    fn vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::UNIT_Y;
        assert_eq!(v.reflect(n), Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn vec3_scale_neg_div() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(-v, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(v / 2.0, Vec3::new(0.5, 1.0, 1.5));
    }
}
