// 2D vector — self-implemented, no external math crates.

use std::fmt;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0 };
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
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
        Self {
            x: self.x / len,
            y: self.y / len,
        }
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }
}

impl Add for Vec2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec2_new_and_splat() {
        assert_eq!(Vec2::new(1.0, 2.0), Vec2 { x: 1.0, y: 2.0 });
        assert_eq!(Vec2::splat(5.0), Vec2 { x: 5.0, y: 5.0 });
    }

    #[test]
    fn vec2_add_sub() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a + b, Vec2::new(4.0, 6.0));
        assert_eq!(b - a, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn vec2_dot() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a.dot(b), 11.0);
    }

    #[test]
    fn vec2_length_and_normalize() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.length(), 5.0);
        assert_eq!(v.length_squared(), 25.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec2_lerp() {
        let a = Vec2::ZERO;
        let b = Vec2::new(10.0, 20.0);
        assert_eq!(a.lerp(b, 0.5), Vec2::new(5.0, 10.0));
    }

    #[test]
    fn vec2_scale_and_neg() {
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(v * 3.0, Vec2::new(3.0, 6.0));
        assert_eq!(-v, Vec2::new(-1.0, -2.0));
    }
}
