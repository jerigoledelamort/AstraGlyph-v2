// 4x4 column-major matrix — self-implemented, no external math crates.
// Stored as 16 f32 values in column-major order (compatible with WGSL).

use crate::engine::math::{Vec3, Vec4};

/// Column-major 4x4 matrix.
///
/// Memory layout (column-major, same as WGSL `mat4x4<f32>`):
/// ```text
/// m[0]  m[4]  m[8]   m[12]
/// m[1]  m[5]  m[9]   m[13]
/// m[2]  m[6]  m[10]  m[14]
/// m[3]  m[7]  m[11]  m[15]
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    pub m: [f32; 16],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        m: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    };

    pub const fn zero() -> Self {
        Self { m: [0.0; 16] }
    }

    /// Create from column-major array.
    pub const fn from_cols_array(m: [f32; 16]) -> Self {
        Self { m }
    }

    /// Create from four column vectors.
    pub fn from_cols(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Self {
        Self {
            m: [
                c0.x, c0.y, c0.z, c0.w,
                c1.x, c1.y, c1.z, c1.w,
                c2.x, c2.y, c2.z, c2.w,
                c3.x, c3.y, c3.z, c3.w,
            ],
        }
    }

    /// Matrix multiplication: `self * other`.
    pub fn mul(self, other: Self) -> Self {
        let mut result = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    // self is column-major: element (row, k) = m[k*4 + row]
                    // other is column-major: element (k, col) = m[col*4 + k]
                    sum += self.m[k * 4 + row] * other.m[col * 4 + k];
                }
                result[col * 4 + row] = sum;
            }
        }
        Self { m: result }
    }

    /// Transform a Vec4 (matrix * vector).
    pub fn transform_vec4(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.m[0] * v.x + self.m[4] * v.y + self.m[8] * v.z + self.m[12] * v.w,
            self.m[1] * v.x + self.m[5] * v.y + self.m[9] * v.z + self.m[13] * v.w,
            self.m[2] * v.x + self.m[6] * v.y + self.m[10] * v.z + self.m[14] * v.w,
            self.m[3] * v.x + self.m[7] * v.y + self.m[11] * v.z + self.m[15] * v.w,
        )
    }

    /// Transform a Vec3 as a point (w=1), return the xyz part.
    pub fn transform_point(self, v: Vec3) -> Vec3 {
        let r = self.transform_vec4(Vec4::from_vec3(v, 1.0));
        if r.w.abs() > 1e-10 {
            r.xyz() * (1.0 / r.w)
        } else {
            r.xyz()
        }
    }

    /// Transform a Vec3 as a direction (w=0).
    pub fn transform_dir(self, v: Vec3) -> Vec3 {
        self.transform_vec4(Vec4::from_vec3(v, 0.0)).xyz()
    }

    /// Transpose the matrix.
    pub fn transpose(self) -> Self {
        let m = self.m;
        Self {
            m: [
                m[0], m[4], m[8], m[12],
                m[1], m[5], m[9], m[13],
                m[2], m[6], m[10], m[14],
                m[3], m[7], m[11], m[15],
            ],
        }
    }

    /// Create a translation matrix.
    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        Self {
            m: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                x, y, z, 1.0,
            ],
        }
    }

    pub fn translation_vec3(v: Vec3) -> Self {
        Self::translation(v.x, v.y, v.z)
    }

    /// Create a scaling matrix.
    pub fn scaling(x: f32, y: f32, z: f32) -> Self {
        Self {
            m: [
                x, 0.0, 0.0, 0.0,
                0.0, y, 0.0, 0.0,
                0.0, 0.0, z, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    pub fn scaling_vec3(v: Vec3) -> Self {
        Self::scaling(v.x, v.y, v.z)
    }

    pub fn scaling_uniform(s: f32) -> Self {
        Self::scaling(s, s, s)
    }

    /// Create a rotation matrix around the X axis.
    pub fn rotation_x(radians: f32) -> Self {
        let (s, c) = radians.sin_cos();
        Self {
            m: [
                1.0, 0.0, 0.0, 0.0,
                0.0, c, s, 0.0,
                0.0, -s, c, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a rotation matrix around the Y axis.
    pub fn rotation_y(radians: f32) -> Self {
        let (s, c) = radians.sin_cos();
        Self {
            m: [
                c, 0.0, -s, 0.0,
                0.0, 1.0, 0.0, 0.0,
                s, 0.0, c, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a rotation matrix around the Z axis.
    pub fn rotation_z(radians: f32) -> Self {
        let (s, c) = radians.sin_cos();
        Self {
            m: [
                c, s, 0.0, 0.0,
                -s, c, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Create a perspective projection matrix (OpenGL convention).
    ///
    /// - `fov_y`: vertical field of view in radians
    /// - `aspect`: width / height
    /// - `near`, `far`: clipping planes (positive)
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y / 2.0).tan();
        Self {
            m: [
                f / aspect, 0.0, 0.0, 0.0,
                0.0, f, 0.0, 0.0,
                0.0, 0.0, (far + near) / (near - far), -1.0,
                0.0, 0.0, (2.0 * far * near) / (near - far), 0.0,
            ],
        }
    }

    /// Create an orthographic projection matrix.
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        Self {
            m: [
                2.0 / (right - left), 0.0, 0.0, 0.0,
                0.0, 2.0 / (top - bottom), 0.0, 0.0,
                0.0, 0.0, -2.0 / (far - near), 0.0,
                -(right + left) / (right - left),
                -(top + bottom) / (top - bottom),
                -(far + near) / (far - near),
                1.0,
            ],
        }
    }

    /// Create a look-at view matrix (right-handed).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalize(); // forward
        let s = f.cross(up).normalize();    // right
        let u = s.cross(f);                 // up (already normalized)

        Self {
            m: [
                s.x, u.x, -f.x, 0.0,
                s.y, u.y, -f.y, 0.0,
                s.z, u.z, -f.z, 0.0,
                -s.dot(eye), -u.dot(eye), f.dot(eye), 1.0,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat4_identity() {
        let m = Mat4::IDENTITY;
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        assert_eq!(m.transform_vec4(v), v);
    }

    #[test]
    fn mat4_mul_identity() {
        let a = Mat4::translation(1.0, 2.0, 3.0);
        assert_eq!(a.mul(Mat4::IDENTITY), a);
        assert_eq!(Mat4::IDENTITY.mul(a), a);
    }

    #[test]
    fn mat4_translation() {
        let m = Mat4::translation(10.0, 20.0, 30.0);
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(m.transform_point(p), Vec3::new(11.0, 22.0, 33.0));
    }

    #[test]
    fn mat4_scaling() {
        let m = Mat4::scaling(2.0, 3.0, 4.0);
        let p = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(m.transform_point(p), Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn mat4_rotation_x() {
        let m = Mat4::rotation_x(std::f32::consts::FRAC_PI_2);
        let p = Vec3::new(0.0, 1.0, 0.0);
        let r = m.transform_dir(p);
        assert!((r - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-5);
    }

    #[test]
    fn mat4_rotation_y() {
        let m = Mat4::rotation_y(std::f32::consts::FRAC_PI_2);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let r = m.transform_dir(p);
        assert!((r - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5);
    }

    #[test]
    fn mat4_transpose() {
        let m = Mat4::from_cols_array([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);
        let t = m.transpose();
        assert_eq!(t.m[1], 5.0); // row 1, col 0 of original
        assert_eq!(t.m[4], 2.0); // row 0, col 1 of original
    }

    #[test]
    fn mat4_look_at() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let target = Vec3::ZERO;
        let up = Vec3::UNIT_Y;
        let view = Mat4::look_at(eye, target, up);
        // The eye position transformed by the view matrix should be at origin.
        let eye_trans = view.transform_point(eye);
        assert!(eye_trans.length() < 1e-5);
    }

    #[test]
    fn mat4_perspective() {
        let p = Mat4::perspective(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0);
        // A point at the origin should map to z = far in NDC (negative).
        let v = p.transform_vec4(Vec4::new(0.0, 0.0, -0.1, 1.0));
        assert!(v.z.abs() < 1.0); // near plane → z_ndc ≈ -1
    }

    #[test]
    fn mat4_mul_composition() {
        let t = Mat4::translation(1.0, 0.0, 0.0);
        let s = Mat4::scaling(2.0, 2.0, 2.0);
        let combined = t.mul(s);
        let p = Vec3::new(1.0, 1.0, 1.0);
        // First scale by 2, then translate by 1 in x.
        assert_eq!(combined.transform_point(p), Vec3::new(3.0, 2.0, 2.0));
    }
}
