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
    /// Inverse of an affine transform (rotation, scale, translation).
    ///
    /// Not a general 4x4 inverse: it assumes the bottom row is `(0, 0, 0, 1)`,
    /// which every transform this engine builds satisfies and no projection matrix
    /// does. That restriction is what makes it a closed form — invert the upper
    /// 3x3, then apply it to the negated translation — rather than a cofactor
    /// expansion, and it means a projection matrix passed here silently gets
    /// nonsense. `None` is returned when the 3x3 part is singular (a zero scale
    /// on some axis), because there is genuinely no inverse then and returning
    /// identity would quietly move whatever was being un-transformed.
    pub fn inverse_affine(self) -> Option<Self> {
        // Element (row, col) is at m[col * 4 + row].
        let m = |r: usize, c: usize| self.m[c * 4 + r];
        // Cofactors of the upper 3x3.
        let c00 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
        let c01 = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
        let c02 = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
        let det = m(0, 0) * c00 + m(0, 1) * c01 + m(0, 2) * c02;
        if det.abs() < 1e-12 || !det.is_finite() {
            return None;
        }
        let inv_det = 1.0 / det;

        // Adjugate, transposed into the inverse.
        let i00 = c00 * inv_det;
        let i01 = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * inv_det;
        let i02 = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * inv_det;
        let i10 = c01 * inv_det;
        let i11 = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * inv_det;
        let i12 = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * inv_det;
        let i20 = c02 * inv_det;
        let i21 = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * inv_det;
        let i22 = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * inv_det;

        // The inverse translation is the inverted rotation applied to -t.
        let (tx, ty, tz) = (m(0, 3), m(1, 3), m(2, 3));
        let t0 = -(i00 * tx + i01 * ty + i02 * tz);
        let t1 = -(i10 * tx + i11 * ty + i12 * tz);
        let t2 = -(i20 * tx + i21 * ty + i22 * tz);

        Some(Self {
            m: [
                i00, i10, i20, 0.0, //
                i01, i11, i21, 0.0, //
                i02, i12, i22, 0.0, //
                t0, t1, t2, 1.0,
            ],
        })
    }

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

    /// The property that matters: composing a matrix with its inverse must be
    /// the identity, checked by round-tripping points rather than by comparing
    /// floats element by element.
    #[test]
    fn inverse_affine_round_trips_points() {
        let cases = [
            Mat4::translation(3.0, -4.0, 5.0),
            Mat4::scaling(2.0, 3.0, 0.5),
            Mat4::rotation_y(0.7),
            Mat4::translation(1.0, 2.0, 3.0)
                .mul(Mat4::rotation_y(0.9))
                .mul(Mat4::scaling(2.0, 2.0, 2.0)),
            Mat4::translation(-2.0, 0.5, 1.5)
                .mul(Mat4::rotation_x(0.3))
                .mul(Mat4::rotation_z(-1.1))
                .mul(Mat4::scaling(1.5, 0.5, 3.0)),
        ];
        let points = [
            Vec3::ZERO,
            Vec3::UNIT_X,
            Vec3::UNIT_Y,
            Vec3::UNIT_Z,
            Vec3::new(-3.5, 7.25, 0.125),
        ];
        for m in cases {
            let inv = m.inverse_affine().expect("these are all invertible");
            for p in points {
                let round_tripped = inv.transform_point(m.transform_point(p));
                assert!(
                    (round_tripped - p).length() < 1e-4,
                    "point {p} came back as {round_tripped}"
                );
            }
        }
    }

    /// A singular matrix has no inverse. Returning identity instead would quietly
    /// leave whatever was being un-transformed in the wrong place.
    #[test]
    fn inverse_affine_refuses_a_singular_matrix() {
        assert!(Mat4::scaling(1.0, 0.0, 1.0).inverse_affine().is_none());
        assert!(Mat4::zero().inverse_affine().is_none());
    }

    #[test]
    fn inverse_of_the_identity_is_the_identity() {
        let inv = Mat4::IDENTITY.inverse_affine().unwrap();
        assert_eq!(inv, Mat4::IDENTITY);
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
