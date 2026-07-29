// Camera: projection + view matrix, with frustum parameters.

use crate::engine::math::{Mat4, Vec3};

/// Camera projection mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Perspective {
        fov_y: f32,  // radians
        aspect: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    },
}

impl Projection {
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self::Perspective { fov_y, aspect, near, far }
    }

    /// Orthographic projection from explicit frustum bounds.
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        Self::Orthographic { left, right, bottom, top, near, far }
    }

    /// Orthographic projection sized by vertical extent and aspect ratio —
    /// the ergonomic form, mirroring how `perspective` is usually called.
    ///
    /// `height` is the total world-space height covered by the viewport.
    pub fn orthographic_sized(height: f32, aspect: f32, near: f32, far: f32) -> Self {
        let half_h = height * 0.5;
        let half_w = half_h * aspect;
        Self::Orthographic {
            left: -half_w,
            right: half_w,
            bottom: -half_h,
            top: half_h,
            near,
            far,
        }
    }

    /// Whether this projection is orthographic.
    pub fn is_orthographic(&self) -> bool {
        matches!(self, Self::Orthographic { .. })
    }

    pub fn to_matrix(&self) -> Mat4 {
        match self {
            Self::Perspective { fov_y, aspect, near, far } => {
                Mat4::perspective(*fov_y, *aspect, *near, *far)
            }
            Self::Orthographic { left, right, bottom, top, near, far } => {
                Mat4::orthographic(*left, *right, *bottom, *top, *near, *far)
            }
        }
    }
}

/// A camera with position, orientation (via look-at), and projection.
#[derive(Clone, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub projection: Projection,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, up: Vec3, projection: Projection) -> Self {
        Self { position, target, up, projection }
    }

    /// View matrix (world → camera space).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(self.position, self.target, self.up)
    }

    /// Projection matrix (camera → clip space).
    pub fn projection_matrix(&self) -> Mat4 {
        self.projection.to_matrix()
    }

    /// Combined view-projection matrix.
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix().mul(self.view_matrix())
    }

    /// Forward direction (from position toward target).
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    /// Right direction (cross of forward and up).
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize()
    }

    /// Update aspect ratio on resize.
    ///
    /// Orthographic cameras keep their vertical extent and widen/narrow
    /// horizontally, so a window resize does not stretch the scene.
    pub fn set_aspect(&mut self, aspect: f32) {
        match self.projection {
            Projection::Perspective { fov_y, near, far, .. } => {
                self.projection = Projection::Perspective { fov_y, aspect, near, far };
            }
            Projection::Orthographic { bottom, top, near, far, .. } => {
                let half_h = (top - bottom) * 0.5;
                let center_y = (top + bottom) * 0.5;
                let half_w = half_h * aspect;
                self.projection = Projection::Orthographic {
                    left: -half_w,
                    right: half_w,
                    bottom: center_y - half_h,
                    top: center_y + half_h,
                    near,
                    far,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::radians;

    #[test]
    fn camera_view_projection() {
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 16.0 / 9.0, 0.1, 100.0),
        );
        let _vp = cam.view_projection();
        // The camera position should map to the origin in view space.
        let view = cam.view_matrix();
        let origin_in_view = view.transform_point(cam.position);
        assert!(origin_in_view.length() < 1e-5);
    }

    #[test]
    fn camera_forward_right() {
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 100.0),
        );
        let f = cam.forward();
        assert!((f - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5);
        let r = cam.right();
        assert!((r - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn projection_orthographic_sized_matches_explicit_bounds() {
        let sized = Projection::orthographic_sized(10.0, 2.0, 0.1, 100.0);
        let explicit = Projection::orthographic(-10.0, 10.0, -5.0, 5.0, 0.1, 100.0);
        assert_eq!(sized, explicit);
        assert!(sized.is_orthographic());
        assert!(!Projection::perspective(radians(60.0), 1.0, 0.1, 100.0).is_orthographic());
    }

    #[test]
    fn orthographic_camera_projects_without_perspective_divide() {
        // Two points at different depths but the same x should land on the same
        // NDC x under an orthographic projection (unlike perspective).
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::orthographic_sized(10.0, 1.0, 0.1, 100.0),
        );
        let vp = cam.view_projection();
        let near_pt = vp.transform_point(Vec3::new(2.0, 0.0, 0.0));
        let far_pt = vp.transform_point(Vec3::new(2.0, 0.0, -5.0));
        assert!((near_pt.x - far_pt.x).abs() < 1e-5);
    }

    #[test]
    fn set_aspect_on_orthographic_preserves_height() {
        let mut cam = Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::UNIT_Y,
            Projection::orthographic_sized(8.0, 1.0, 0.1, 100.0),
        );
        cam.set_aspect(2.0);
        match cam.projection {
            Projection::Orthographic { left, right, bottom, top, .. } => {
                assert!((top - bottom - 8.0).abs() < 1e-5, "height must be preserved");
                assert!((right - left - 16.0).abs() < 1e-5, "width must follow the aspect");
            }
            _ => panic!("expected orthographic"),
        }
    }

    #[test]
    fn camera_set_aspect() {
        let mut cam = Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::UNIT_Y,
            Projection::perspective(radians(90.0), 1.0, 0.1, 100.0),
        );
        cam.set_aspect(16.0 / 9.0);
        match cam.projection {
            Projection::Perspective { aspect, .. } => {
                assert!((aspect - 16.0 / 9.0).abs() < 1e-5);
            }
            _ => panic!("expected perspective"),
        }
    }
}