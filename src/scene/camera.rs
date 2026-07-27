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
    pub fn set_aspect(&mut self, aspect: f32) {
        if let Projection::Perspective { fov_y, near, far, .. } = self.projection {
            self.projection = Projection::Perspective { fov_y, aspect, near, far };
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
        let vp = cam.view_projection();
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