use glam::{Mat4, Vec3};

/// A perspective projection.
#[derive(Copy, Clone, Debug)]
pub struct Perspective {
    /// Vertical field of view, in radians.
    pub fov_y_radians: f32,
    /// Width / height.
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Perspective {
    pub fn new(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            fov_y_radians,
            aspect,
            near,
            far,
        }
    }

    /// The view-to-clip matrix (wgpu depth range [0, 1]).
    pub fn matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_radians, self.aspect, self.near, self.far)
    }
}

/// A camera: where the viewer is and how the scene projects.
#[derive(Copy, Clone, Debug)]
pub struct Camera {
    /// World-to-view matrix.
    pub view: Mat4,
    /// View-to-clip matrix.
    pub projection: Mat4,
    /// Camera position in world space.
    pub position: Vec3,
}

impl Camera {
    /// A camera from a camera-to-world transform and a projection.
    pub fn from_camera_to_world(camera_to_world: Mat4, projection: Perspective) -> Self {
        Self {
            view: camera_to_world.inverse(),
            projection: projection.matrix(),
            position: camera_to_world.col(3).truncate(),
        }
    }

    /// A camera looking from `eye` toward `target`.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3, projection: Perspective) -> Self {
        Self {
            view: Mat4::look_at_rh(eye, target, up),
            projection: projection.matrix(),
            position: eye,
        }
    }
}
