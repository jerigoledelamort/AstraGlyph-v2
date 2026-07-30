// Physics: rigid bodies, contact resolution, gameplay raycasting.
//
// A top-level module rather than a corner of `scene/`, because it owns state the
// scene does not: velocities, masses, and the contacts between frames. The
// geometry it collides against lives in `engine::geometry`, shared with the CPU
// tracer — one definition of where a body is, so it cannot collide somewhere
// other than where it is drawn.

pub mod body;
pub mod world;

// `Material` and `DEFAULT_GRAVITY` are reachable as `physics::body::*`; they are
// not re-exported here because nothing outside the module names them yet, and an
// unused re-export is a warning rather than a convenience.
pub use body::RigidBody;
pub use world::{ray_through_grid_cell, BodyId, PhysicsWorld};
