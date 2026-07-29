// Analytic geometry: shapes and the ray intersections against them.
//
// Separate from `engine::math`, which is pure linear algebra with no opinion
// about what a shape is, and separate from `renderer` and `physics`, which are
// the two consumers. Both need the same answers — a reflection and a gameplay
// raycast that disagreed would show the player in a mirror somewhere other than
// where they can be hit.

pub mod ray;
pub mod shapes;

pub use ray::{Ray, RayHit};
pub use shapes::{Shape, WorldShape};
