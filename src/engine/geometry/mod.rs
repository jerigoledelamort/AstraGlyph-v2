// Analytic geometry: shapes and the ray intersections against them.
//
// Separate from `engine::math`, which is pure linear algebra with no opinion
// about what a shape is, and separate from `renderer` and `physics`, which are
// the two consumers. Both need the same answers — a reflection and a gameplay
// raycast that disagreed would show the player in a mirror somewhere other than
// where they can be hit.

pub mod collision;
pub mod ray;
pub mod shapes;

// `Contact` and `ContactPair` are used through `collision::` by the physics
// world; re-exporting them here as well would be an unused import.
pub use ray::{Ray, RayHit};
pub use shapes::{Basis, Shape, WorldShape};
