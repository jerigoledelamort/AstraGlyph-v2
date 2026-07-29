// Core engine utilities: error types, Result alias, async runner, Pod trait,
// JSON parsing.

pub mod block_on;
pub mod error;
pub mod json;
pub mod pod;

pub use block_on::block_on;
pub use error::{EngineError, Result};
pub use pod::{cast_slice, Pod};
// Only the value type is re-exported: `parse` stays behind `json::` so the name
// keeps its meaning at call sites.
#[allow(unused_imports)]
pub use json::JsonValue;
