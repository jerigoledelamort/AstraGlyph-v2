// Core engine utilities: error types, Result alias, async runner, Pod trait.

pub mod block_on;
pub mod error;
pub mod pod;

pub use block_on::block_on;
pub use error::{EngineError, Result};
pub use pod::{cast_slice, Pod};
