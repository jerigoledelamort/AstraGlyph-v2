// Core Entity: a lightweight ID-based handle.
// Components are stored externally (ECS-like pattern), but Entity itself is just an ID.

use std::num::NonZeroU64;

/// A unique entity identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Entity {
    id: NonZeroU64,
}

impl Entity {
    /// Create a new entity with the given raw ID.
    /// Panics if id is 0.
    pub const fn new(id: u64) -> Self {
        match NonZeroU64::new(id) {
            Some(n) => Self { id: n },
            None => panic!("Entity ID cannot be zero"),
        }
    }

    pub fn id(&self) -> u64 {
        self.id.get()
    }
}