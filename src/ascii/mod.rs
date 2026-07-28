// ASCII module: glyph atlas, cell grid, and converter.

pub mod cell_grid;
pub mod glyph_atlas;

pub use glyph_atlas::{build_atlas, glyph_count, GLYPH_SIZE};