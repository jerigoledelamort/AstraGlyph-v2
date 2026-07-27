// ASCII module: glyph atlas, cell grid, and converter.

pub mod cell_grid;
pub mod glyph_atlas;

pub use cell_grid::{Cell, CellGrid, CellGridConfig};
pub use glyph_atlas::{build_atlas, brightness_to_index, glyph_count, ALL_CHARS, BRIGHTNESS_RAMP, GLYPH_SIZE};