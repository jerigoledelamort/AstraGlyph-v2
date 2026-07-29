// ASCII module: glyph atlas, cell grid, dynamic grid layout, converter.

pub mod cell_grid;
pub mod color;
pub mod glyph_atlas;
pub mod grid_layout;
pub mod overlay;

pub use glyph_atlas::{build_atlas, glyph_count, GLYPH_SIZE};
#[allow(unused_imports)]
pub use grid_layout::{average_block_color, compute_tiles, SubdivisionPolicy, Tile};
#[allow(unused_imports)]
pub use overlay::{Overlay, OverlayCell, SceneCell};
#[allow(unused_imports)]
pub use color::{luminance, quantize_buffer, ColorMode};