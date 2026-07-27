// Dynamic cell grid — defines the ASCII character grid overlaying the scene.
//
// Design notes:
// - Cells can have different sizes (larger for background, smaller for details).
// - For MVP, we start with a uniform grid and support per-cell size overrides.
// - The grid is defined in "grid units" (cell coordinates), not pixels.
// - Each cell maps to a region of the render target texture.

use crate::engine::math::Vec2;

/// A single cell in the grid.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    /// Column (x) in grid units.
    pub col: u32,
    /// Row (y) in grid units.
    pub row: u32,
    /// Width of this cell in pixels (at the current render target resolution).
    pub width: u32,
    /// Height of this cell in pixels.
    pub height: u32,
    /// The computed brightness from the scene render pass.
    pub brightness: f32,
    /// The color sampled from the scene (RGB).
    pub color: [f32; 3],
    /// The ASCII glyph index to render.
    pub glyph_index: u32,
}

/// Configuration for a uniform cell grid.
#[derive(Clone, Copy, Debug)]
pub struct CellGridConfig {
    /// Number of columns (horizontal cells).
    pub cols: u32,
    /// Number of rows (vertical cells).
    pub rows: u32,
    /// Base cell size in pixels (will be computed to fill the viewport).
    pub cell_size: u32,
}

impl Default for CellGridConfig {
    fn default() -> Self {
        Self {
            cols: 80,
            rows: 48,
            cell_size: 16,
        }
    }
}

/// The cell grid managing the ASCII mesh.
#[derive(Clone, Debug)]
pub struct CellGrid {
    config: CellGridConfig,
    cells: Vec<Cell>,
}

impl CellGrid {
    /// Create a new uniform grid from the configuration.
    pub fn new(config: CellGridConfig) -> Self {
        let mut cells = Vec::with_capacity((config.cols * config.rows) as usize);
        for row in 0..config.rows {
            for col in 0..config.cols {
                cells.push(Cell {
                    col,
                    row,
                    width: config.cell_size,
                    height: config.cell_size,
                    brightness: 0.0,
                    color: [0.0, 0.0, 0.0],
                    glyph_index: 0,
                });
            }
        }
        Self { config, cells }
    }

    /// Total number of cells in the grid.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Immutable slice of all cells.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Mutable slice of all cells (for updating brightness/color/glyph).
    pub fn cells_mut(&mut self) -> &mut [Cell] {
        &mut self.cells
    }

    /// Get a mutable reference to a specific cell by grid coordinates.
    pub fn cell_mut(&mut self, col: u32, row: u32) -> Option<&mut Cell> {
        if col >= self.config.cols || row >= self.config.rows {
            return None;
        }
        let idx = (row * self.config.cols + col) as usize;
        self.cells.get_mut(idx)
    }

    pub fn cols(&self) -> u32 {
        self.config.cols
    }

    pub fn rows(&self) -> u32 {
        self.config.rows
    }

    pub fn config(&self) -> &CellGridConfig {
        &self.config
    }

    /// Get the pixel region covered by this grid (total width and height).
    pub fn pixel_size(&self) -> (u32, u32) {
        (
            self.config.cols * self.config.cell_size,
            self.config.rows * self.config.cell_size,
        )
    }

    /// Update all cell brightness/color from a raw RGBA pixel buffer.
    /// The buffer must be exactly `cols * rows` pixels (the low-res scene render).
    pub fn update_from_pixels(&mut self, pixels: &[[u8; 4]]) {
        assert_eq!(pixels.len(), self.cells.len());
        for (cell, pixel) in self.cells.iter_mut().zip(pixels.iter()) {
            // Compute perceived luminance (standard ITU-R BT.601).
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let luminance = 0.299 * r + 0.587 * g + 0.114 * b;

            cell.brightness = luminance;
            cell.color = [r, g, b];
            cell.glyph_index = crate::ascii::glyph_atlas::brightness_to_index(luminance);
        }
    }

    /// Get the screen-space position of a cell's top-left corner in normalized
    /// device coordinates (-1..1).
    pub fn cell_ndc(&self, col: u32, row: u32, screen_w: u32, screen_h: u32) -> Vec2 {
        let grid_w = self.pixel_size().0 as f32;
        let grid_h = self.pixel_size().1 as f32;
        let offset_x = ((screen_w as f32 - grid_w) / 2.0).max(0.0);
        let offset_y = ((screen_h as f32 - grid_h) / 2.0).max(0.0);

        let x = offset_x + col as f32 * self.config.cell_size as f32;
        let y = offset_y + (self.config.rows - 1 - row) as f32 * self.config.cell_size as f32;

        Vec2::new(
            (x / screen_w as f32) * 2.0 - 1.0,
            (y / screen_h as f32) * 2.0 - 1.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_creation() {
        let config = CellGridConfig { cols: 10, rows: 8, cell_size: 16 };
        let grid = CellGrid::new(config);
        assert_eq!(grid.cell_count(), 80);
        assert_eq!(grid.pixel_size(), (160, 128));
    }

    #[test]
    fn grid_update_from_pixels() {
        let config = CellGridConfig { cols: 4, rows: 4, cell_size: 8 };
        let mut grid = CellGrid::new(config);
        let pixels: Vec<[u8; 4]> = (0..16).map(|i| {
            let v = (i as f32 / 15.0 * 255.0) as u8;
            [v, v, v, 255]
        }).collect();
        grid.update_from_pixels(&pixels);
        // Brightest cell should have the highest glyph index.
        let cells = grid.cells();
        assert!(cells[0].glyph_index <= cells[15].glyph_index);
    }

    #[test]
    fn grid_cell_mut_out_of_bounds() {
        let mut grid = CellGrid::new(CellGridConfig::default());
        assert!(grid.cell_mut(100, 100).is_none());
        assert!(grid.cell_mut(0, 0).is_some());
    }
}