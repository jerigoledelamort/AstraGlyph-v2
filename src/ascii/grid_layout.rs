// Dynamic cell grid layout — decides which glyph cells get merged into larger
// ones (ROADMAP Phase 3.1: "Dynamic cell grid (variable cell sizes — higher
// detail near camera)").
//
// Design note on what "higher detail near camera" can actually mean here:
// the scene is rendered to a low-res target with exactly one pixel per base
// cell, so there is no extra detail available to *add* close to the camera.
// The achievable — and equivalent — effect is the inverse: keep full 1:1
// detail on near geometry and MERGE distant, depth-flat regions into bigger
// glyphs. Detail then concentrates where the camera is looking, and the glyph
// instance count drops. `InstanceData` already carries per-instance width and
// height, so a non-uniform layout needs no shader change.
//
// Merging is mipmap-style: spans are powers of two and each candidate block is
// aligned to its own size. That keeps the tiling predictable, makes the
// "tiles exactly cover the grid" invariant easy to hold, and avoids the seams a
// greedy unaligned merge produces.

/// One laid-out cell: a `span` x `span` block of base cells drawn as a single
/// glyph. `span == 1` is an ordinary cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// Column of the block's top-left base cell.
    pub col: u32,
    /// Row of the block's top-left base cell.
    pub row: u32,
    /// Block edge length in base cells (a power of two, >= 1).
    pub span: u32,
}

impl Tile {
    /// Number of base cells this tile covers.
    pub fn area(&self) -> u32 {
        self.span * self.span
    }
}

/// When to merge base cells into a larger tile.
#[derive(Clone, Copy, Debug)]
pub struct SubdivisionPolicy {
    /// Window-space depth (0..1) beyond which a region is considered "far
    /// enough" to lose detail. Cells nearer than this are never merged.
    pub merge_depth: f32,
    /// How flat a block must be to merge: the max-min depth spread allowed
    /// inside it. Keeps silhouettes and depth discontinuities sharp — merging
    /// across an edge is what makes this kind of scheme look broken.
    pub depth_tolerance: f32,
    /// Largest allowed span. Clamped to a power of two <= 8; 1 disables merging.
    pub max_span: u32,
}

impl Default for SubdivisionPolicy {
    /// Conservative defaults: only clearly distant and clearly flat regions
    /// merge, and never coarser than 2x2, so the effect is visible without
    /// obviously degrading the image.
    fn default() -> Self {
        Self {
            merge_depth: 0.90,
            depth_tolerance: 0.01,
            max_span: 2,
        }
    }
}

impl SubdivisionPolicy {
    /// A policy that performs no merging at all — a plain uniform grid.
    pub fn uniform() -> Self {
        Self {
            merge_depth: 1.0,
            depth_tolerance: 0.0,
            max_span: 1,
        }
    }

    /// Whether this policy can ever merge anything.
    pub fn merges(&self) -> bool {
        self.effective_max_span() > 1
    }

    /// `max_span` rounded down to a power of two in 1..=8, treating
    /// zero/garbage as 1 (no merging) rather than panicking.
    fn effective_max_span(&self) -> u32 {
        match self.max_span {
            0 | 1 => 1,
            2 | 3 => 2,
            4..=7 => 4,
            _ => 8,
        }
    }
}

/// Compute the tile layout for one frame.
///
/// `depth` is window-space depth in 0..1, row-major, `cols * rows` entries
/// (1.0 = far plane / nothing drawn). A length mismatch is treated as "no depth
/// information available", which falls back to a uniform 1:1 grid rather than
/// producing a wrong layout.
///
/// The returned tiles always cover every base cell exactly once: the sum of
/// `area()` equals `cols * rows`, with no gaps and no overlaps.
pub fn compute_tiles(
    depth: &[f32],
    cols: u32,
    rows: u32,
    policy: &SubdivisionPolicy,
) -> Vec<Tile> {
    if cols == 0 || rows == 0 {
        return Vec::new();
    }

    let expected = (cols as usize) * (rows as usize);
    let have_depth = depth.len() == expected;
    let max_span = policy.effective_max_span();

    // Fast path: nothing to merge, or no usable depth.
    if max_span == 1 || !have_depth {
        return uniform_tiles(cols, rows);
    }

    let mut covered = vec![false; expected];
    let mut tiles = Vec::with_capacity(expected);

    // Coarse to fine, so the largest legal merge wins.
    let mut span = max_span;
    while span > 1 {
        let mut row = 0;
        while row + span <= rows {
            let mut col = 0;
            while col + span <= cols {
                if block_is_free(&covered, cols, col, row, span)
                    && block_is_mergeable(depth, cols, col, row, span, policy)
                {
                    mark_covered(&mut covered, cols, col, row, span);
                    tiles.push(Tile { col, row, span });
                }
                col += span;
            }
            row += span;
        }
        span /= 2;
    }

    // Everything the merge passes left behind stays a 1x1 cell. This also
    // covers the right/bottom remainder when cols/rows are not multiples of
    // the span, so partial blocks are never dropped.
    for row in 0..rows {
        for col in 0..cols {
            let idx = (row as usize) * (cols as usize) + (col as usize);
            if !covered[idx] {
                tiles.push(Tile { col, row, span: 1 });
            }
        }
    }

    tiles
}

/// Every base cell as its own tile.
fn uniform_tiles(cols: u32, rows: u32) -> Vec<Tile> {
    let mut tiles = Vec::with_capacity((cols as usize) * (rows as usize));
    for row in 0..rows {
        for col in 0..cols {
            tiles.push(Tile { col, row, span: 1 });
        }
    }
    tiles
}

fn block_is_free(covered: &[bool], cols: u32, col: u32, row: u32, span: u32) -> bool {
    for dy in 0..span {
        for dx in 0..span {
            let idx = ((row + dy) as usize) * (cols as usize) + ((col + dx) as usize);
            match covered.get(idx) {
                Some(true) => return false,
                Some(false) => {}
                None => return false,
            }
        }
    }
    true
}

fn mark_covered(covered: &mut [bool], cols: u32, col: u32, row: u32, span: u32) {
    for dy in 0..span {
        for dx in 0..span {
            let idx = ((row + dy) as usize) * (cols as usize) + ((col + dx) as usize);
            if let Some(slot) = covered.get_mut(idx) {
                *slot = true;
            }
        }
    }
}

/// A block merges when every cell in it is at least `merge_depth` away and the
/// depth spread across the block is within `depth_tolerance`.
///
/// Non-finite depth values (which would make min/max comparisons meaningless)
/// block the merge, keeping the region at full detail.
fn block_is_mergeable(
    depth: &[f32],
    cols: u32,
    col: u32,
    row: u32,
    span: u32,
    policy: &SubdivisionPolicy,
) -> bool {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for dy in 0..span {
        for dx in 0..span {
            let idx = ((row + dy) as usize) * (cols as usize) + ((col + dx) as usize);
            let Some(&d) = depth.get(idx) else {
                return false;
            };
            if !d.is_finite() || d < policy.merge_depth {
                return false;
            }
            if d < min {
                min = d;
            }
            if d > max {
                max = d;
            }
        }
    }

    max - min <= policy.depth_tolerance.max(0.0)
}

/// Average an RGBA8 block into a single linear-ish RGB triple, for the colour a
/// merged tile should be drawn with. Out-of-range coordinates are skipped, and
/// an entirely out-of-range block yields black rather than a division by zero.
pub fn average_block_color(
    pixels: &[[u8; 4]],
    cols: u32,
    rows: u32,
    tile: &Tile,
) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    let mut count = 0.0f32;

    for dy in 0..tile.span {
        let y = tile.row + dy;
        if y >= rows {
            break;
        }
        for dx in 0..tile.span {
            let x = tile.col + dx;
            if x >= cols {
                break;
            }
            let idx = (y as usize) * (cols as usize) + (x as usize);
            if let Some(p) = pixels.get(idx) {
                sum[0] += p[0] as f32;
                sum[1] += p[1] as f32;
                sum[2] += p[2] as f32;
                count += 1.0;
            }
        }
    }

    if count == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [
        sum[0] / count / 255.0,
        sum[1] / count / 255.0,
        sum[2] / count / 255.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The invariant the whole module rests on: tiles partition the grid.
    fn assert_exact_cover(tiles: &[Tile], cols: u32, rows: u32) {
        let mut hits = vec![0u32; (cols as usize) * (rows as usize)];
        for t in tiles {
            for dy in 0..t.span {
                for dx in 0..t.span {
                    let idx = ((t.row + dy) as usize) * (cols as usize) + ((t.col + dx) as usize);
                    hits[idx] += 1;
                }
            }
        }
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(*h, 1, "cell {i} covered {h} times, expected exactly once");
        }
        let area: u32 = tiles.iter().map(|t| t.area()).sum();
        assert_eq!(area, cols * rows, "total tile area must equal the grid");
    }

    #[test]
    fn uniform_policy_yields_one_tile_per_cell() {
        let depth = vec![1.0; 4 * 3];
        let tiles = compute_tiles(&depth, 4, 3, &SubdivisionPolicy::uniform());
        assert_eq!(tiles.len(), 12);
        assert!(tiles.iter().all(|t| t.span == 1));
        assert_exact_cover(&tiles, 4, 3);
    }

    #[test]
    fn flat_far_region_merges_to_max_span() {
        let policy = SubdivisionPolicy { merge_depth: 0.9, depth_tolerance: 0.01, max_span: 2 };
        let depth = vec![1.0; 4 * 4];
        let tiles = compute_tiles(&depth, 4, 4, &policy);
        // 4x4 of uniform far depth => four 2x2 tiles, no 1x1 leftovers.
        assert_eq!(tiles.len(), 4);
        assert!(tiles.iter().all(|t| t.span == 2));
        assert_exact_cover(&tiles, 4, 4);
    }

    #[test]
    fn near_region_keeps_full_detail() {
        let policy = SubdivisionPolicy { merge_depth: 0.9, depth_tolerance: 0.01, max_span: 2 };
        // Everything close to the camera.
        let depth = vec![0.2; 4 * 4];
        let tiles = compute_tiles(&depth, 4, 4, &policy);
        assert_eq!(tiles.len(), 16);
        assert!(tiles.iter().all(|t| t.span == 1));
        assert_exact_cover(&tiles, 4, 4);
    }

    #[test]
    fn depth_discontinuity_blocks_merging() {
        let policy = SubdivisionPolicy { merge_depth: 0.5, depth_tolerance: 0.01, max_span: 2 };
        // All cells are "far" but the block straddles a big depth step, so
        // merging across the edge must be refused.
        let depth = vec![
            0.60, 0.99, //
            0.60, 0.99, //
        ];
        let tiles = compute_tiles(&depth, 2, 2, &policy);
        assert_eq!(tiles.len(), 4, "an edge must not be merged away");
        assert_exact_cover(&tiles, 2, 2);
    }

    #[test]
    fn near_object_surrounded_by_far_background() {
        let policy = SubdivisionPolicy { merge_depth: 0.9, depth_tolerance: 0.01, max_span: 2 };
        let (cols, rows) = (4u32, 4u32);
        let mut depth = vec![1.0f32; (cols * rows) as usize];
        // A single near cell at (1,1) — its aligned 2x2 block must stay fine,
        // while the other three blocks merge.
        depth[(1 * cols + 1) as usize] = 0.3;
        let tiles = compute_tiles(&depth, cols, rows, &policy);
        assert_exact_cover(&tiles, cols, rows);

        let merged = tiles.iter().filter(|t| t.span == 2).count();
        let fine = tiles.iter().filter(|t| t.span == 1).count();
        assert_eq!(merged, 3, "the three untouched blocks merge");
        assert_eq!(fine, 4, "the block containing the near cell stays 1:1");
    }

    #[test]
    fn odd_dimensions_leave_remainder_as_single_cells() {
        let policy = SubdivisionPolicy { merge_depth: 0.9, depth_tolerance: 0.01, max_span: 2 };
        let (cols, rows) = (5u32, 3u32);
        let depth = vec![1.0; (cols * rows) as usize];
        let tiles = compute_tiles(&depth, cols, rows, &policy);
        // One 2x2 block fits in the 5x3 grid's first two rows/cols... two of
        // them horizontally; the rest is remainder.
        assert_exact_cover(&tiles, cols, rows);
        assert!(tiles.iter().any(|t| t.span == 2));
        assert!(tiles.iter().any(|t| t.span == 1));
    }

    #[test]
    fn coarse_span_beats_fine_when_allowed() {
        let policy = SubdivisionPolicy { merge_depth: 0.9, depth_tolerance: 0.01, max_span: 4 };
        let depth = vec![1.0; 4 * 4];
        let tiles = compute_tiles(&depth, 4, 4, &policy);
        assert_eq!(tiles.len(), 1, "a flat far 4x4 becomes one tile");
        assert_eq!(tiles[0].span, 4);
        assert_exact_cover(&tiles, 4, 4);
    }

    #[test]
    fn max_span_is_clamped_to_a_power_of_two() {
        assert_eq!(SubdivisionPolicy { max_span: 0, ..Default::default() }.effective_max_span(), 1);
        assert_eq!(SubdivisionPolicy { max_span: 3, ..Default::default() }.effective_max_span(), 2);
        assert_eq!(SubdivisionPolicy { max_span: 7, ..Default::default() }.effective_max_span(), 4);
        assert_eq!(SubdivisionPolicy { max_span: 99, ..Default::default() }.effective_max_span(), 8);
        assert!(!SubdivisionPolicy { max_span: 1, ..Default::default() }.merges());
        assert!(SubdivisionPolicy { max_span: 2, ..Default::default() }.merges());
    }

    #[test]
    fn wrong_depth_length_falls_back_to_uniform() {
        let policy = SubdivisionPolicy { merge_depth: 0.0, depth_tolerance: 1.0, max_span: 4 };
        // Depth buffer from a previous, differently-sized frame.
        let stale = vec![1.0; 3];
        let tiles = compute_tiles(&stale, 4, 4, &policy);
        assert_eq!(tiles.len(), 16, "a mismatched depth buffer must not merge blindly");
        assert_exact_cover(&tiles, 4, 4);
    }

    #[test]
    fn non_finite_depth_does_not_merge_or_panic() {
        let policy = SubdivisionPolicy { merge_depth: 0.5, depth_tolerance: 1.0, max_span: 2 };
        let depth = vec![f32::NAN, 1.0, 1.0, f32::INFINITY];
        let tiles = compute_tiles(&depth, 2, 2, &policy);
        assert_eq!(tiles.len(), 4);
        assert_exact_cover(&tiles, 2, 2);
    }

    #[test]
    fn negative_tolerance_is_treated_as_zero() {
        let policy = SubdivisionPolicy { merge_depth: 0.5, depth_tolerance: -5.0, max_span: 2 };
        // Perfectly flat, so a zero tolerance still permits the merge.
        let depth = vec![1.0; 4];
        let tiles = compute_tiles(&depth, 2, 2, &policy);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].span, 2);
    }

    #[test]
    fn empty_grid_is_handled() {
        assert!(compute_tiles(&[], 0, 0, &SubdivisionPolicy::default()).is_empty());
        assert!(compute_tiles(&[], 4, 0, &SubdivisionPolicy::default()).is_empty());
        assert!(compute_tiles(&[], 0, 4, &SubdivisionPolicy::default()).is_empty());
    }

    #[test]
    fn single_cell_grid_yields_one_tile() {
        let tiles = compute_tiles(&[1.0], 1, 1, &SubdivisionPolicy::default());
        assert_eq!(tiles, vec![Tile { col: 0, row: 0, span: 1 }]);
    }

    #[test]
    fn average_block_color_averages_the_span() {
        let pixels = vec![
            [0, 0, 0, 255], [255, 255, 255, 255], //
            [255, 255, 255, 255], [0, 0, 0, 255], //
        ];
        let tile = Tile { col: 0, row: 0, span: 2 };
        let avg = average_block_color(&pixels, 2, 2, &tile);
        for c in avg {
            assert!((c - 0.5).abs() < 1e-3, "expected mid grey, got {avg:?}");
        }
    }

    #[test]
    fn average_block_color_of_single_cell_is_that_cell() {
        let pixels = vec![[51, 102, 153, 255]];
        let avg = average_block_color(&pixels, 1, 1, &Tile { col: 0, row: 0, span: 1 });
        assert!((avg[0] - 0.2).abs() < 1e-2);
        assert!((avg[1] - 0.4).abs() < 1e-2);
        assert!((avg[2] - 0.6).abs() < 1e-2);
    }

    #[test]
    fn average_block_color_clips_at_the_edges() {
        // A span-2 tile at the bottom-right of a 3x3 grid only has one valid
        // cell; it must average that one rather than reading out of bounds.
        let mut pixels = vec![[0u8, 0, 0, 255]; 9];
        pixels[8] = [200, 200, 200, 255];
        let tile = Tile { col: 2, row: 2, span: 2 };
        let avg = average_block_color(&pixels, 3, 3, &tile);
        assert!((avg[0] - 200.0 / 255.0).abs() < 1e-3, "got {avg:?}");
    }

    #[test]
    fn average_block_color_out_of_range_is_black_not_a_panic() {
        let pixels = vec![[10u8, 20, 30, 255]];
        let avg = average_block_color(&pixels, 1, 1, &Tile { col: 9, row: 9, span: 2 });
        assert_eq!(avg, [0.0, 0.0, 0.0]);
        assert_eq!(average_block_color(&[], 4, 4, &Tile { col: 0, row: 0, span: 2 }), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn merging_reduces_instance_count_on_a_mostly_far_frame() {
        // The point of the feature: fewer glyphs for the same grid.
        let (cols, rows) = (16u32, 16u32);
        let mut depth = vec![1.0f32; (cols * rows) as usize];
        // A near patch in the middle keeps its detail.
        for row in 6..10 {
            for col in 6..10 {
                depth[(row * cols + col) as usize] = 0.25;
            }
        }
        let policy = SubdivisionPolicy { merge_depth: 0.9, depth_tolerance: 0.01, max_span: 4 };
        let tiles = compute_tiles(&depth, cols, rows, &policy);
        assert_exact_cover(&tiles, cols, rows);
        assert!(
            tiles.len() < (cols * rows) as usize,
            "merging must lower the glyph count: {} vs {}",
            tiles.len(),
            cols * rows
        );
        // The near patch must still be at full resolution.
        assert!(tiles.iter().filter(|t| t.span == 1).count() >= 16);
    }
}
