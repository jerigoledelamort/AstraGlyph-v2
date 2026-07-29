// Frame performance metrics: FPS, CPU frame time, GPU frame time estimate.
// Outputs to stderr once per second.

use std::time::Instant;

/// Tracks per-frame timing metrics and logs them periodically.
pub struct FrameMetrics {
    /// Start of the current frame (begin of render()).
    cpu_start: Option<Instant>,
    /// Accumulated CPU time across all phases of the frame.
    cpu_time_us: u64,
    /// Estimated GPU time (measured around submit calls).
    gpu_time_us: u64,
    /// FPS tracking.
    frame_count: u64,
    fps: f32,
    /// Last time we logged to stderr.
    last_log: Instant,
    /// Scene stats for the current frame: meshes drawn, meshes frustum-culled,
    /// distinct materials uploaded.
    drawn: usize,
    culled: usize,
    material_slots: usize,
    /// Glyph quads submitted this frame. Below cols*rows when the dynamic cell
    /// grid merged distant cells.
    glyphs: usize,
}

impl FrameMetrics {
    pub fn new() -> Self {
        Self {
            cpu_start: None,
            cpu_time_us: 0,
            gpu_time_us: 0,
            frame_count: 0,
            fps: 0.0,
            last_log: Instant::now(),
            drawn: 0,
            culled: 0,
            material_slots: 0,
            glyphs: 0,
        }
    }

    /// Record this frame's scene stats, shown on the periodic log line.
    pub fn set_scene_stats(&mut self, drawn: usize, culled: usize, material_slots: usize) {
        self.drawn = drawn;
        self.culled = culled;
        self.material_slots = material_slots;
    }

    /// Record how many glyph quads were submitted this frame.
    pub fn set_glyph_count(&mut self, glyphs: usize) {
        self.glyphs = glyphs;
    }

    /// Call at the beginning of render().
    pub fn begin_frame(&mut self) {
        self.cpu_start = Some(Instant::now());
        self.cpu_time_us = 0;
        self.gpu_time_us = 0;
    }

    /// Record a GPU submit phase. Measures wall-clock time around the submit call
    /// as an approximation of GPU work (not precise, but sufficient for Phase 1).
    pub fn record_gpu_phase(&mut self, phase_start: Instant) {
        let elapsed = phase_start.elapsed().as_micros() as u64;
        self.gpu_time_us += elapsed;
    }

    /// Call at the end of render(). Finalizes CPU time and logs if 1s has passed.
    pub fn end_frame(&mut self) {
        if let Some(start) = self.cpu_start {
            self.cpu_time_us = start.elapsed().as_micros() as u64;
        }

        self.frame_count += 1;

        // Frames divided by the REAL wall time since the last log.
        //
        // This used to accumulate `last_log.elapsed()` once per frame — the time
        // since the last LOG, not since the previous frame — so the accumulator
        // grew quadratically and crossed 1.0 after only sqrt(2/frame_time) frames.
        // The reported number therefore collapsed to ~18 for any frame time near
        // 5.5ms and barely moved whatever the engine did, which looked exactly
        // like a hard 18 FPS cap. There was no cap; the counter was wrong.
        let elapsed = self.last_log.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed;
            eprintln!(
                "FPS: {:.0}, CPU: {:.1}ms, GPU: {:.1}ms | drawn: {}, culled: {}, materials: {}, glyphs: {}",
                self.fps,
                self.cpu_time_us as f32 / 1000.0,
                self.gpu_time_us as f32 / 1000.0,
                self.drawn,
                self.culled,
                self.material_slots,
                self.glyphs,
            );
            self.frame_count = 0;
            self.last_log = Instant::now();
        }
    }

    /// Current FPS estimate (updated every ~1s).
    pub fn fps(&self) -> f32 {
        self.fps
    }
}

impl Default for FrameMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_begin_end_frame() {
        let mut m = FrameMetrics::new();
        m.begin_frame();
        // Simulate some work.
        std::thread::sleep(std::time::Duration::from_micros(100));
        m.end_frame();
        // CPU time should be > 0.
        assert!(m.cpu_time_us > 0);
    }

    #[test]
    fn metrics_record_gpu_phase() {
        let mut m = FrameMetrics::new();
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(50));
        m.record_gpu_phase(start);
        assert!(m.gpu_time_us > 0);
    }

    #[test]
    fn metrics_fps_initial_zero() {
        let m = FrameMetrics::new();
        assert_eq!(m.fps(), 0.0);
    }

    /// The regression this guards: FPS must be frames divided by real elapsed
    /// time. The old code accumulated "time since last log" once per frame, so
    /// the reported rate collapsed to a constant ~18 regardless of the actual
    /// frame time — indistinguishable from a hard frame cap.
    #[test]
    fn fps_reflects_the_real_frame_rate() {
        let mut m = FrameMetrics::new();
        // ~2ms per frame => ~500 FPS. Run for just over a second of wall time so
        // exactly one log window closes.
        let mut frames = 0u32;
        let start = Instant::now();
        while start.elapsed().as_secs_f32() < 1.05 {
            m.begin_frame();
            std::thread::sleep(std::time::Duration::from_millis(2));
            m.end_frame();
            frames += 1;
        }

        let reported = m.fps();
        let actual = frames as f32 / start.elapsed().as_secs_f32();
        assert!(reported > 0.0, "FPS never got computed");
        // Generous bound: sleep granularity varies a lot across machines, but the
        // reported value must track the real rate rather than sitting on a
        // frame-time-independent constant.
        assert!(
            (reported - actual).abs() < actual * 0.5,
            "reported {reported:.0} FPS but ran at {actual:.0} FPS"
        );
        assert!(
            reported > 30.0,
            "reported {reported:.0} FPS for ~2ms frames — the old quadratic \
             accumulator bug would land near 18"
        );
    }
}
