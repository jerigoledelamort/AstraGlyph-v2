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
    submit_time_us: u64,
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
            submit_time_us: 0,
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
        self.submit_time_us = 0;
    }

    /// Record time spent handing work to the GPU.
    ///
    /// This is wall-clock around `submit`, which is **not** GPU time: `submit`
    /// returns once the commands are queued, so it measures CPU-side submission
    /// cost and reads the same whether the GPU took a microsecond or ten
    /// milliseconds. It was labelled "GPU" through Phase 1-5 and that was
    /// misleading; the honest per-pass figures come from
    /// `graphics::timing::GpuTimer` and its timestamp queries.
    ///
    /// Kept because it is still a real number worth seeing — submission cost shows
    /// up here and nowhere else — but named for what it measures.
    pub fn record_submit_phase(&mut self, phase_start: Instant) {
        let elapsed = phase_start.elapsed().as_micros() as u64;
        self.submit_time_us += elapsed;
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
                "FPS: {:.0}, CPU: {:.1}ms, submit: {:.1}ms | drawn: {}, culled: {}, materials: {}, glyphs: {}",
                self.fps,
                self.cpu_time_us as f32 / 1000.0,
                self.submit_time_us as f32 / 1000.0,
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

    /// CPU time for the last completed frame, in milliseconds.
    pub fn cpu_ms(&self) -> f32 {
        self.cpu_time_us as f32 / 1000.0
    }

    /// Time spent submitting to the GPU, in milliseconds. See
    /// `record_submit_phase` for why this is not GPU time.
    pub fn submit_ms(&self) -> f32 {
        self.submit_time_us as f32 / 1000.0
    }

    /// Wall-clock frame time implied by the FPS estimate, in milliseconds.
    ///
    /// Derived from the FPS average rather than measured per frame: the per-frame
    /// figure at 1300 FPS is 0.77 ms with enormous variance, and a profiler row that
    /// flickers is unreadable. Zero FPS (before the first second elapses) reports 0
    /// rather than dividing by it.
    pub fn frame_ms(&self) -> f32 {
        if self.fps > 0.0 {
            1000.0 / self.fps
        } else {
            0.0
        }
    }

    /// Glyph quads drawn last frame.
    pub fn glyph_count(&self) -> usize {
        self.glyphs
    }

    /// Meshes drawn and culled last frame.
    pub fn scene_counts(&self) -> (usize, usize, usize) {
        (self.drawn, self.culled, self.material_slots)
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

    /// Renamed from `metrics_record_gpu_phase` along with the method: what this
    /// measures is submission cost, not GPU work.
    #[test]
    fn metrics_record_submit_phase() {
        let mut m = FrameMetrics::new();
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(50));
        m.record_submit_phase(start);
        assert!(m.submit_time_us > 0);
    }

    /// The profiler reads these; a missing accessor is a compile error but a *wrong*
    /// one is not, so the derivations are pinned.
    #[test]
    fn frame_ms_is_the_reciprocal_of_fps_and_safe_at_zero() {
        let mut metrics = FrameMetrics::new();
        assert_eq!(
            metrics.frame_ms(),
            0.0,
            "before the first second, dividing by a zero FPS must not produce infinity"
        );
        metrics.fps = 200.0;
        assert!((metrics.frame_ms() - 5.0).abs() < 1e-4, "{}", metrics.frame_ms());
    }

    #[test]
    fn the_submit_accessor_reports_what_was_recorded() {
        let mut metrics = FrameMetrics::new();
        metrics.begin_frame();
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        metrics.record_submit_phase(start);
        assert!(
            metrics.submit_ms() >= 1.5,
            "recorded {} ms for a 2 ms sleep",
            metrics.submit_ms()
        );
    }

    #[test]
    fn counters_round_trip_through_their_accessors() {
        let mut metrics = FrameMetrics::new();
        metrics.set_scene_stats(7, 3, 5);
        metrics.set_glyph_count(1234);
        assert_eq!(metrics.scene_counts(), (7, 3, 5));
        assert_eq!(metrics.glyph_count(), 1234);
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
