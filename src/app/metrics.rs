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
    fps_accum: f32,
    fps: f32,
    /// Last time we logged to stderr.
    last_log: Instant,
    /// Scene stats for the current frame: meshes drawn, meshes frustum-culled,
    /// distinct materials uploaded.
    drawn: usize,
    culled: usize,
    material_slots: usize,
}

impl FrameMetrics {
    pub fn new() -> Self {
        Self {
            cpu_start: None,
            cpu_time_us: 0,
            gpu_time_us: 0,
            frame_count: 0,
            fps_accum: 0.0,
            fps: 0.0,
            last_log: Instant::now(),
            drawn: 0,
            culled: 0,
            material_slots: 0,
        }
    }

    /// Record this frame's scene stats, shown on the periodic log line.
    pub fn set_scene_stats(&mut self, drawn: usize, culled: usize, material_slots: usize) {
        self.drawn = drawn;
        self.culled = culled;
        self.material_slots = material_slots;
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

        let dt = self.last_log.elapsed().as_secs_f32();
        self.frame_count += 1;
        self.fps_accum += dt;

        // Update FPS every 0.5s, log every 1.0s.
        if self.fps_accum >= 1.0 {
            self.fps = self.frame_count as f32 / self.fps_accum;
            eprintln!(
                "FPS: {:.0}, CPU: {:.1}ms, GPU: {:.1}ms | drawn: {}, culled: {}, materials: {}",
                self.fps,
                self.cpu_time_us as f32 / 1000.0,
                self.gpu_time_us as f32 / 1000.0,
                self.drawn,
                self.culled,
                self.material_slots,
            );
            self.frame_count = 0;
            self.fps_accum = 0.0;
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
}
