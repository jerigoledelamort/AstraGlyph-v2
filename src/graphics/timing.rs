// GPU timestamp queries: what each render pass actually cost on the GPU.
//
// `app::metrics` has always measured GPU time as wall-clock around `submit`. That
// number is not GPU time at all — `submit` returns as soon as the commands are
// queued, so it measures how long it took the CPU to hand work over. It reads as
// "GPU: 0.1ms" whether the GPU took a microsecond or ten milliseconds. Phase 6.3
// asks for honest per-pass timings, and this is the only way to get them: the GPU
// writes its own clock into a query set at the start and end of each pass.
//
// Three things make this awkward, and all three are handled here rather than at the
// call sites:
//
// 1. `TIMESTAMP_QUERY` is optional. On an adapter without it the whole mechanism is
//    absent and the profiler has to say so, not report zeros.
// 2. Results are not available when the pass ends. The GPU writes them when it gets
//    there, so they must be resolved into a buffer, copied back, and read a frame or
//    more later. Reading them synchronously would stall the CPU on the GPU — which
//    is exactly what Phase 1.2 removed from the readback path.
// 3. Timestamps are in ticks, not nanoseconds, and the tick period is per-adapter.

use std::collections::HashMap;

/// Passes that can be timed, in the order they run.
///
/// A fixed enum rather than arbitrary string labels: the query set has to be sized
/// at creation, and a fixed set means the indices are compile-time constants
/// instead of a runtime map lookup per pass per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GpuPass {
    /// Depth-only pass from the shadow-casting light. Absent while tracing.
    Shadow,
    /// The 3D scene into the offscreen target — rasterised or traced.
    Scene,
    /// Glyph quads to the swapchain.
    Composite,
}

impl GpuPass {
    /// Every pass, in execution order.
    pub const ALL: [GpuPass; 3] = [GpuPass::Shadow, GpuPass::Scene, GpuPass::Composite];

    /// Short label for the HUD, where there is one cell per character.
    pub fn label(self) -> &'static str {
        match self {
            Self::Shadow => "SHADOW",
            Self::Scene => "SCENE",
            Self::Composite => "COMPOSITE",
        }
    }

    /// Index of this pass's *pair* of timestamps in the query set.
    fn slot(self) -> u32 {
        match self {
            Self::Shadow => 0,
            Self::Scene => 1,
            Self::Composite => 2,
        }
    }

    /// Query index for the beginning-of-pass timestamp.
    fn begin_index(self) -> u32 {
        self.slot() * 2
    }

    /// Query index for the end-of-pass timestamp.
    fn end_index(self) -> u32 {
        self.slot() * 2 + 1
    }
}

/// Two timestamps per pass.
const QUERIES_PER_PASS: u32 = 2;

/// Total query slots.
const QUERY_COUNT: u32 = GpuPass::ALL.len() as u32 * QUERIES_PER_PASS;

/// Bytes per resolved timestamp (`u64`).
const TIMESTAMP_SIZE: u64 = 8;

/// How many readback slots are kept.
///
/// Four rather than the obvious two. A map callback fires when the driver gets to
/// it, not on a schedule, and at 1300 FPS several frames can be submitted in the
/// meantime — measured on this machine, a two-slot rotation ran out and dropped
/// most frames' timings. Four costs 4 x 48 bytes of buffer and keeps the sample
/// rate high; `skipped()` reports it if even that is not enough.
const FRAMES_IN_FLIGHT: usize = 4;

/// Smoothing applied to each pass's reported time.
///
/// GPU timings are noisy frame to frame (clock boost, other processes, driver
/// scheduling), and an unsmoothed number in a HUD is unreadable — it flickers
/// through three digits. This is a per-frame exponential factor: low enough to
/// settle quickly, high enough to be steady.
const SMOOTHING: f64 = 0.1;

/// What a readback slot is currently doing.
///
/// Three states, not a boolean. A slot is not simply "in use or not": between the
/// submit and the map callback the buffer belongs to wgpu, and writing to it — or
/// even submitting a copy into it — is a validation error. The first version of this
/// used one `in_use` flag cleared on unmap, and it crashed with
/// "Buffer with 'gpu_timestamp_readback' label is still mapped" on `Queue::submit`:
/// the slot rotation came back around before the callback had fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    /// Nothing outstanding; safe to resolve into.
    Free,
    /// A copy into this buffer has been *recorded* but not yet submitted. The map
    /// must not be requested yet — see `after_submit`.
    Recorded,
    /// The copy is submitted and a map is outstanding. The buffer belongs to wgpu
    /// until the callback fires, so this slot must not be touched.
    AwaitingMap,
    /// The map completed and the data has not been read yet.
    Ready,
}

/// One frame's in-flight readback.
struct PendingFrame {
    /// GPU-visible buffer the query set resolves into.
    resolve: wgpu::Buffer,
    /// Mappable buffer the resolve is copied to.
    readback: wgpu::Buffer,
    /// Set by the map callback, which runs on whichever thread polls the device.
    mapped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    state: SlotState,
    /// Which passes were actually recorded, so a pass that did not run this frame
    /// is reported as absent rather than as zero.
    recorded: Vec<GpuPass>,
}

/// GPU timing, or a clear statement that it is unavailable.
pub struct GpuTimer {
    query_set: Option<wgpu::QuerySet>,
    frames: Vec<PendingFrame>,
    /// Frames whose timings were dropped because every slot was still outstanding.
    ///
    /// Exposed because it is the honest measure of how much the profiler is
    /// actually sampling: a high count means the smoothed figures come from a
    /// fraction of frames, which is fine for a profiler but worth knowing.
    skipped: u64,
    /// Nanoseconds per timestamp tick, from the queue.
    period_ns: f32,
    /// Smoothed milliseconds per pass.
    smoothed: HashMap<GpuPass, f64>,
    /// Passes recorded this frame, reset by `begin_frame`.
    recording: Vec<GpuPass>,
    /// Successful resolves, so "no numbers yet" is distinguishable from "broken".
    samples: u64,
}

impl GpuTimer {
    /// Create a timer, or a disabled one if the device lacks `TIMESTAMP_QUERY`.
    ///
    /// Never fails: a machine without the feature must still run, and a profiler
    /// that refused to construct would take the renderer with it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let supported = device
            .features()
            .contains(wgpu::Features::from(wgpu::FeaturesWebGPU::TIMESTAMP_QUERY));
        if !supported {
            return Self::disabled();
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_pass_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });

        let size = QUERY_COUNT as u64 * TIMESTAMP_SIZE;
        let frames = (0..FRAMES_IN_FLIGHT)
            .map(|_| PendingFrame {
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpu_timestamp_resolve"),
                    size,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpu_timestamp_readback"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                state: SlotState::Free,
                recorded: Vec::new(),
            })
            .collect();

        Self {
            query_set: Some(query_set),
            frames,
            skipped: 0,
            period_ns: queue.get_timestamp_period(),
            smoothed: HashMap::new(),
            recording: Vec::new(),
            samples: 0,
        }
    }

    /// A timer that measures nothing, for adapters without the feature.
    pub fn disabled() -> Self {
        Self {
            query_set: None,
            frames: Vec::new(),
            skipped: 0,
            period_ns: 1.0,
            smoothed: HashMap::new(),
            recording: Vec::new(),
            samples: 0,
        }
    }

    /// Whether timing is available on this device.
    pub fn is_available(&self) -> bool {
        self.query_set.is_some()
    }

    /// Resolved frames since startup.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Frames whose timings were dropped because no readback slot was free.
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// Start recording a frame. Clears the per-frame pass list.
    pub fn begin_frame(&mut self) {
        self.recording.clear();
    }

    /// Timestamp writes to attach to a render pass descriptor.
    ///
    /// Returns `None` when timing is unavailable, which is exactly the value
    /// `RenderPassDescriptor::timestamp_writes` wants — so a call site reads the
    /// same whether or not the feature exists.
    pub fn pass_writes(&mut self, pass: GpuPass) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let query_set = self.query_set.as_ref()?;
        if !self.recording.contains(&pass) {
            self.recording.push(pass);
        }
        Some(wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(pass.begin_index()),
            end_of_pass_write_index: Some(pass.end_index()),
        })
    }

    /// Resolve this frame's queries into a readback buffer.
    ///
    /// Must be called on an encoder submitted *after* every timed pass, and its
    /// commands must be submitted for the results ever to arrive. Silently does
    /// nothing when no slot is free, which is the correct answer: skipping a frame's
    /// timings costs a sample, while reusing a mapped buffer is a hard validation
    /// error.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(query_set) = self.query_set.as_ref() else {
            return;
        };
        if self.recording.is_empty() {
            return;
        }
        // Search for a free slot rather than trusting a rotating index. The rotation
        // assumed a slot would always be finished by the time it came round again,
        // and at 1300 FPS against a map callback that lands whenever the driver gets
        // to it, that is simply untrue.
        let Some(slot) = self
            .frames
            .iter()
            .position(|f| f.state == SlotState::Free)
        else {
            self.skipped += 1;
            return;
        };
        let frame = &mut self.frames[slot];
        encoder.resolve_query_set(query_set, 0..QUERY_COUNT, &frame.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &frame.resolve,
            0,
            &frame.readback,
            0,
            QUERY_COUNT as u64 * TIMESTAMP_SIZE,
        );
        // Recorded, not yet mapped. `map_async` must wait for the submit — see
        // `after_submit`.
        frame.state = SlotState::Recorded;
        frame.recorded = self.recording.clone();
    }

    /// Request the maps for everything `resolve` recorded.
    ///
    /// Must be called *after* the encoder holding those copies has been submitted.
    /// This split is not tidiness, it is a requirement: a buffer with an outstanding
    /// `map_async` may not be used by a submitted command, so calling `map_async`
    /// inside `resolve` — before the submit — fails validation with
    /// "Buffer with 'gpu_timestamp_readback' label is still mapped" on the very
    /// submit that carries the copy. The buffer has to be handed to the GPU first
    /// and asked for back second.
    pub fn after_submit(&mut self) {
        for frame in &mut self.frames {
            if frame.state != SlotState::Recorded {
                continue;
            }
            frame.state = SlotState::AwaitingMap;
            let flag = frame.mapped.clone();
            flag.store(false, std::sync::atomic::Ordering::Release);
            let signal = flag;
            frame.readback.clone().slice(..).map_async(
                wgpu::MapMode::Read,
                move |result| {
                    if result.is_ok() {
                        signal.store(true, std::sync::atomic::Ordering::Release);
                    }
                },
            );
        }
    }

    /// Read back any completed frame and fold it into the smoothed averages.
    ///
    /// Non-blocking: a frame whose map has not landed yet is left for next time.
    /// Polling here rather than waiting is the whole reason this does not stall.
    pub fn collect(&mut self, device: &wgpu::Device) {
        if self.query_set.is_none() {
            return;
        }
        // Drives the map callbacks without blocking, the same `Poll` the ASCII
        // readback uses. `Wait` would stall the CPU on the GPU and change the very
        // number being measured.
        let _ = device.poll(wgpu::PollType::Poll);

        for index in 0..self.frames.len() {
            if self.frames[index].state != SlotState::AwaitingMap {
                continue;
            }
            if !self.frames[index]
                .mapped
                .load(std::sync::atomic::Ordering::Acquire)
            {
                continue;
            }
            self.frames[index].state = SlotState::Ready;
            let ticks = {
                let frame = &self.frames[index];
                let Ok(view) = frame.readback.slice(..).get_mapped_range() else {
                    continue;
                };
                let mut ticks = [0u64; QUERY_COUNT as usize];
                for (i, slot) in ticks.iter_mut().enumerate() {
                    let start = i * TIMESTAMP_SIZE as usize;
                    let bytes: [u8; 8] = view[start..start + 8]
                        .try_into()
                        .unwrap_or([0; 8]);
                    *slot = u64::from_le_bytes(bytes);
                }
                ticks
            };
            let recorded = self.frames[index].recorded.clone();
            // Unmap before marking free, in that order: the slot is only safe to
            // resolve into once wgpu has the buffer back.
            self.frames[index].readback.unmap();
            self.frames[index]
                .mapped
                .store(false, std::sync::atomic::Ordering::Release);
            self.frames[index].state = SlotState::Free;
            self.fold(&ticks, &recorded);
        }
    }

    /// Turn one frame's ticks into smoothed milliseconds.
    fn fold(&mut self, ticks: &[u64], recorded: &[GpuPass]) {
        let mut any = false;
        for pass in recorded {
            let begin = ticks.get(pass.begin_index() as usize).copied().unwrap_or(0);
            let end = ticks.get(pass.end_index() as usize).copied().unwrap_or(0);
            // A pass whose timestamps did not land reads as end <= begin. Skipping
            // it keeps the last good value rather than reporting a spurious zero,
            // which in a profiler looks like the pass became free.
            if end <= begin {
                continue;
            }
            let ms = (end - begin) as f64 * self.period_ns as f64 / 1.0e6;
            // A wildly implausible figure means the timestamps came from different
            // frames or the clock wrapped. Discarded rather than folded in, because
            // one bad sample would poison the average for hundreds of frames.
            if !ms.is_finite() || ms > 1000.0 {
                continue;
            }
            let entry = self.smoothed.entry(*pass).or_insert(ms);
            *entry += (ms - *entry) * SMOOTHING;
            any = true;
        }
        if any {
            self.samples += 1;
        }
    }

    /// Smoothed time for one pass, in milliseconds. `None` if never measured.
    pub fn pass_ms(&self, pass: GpuPass) -> Option<f64> {
        self.smoothed.get(&pass).copied()
    }

    /// Total measured GPU time across all passes, in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.smoothed.values().sum()
    }

    /// Per-pass breakdown in execution order, skipping passes never measured.
    pub fn breakdown(&self) -> Vec<(GpuPass, f64)> {
        GpuPass::ALL
            .iter()
            .filter_map(|pass| self.pass_ms(*pass).map(|ms| (*pass, ms)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Query indices must be distinct across every pass. An overlap would make two
    /// passes overwrite each other's timestamps, and the result would be a plausible
    /// number that is simply wrong — the worst kind of profiler output.
    #[test]
    fn every_pass_has_its_own_pair_of_query_slots() {
        let mut seen = Vec::new();
        for pass in GpuPass::ALL {
            seen.push(pass.begin_index());
            seen.push(pass.end_index());
        }
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            seen.len(),
            "query indices collide: {seen:?}"
        );
        assert!(
            seen.iter().all(|i| *i < QUERY_COUNT),
            "an index exceeds the query set size {QUERY_COUNT}: {seen:?}"
        );
    }

    #[test]
    fn the_query_set_is_sized_for_every_pass() {
        assert_eq!(QUERY_COUNT, GpuPass::ALL.len() as u32 * 2);
        assert!(QUERY_COUNT > 0, "a query set of zero is a validation error");
    }

    #[test]
    fn a_begin_index_always_precedes_its_end() {
        for pass in GpuPass::ALL {
            assert!(
                pass.begin_index() < pass.end_index(),
                "{:?} has its end before its begin",
                pass
            );
        }
    }

    /// A device without the feature must produce a timer that measures nothing and
    /// says so — not one that reports zeros, which would read as "the GPU is
    /// infinitely fast".
    #[test]
    fn a_disabled_timer_reports_no_measurements() {
        let timer = GpuTimer::disabled();
        assert!(!timer.is_available());
        assert!(timer.breakdown().is_empty());
        assert_eq!(timer.total_ms(), 0.0);
        assert_eq!(timer.samples(), 0);
        for pass in GpuPass::ALL {
            assert_eq!(timer.pass_ms(pass), None);
        }
    }

    /// A disabled timer must tolerate the whole per-frame sequence, so the call
    /// sites need no branch.
    #[test]
    fn a_disabled_timer_survives_the_frame_sequence() {
        let mut timer = GpuTimer::disabled();
        timer.begin_frame();
        for pass in GpuPass::ALL {
            assert!(
                timer.pass_writes(pass).is_none(),
                "a disabled timer must return None, which is what the descriptor wants"
            );
        }
        assert_eq!(timer.samples(), 0);
    }

    /// The fold is the arithmetic that turns ticks into milliseconds, and it is
    /// testable without a GPU. 1000 ticks at 1 ns each is one microsecond.
    #[test]
    fn ticks_become_milliseconds_using_the_adapter_period() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        let mut ticks = [0u64; QUERY_COUNT as usize];
        // Scene pass: 2,500,000 ticks = 2.5 ms at 1 ns per tick.
        ticks[GpuPass::Scene.begin_index() as usize] = 1_000_000;
        ticks[GpuPass::Scene.end_index() as usize] = 3_500_000;
        timer.fold(&ticks, &[GpuPass::Scene]);
        let ms = timer.pass_ms(GpuPass::Scene).expect("should have a value");
        assert!((ms - 2.5).abs() < 1e-9, "got {ms} ms");
    }

    /// A different tick period must scale the result. Hard-coding nanoseconds is the
    /// obvious mistake, and it would be silently wrong by whatever factor the
    /// adapter's clock differs by.
    #[test]
    fn the_adapter_period_scales_the_result() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 38.4; // a real figure from some AMD parts
        let mut ticks = [0u64; QUERY_COUNT as usize];
        ticks[GpuPass::Scene.begin_index() as usize] = 0;
        ticks[GpuPass::Scene.end_index() as usize] = 100_000;
        timer.fold(&ticks, &[GpuPass::Scene]);
        let ms = timer.pass_ms(GpuPass::Scene).unwrap();
        assert!((ms - 3.84).abs() < 1e-6, "got {ms} ms");
    }

    /// The first sample establishes the value rather than being smoothed toward
    /// zero from nothing — otherwise the HUD would take a hundred frames to show
    /// the right number.
    #[test]
    fn the_first_sample_is_taken_as_is() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        let mut ticks = [0u64; QUERY_COUNT as usize];
        ticks[GpuPass::Composite.begin_index() as usize] = 0;
        ticks[GpuPass::Composite.end_index() as usize] = 1_000_000;
        timer.fold(&ticks, &[GpuPass::Composite]);
        assert!((timer.pass_ms(GpuPass::Composite).unwrap() - 1.0).abs() < 1e-9);
    }

    /// Later samples are smoothed, so a HUD figure is readable rather than
    /// flickering through three digits.
    #[test]
    fn later_samples_are_smoothed_toward_the_new_value() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        let sample = |timer: &mut GpuTimer, ns: u64| {
            let mut ticks = [0u64; QUERY_COUNT as usize];
            ticks[GpuPass::Scene.begin_index() as usize] = 0;
            ticks[GpuPass::Scene.end_index() as usize] = ns;
            timer.fold(&ticks, &[GpuPass::Scene]);
        };
        sample(&mut timer, 1_000_000); // 1.0 ms
        sample(&mut timer, 2_000_000); // 2.0 ms
        let after_one = timer.pass_ms(GpuPass::Scene).unwrap();
        assert!(
            after_one > 1.0 && after_one < 2.0,
            "one smoothed step should land between the two: {after_one}"
        );
        // And it converges rather than oscillating.
        for _ in 0..200 {
            sample(&mut timer, 2_000_000);
        }
        let converged = timer.pass_ms(GpuPass::Scene).unwrap();
        assert!(
            (converged - 2.0).abs() < 0.01,
            "should have converged on 2.0 ms, got {converged}"
        );
    }

    /// A pass whose timestamps did not land must keep its last good value, not
    /// report zero — a zero in a profiler reads as "this pass became free".
    #[test]
    fn a_missing_timestamp_keeps_the_previous_value() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        let mut good = [0u64; QUERY_COUNT as usize];
        good[GpuPass::Scene.begin_index() as usize] = 0;
        good[GpuPass::Scene.end_index() as usize] = 1_000_000;
        timer.fold(&good, &[GpuPass::Scene]);
        let established = timer.pass_ms(GpuPass::Scene).unwrap();

        // All zeros: end == begin, so nothing usable.
        timer.fold(&[0u64; QUERY_COUNT as usize], &[GpuPass::Scene]);
        assert_eq!(
            timer.pass_ms(GpuPass::Scene),
            Some(established),
            "a dropped sample must not overwrite a good one"
        );
    }

    /// One absurd sample must not poison the average for hundreds of frames.
    #[test]
    fn an_implausible_sample_is_discarded() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        let mut good = [0u64; QUERY_COUNT as usize];
        good[GpuPass::Scene.begin_index() as usize] = 0;
        good[GpuPass::Scene.end_index() as usize] = 1_000_000;
        timer.fold(&good, &[GpuPass::Scene]);

        // Ten seconds' worth of ticks: the timestamps came from different frames, or
        // the clock wrapped.
        let mut absurd = [0u64; QUERY_COUNT as usize];
        absurd[GpuPass::Scene.begin_index() as usize] = 0;
        absurd[GpuPass::Scene.end_index() as usize] = 10_000_000_000;
        timer.fold(&absurd, &[GpuPass::Scene]);
        assert!(
            (timer.pass_ms(GpuPass::Scene).unwrap() - 1.0).abs() < 0.01,
            "an absurd sample leaked into the average: {:?}",
            timer.pass_ms(GpuPass::Scene)
        );
    }

    /// A pass that did not run this frame must be absent from the breakdown, not
    /// present with a stale or zero figure. The shadow pass really does not run
    /// while tracing, so this distinction is load-bearing.
    #[test]
    fn only_recorded_passes_appear_in_the_breakdown() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        let mut ticks = [0u64; QUERY_COUNT as usize];
        for pass in GpuPass::ALL {
            ticks[pass.begin_index() as usize] = 0;
            ticks[pass.end_index() as usize] = 1_000_000;
        }
        // Only two of the three were recorded, even though all six slots hold data.
        timer.fold(&ticks, &[GpuPass::Scene, GpuPass::Composite]);
        let breakdown = timer.breakdown();
        assert_eq!(breakdown.len(), 2, "got {breakdown:?}");
        assert!(breakdown.iter().all(|(p, _)| *p != GpuPass::Shadow));
    }

    /// The breakdown is in execution order, so the HUD reads top to bottom the way
    /// the frame actually runs.
    #[test]
    fn the_breakdown_is_in_execution_order() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        let mut ticks = [0u64; QUERY_COUNT as usize];
        for pass in GpuPass::ALL {
            ticks[pass.begin_index() as usize] = 0;
            ticks[pass.end_index() as usize] = 1_000_000;
        }
        // Folded in reverse order on purpose.
        timer.fold(
            &ticks,
            &[GpuPass::Composite, GpuPass::Scene, GpuPass::Shadow],
        );
        let order: Vec<GpuPass> = timer.breakdown().into_iter().map(|(p, _)| p).collect();
        assert_eq!(
            order,
            vec![GpuPass::Shadow, GpuPass::Scene, GpuPass::Composite]
        );
    }

    #[test]
    fn the_total_is_the_sum_of_the_passes() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        let mut ticks = [0u64; QUERY_COUNT as usize];
        ticks[GpuPass::Scene.begin_index() as usize] = 0;
        ticks[GpuPass::Scene.end_index() as usize] = 2_000_000;
        ticks[GpuPass::Composite.begin_index() as usize] = 0;
        ticks[GpuPass::Composite.end_index() as usize] = 500_000;
        timer.fold(&ticks, &[GpuPass::Scene, GpuPass::Composite]);
        assert!((timer.total_ms() - 2.5).abs() < 1e-9, "{}", timer.total_ms());
    }

    #[test]
    fn every_pass_has_a_label() {
        for pass in GpuPass::ALL {
            assert!(!pass.label().is_empty());
        }
        // And they are distinct, or the HUD would show two identical rows.
        let mut labels: Vec<&str> = GpuPass::ALL.iter().map(|p| p.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), GpuPass::ALL.len());
    }

    /// The slot states must be distinct, and `Free` must be the only one a resolve
    /// will write into. A boolean flag here is what caused
    /// "Buffer ... is still mapped" on submit: the rotation came back to a slot
    /// whose map callback had not fired.
    #[test]
    fn a_slot_is_only_reusable_once_wgpu_has_the_buffer_back() {
        assert_ne!(SlotState::Free, SlotState::AwaitingMap);
        assert_ne!(SlotState::AwaitingMap, SlotState::Ready);
        assert_ne!(SlotState::Free, SlotState::Ready);
    }

    /// The state machine has to distinguish "copy recorded" from "map requested",
    /// because `map_async` may only be called after the submit. Collapsing them was
    /// the second version of this bug: the map went out before the submit and the
    /// submit then failed validation on its own copy target.
    #[test]
    fn recording_a_copy_is_a_distinct_state_from_awaiting_a_map() {
        assert_ne!(SlotState::Recorded, SlotState::AwaitingMap);
        assert_ne!(SlotState::Recorded, SlotState::Free);
        // And a recorded slot is not reusable: resolve looks for Free only.
        assert_ne!(SlotState::Recorded, SlotState::Free);
    }

    /// `after_submit` must be safe to call with nothing recorded, so the frame loop
    /// needs no branch around it.
    #[test]
    fn after_submit_is_harmless_with_nothing_pending() {
        let mut timer = GpuTimer::disabled();
        timer.after_submit();
        timer.after_submit();
        assert_eq!(timer.samples(), 0);
    }

    /// Enough slots that a driver-scheduled callback does not starve the profiler.
    /// Two was measured to be too few at this frame rate.
    #[test]
    fn there_are_enough_readback_slots_for_a_high_frame_rate() {
        assert!(
            FRAMES_IN_FLIGHT >= 3,
            "{FRAMES_IN_FLIGHT} slots will drop most frames' timings"
        );
    }

    #[test]
    fn a_disabled_timer_never_skips_because_it_never_resolves() {
        let timer = GpuTimer::disabled();
        assert_eq!(timer.skipped(), 0);
    }

    #[test]
    fn samples_count_only_successful_folds() {
        let mut timer = GpuTimer::disabled();
        timer.period_ns = 1.0;
        timer.fold(&[0u64; QUERY_COUNT as usize], &[GpuPass::Scene]);
        assert_eq!(timer.samples(), 0, "a dropped sample must not count");
        let mut good = [0u64; QUERY_COUNT as usize];
        good[GpuPass::Scene.begin_index() as usize] = 0;
        good[GpuPass::Scene.end_index() as usize] = 1_000;
        timer.fold(&good, &[GpuPass::Scene]);
        assert_eq!(timer.samples(), 1);
    }
}
