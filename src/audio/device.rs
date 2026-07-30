// Audio output device: hands mixed samples to the operating system.
//
// This is the one part of the audio stack that cannot be unit tested — it needs
// a sound card — so it is kept as thin as possible and everything decidable is
// pushed into `mixer`, which is pure computation over slices.
//
// Windows: `waveOut` from winmm, declared here as raw FFI. No `windows` crate,
// per the project's "no external crates" rule. waveOut rather than WASAPI because
// WASAPI's initialisation path is COM — `CoInitializeEx`, several interface
// pointers, a format negotiation and an event handle — where waveOut is four
// calls, and the extra latency (a few tens of milliseconds) does not matter for a
// renderer whose whole output is characters.
//
// Elsewhere: a silent stub that reports its own absence. Deliberately not a
// compile error: the rest of the engine must build and run on any platform, and
// the mixer is still fully testable without an output device.

use crate::audio::mixer::{Mixer, OUTPUT_CHANNELS};
use crate::engine::core::{EngineError, Result};

/// Output sample rate. 48 kHz is the native rate of essentially every modern
/// device, so choosing it avoids a resample in the OS layer that would be outside
/// this engine's control (and therefore outside its ability to test).
pub const SAMPLE_RATE: u32 = 48_000;

/// Frames per buffer submitted to the OS.
///
/// 1024 frames at 48 kHz is ~21 ms. Small enough that a sound triggered by a key
/// press does not feel detached from it, large enough that the mixing call
/// happens ~47 times a second rather than per audio frame.
pub const BUFFER_FRAMES: usize = 1024;

/// How many buffers are kept queued.
///
/// Three: one playing, one queued behind it, one being filled. Two is enough in
/// principle but leaves no slack — a single late frame produces an audible gap,
/// and a renderer that occasionally takes 30 ms per frame will have late frames.
const BUFFER_COUNT: usize = 3;

/// Whether an audio device is present, and if not, why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Open and accepting samples.
    Active,
    /// No output device, or the platform backend refused to open one.
    Unavailable,
    /// This platform has no backend in this engine.
    Unsupported,
}

impl DeviceStatus {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    /// One-line explanation for the startup log and the console.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Active => "audio output open",
            Self::Unavailable => "no audio output device, running silent",
            Self::Unsupported => "no audio backend for this platform, running silent",
        }
    }

    /// Short tag for the HUD, where there is one cell per character.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Active => "ON",
            Self::Unavailable => "NO-DEV",
            Self::Unsupported => "NO-BACKEND",
        }
    }
}

#[cfg(windows)]
mod winmm {
    //! Minimal `waveOut` bindings. Only the handful of entry points and the two
    //! structs this engine needs, declared rather than pulled from a crate.

    use core::ffi::c_void;

    pub type Handle = *mut c_void;
    pub type MmResult = u32;

    pub const MMSYSERR_NOERROR: MmResult = 0;
    /// `WAVE_MAPPER`: let the OS pick the default output device.
    pub const WAVE_MAPPER: u32 = 0xFFFF_FFFF;
    /// `WAVE_FORMAT_PCM`.
    pub const WAVE_FORMAT_PCM: u16 = 1;
    /// `WHDR_DONE`: the device has finished with this buffer.
    pub const WHDR_DONE: u32 = 0x0000_0001;

    /// `WAVEFORMATEX`, as the OS expects it byte for byte.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct WaveFormatEx {
        pub format_tag: u16,
        pub channels: u16,
        pub samples_per_sec: u32,
        pub avg_bytes_per_sec: u32,
        pub block_align: u16,
        pub bits_per_sample: u16,
        pub extra_size: u16,
    }

    /// `WAVEHDR`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct WaveHdr {
        pub data: *mut u8,
        pub buffer_length: u32,
        pub bytes_recorded: u32,
        pub user: usize,
        pub flags: u32,
        pub loops: u32,
        pub next: *mut WaveHdr,
        pub reserved: usize,
    }

    impl Default for WaveHdr {
        fn default() -> Self {
            Self {
                data: core::ptr::null_mut(),
                buffer_length: 0,
                bytes_recorded: 0,
                user: 0,
                flags: 0,
                loops: 0,
                next: core::ptr::null_mut(),
                reserved: 0,
            }
        }
    }

    #[link(name = "winmm")]
    unsafe extern "system" {
        pub fn waveOutOpen(
            handle: *mut Handle,
            device_id: u32,
            format: *const WaveFormatEx,
            callback: usize,
            instance: usize,
            flags: u32,
        ) -> MmResult;
        pub fn waveOutClose(handle: Handle) -> MmResult;
        pub fn waveOutPrepareHeader(handle: Handle, header: *mut WaveHdr, size: u32) -> MmResult;
        pub fn waveOutUnprepareHeader(handle: Handle, header: *mut WaveHdr, size: u32) -> MmResult;
        pub fn waveOutWrite(handle: Handle, header: *mut WaveHdr, size: u32) -> MmResult;
        pub fn waveOutReset(handle: Handle) -> MmResult;
    }
}

/// An open output device, or a silent stand-in for one.
pub struct AudioDevice {
    status: DeviceStatus,
    /// Frames submitted since the device opened, for the HUD.
    frames_submitted: u64,
    /// Times `submit` found the device with nothing left to play while a voice
    /// was still active — i.e. audible gaps.
    ///
    /// This is deliberately *not* "times every buffer was busy". A full queue is
    /// the healthy state: it means the renderer is ahead of the device, which at
    /// 1300 FPS against a 47 Hz buffer rate is almost every frame. Counting that
    /// reported 1876 starvations on a run whose audio was in fact continuous —
    /// a metric that fires constantly during correct operation is worse than no
    /// metric, because it trains you to ignore it.
    starved: u64,
    #[cfg(windows)]
    backend: Option<Box<WindowsBackend>>,
}

#[cfg(windows)]
struct WindowsBackend {
    handle: winmm::Handle,
    /// Interleaved 16-bit sample storage, one block per queued buffer. Boxed and
    /// never moved after `waveOutPrepareHeader`: the OS holds a raw pointer into
    /// each block for as long as it is queued, so a reallocation would leave the
    /// device reading freed memory.
    blocks: Vec<Box<[i16]>>,
    headers: Vec<winmm::WaveHdr>,
    /// Whether each block has ever been handed to the device.
    ///
    /// Tracked here rather than inferred from `WHDR_DONE`, because
    /// `waveOutPrepareHeader` rewrites the flags field: pre-setting `WHDR_DONE`
    /// to mean "free" does not survive preparation, so every block looked
    /// permanently busy and the mixer submitted exactly zero buffers while
    /// reporting a starvation every frame. A never-submitted block is free by
    /// definition and needs no flag to say so.
    submitted: Vec<bool>,
    /// Next block to try to fill.
    next: usize,
}

#[cfg(windows)]
impl Drop for WindowsBackend {
    fn drop(&mut self) {
        // Order matters: reset first so the device stops reading, then unprepare
        // each header, then close. Closing while buffers are still queued leaks
        // them and can hang the call.
        unsafe {
            winmm::waveOutReset(self.handle);
            let size = core::mem::size_of::<winmm::WaveHdr>() as u32;
            for header in &mut self.headers {
                winmm::waveOutUnprepareHeader(self.handle, header, size);
            }
            winmm::waveOutClose(self.handle);
        }
    }
}

impl AudioDevice {
    /// Open the default output device, or fall back to silence.
    ///
    /// Never fails: a machine with no sound card must still run the engine, and
    /// an audio subsystem that refuses to initialise should not take the renderer
    /// with it. What happened is reported through `status()`.
    pub fn open() -> Self {
        #[cfg(windows)]
        {
            match Self::open_windows() {
                Ok(backend) => Self {
                    status: DeviceStatus::Active,
                    frames_submitted: 0,
                    starved: 0,
                    backend: Some(Box::new(backend)),
                },
                Err(e) => {
                    eprintln!("audio: {e}");
                    Self {
                        status: DeviceStatus::Unavailable,
                        frames_submitted: 0,
                        starved: 0,
                        backend: None,
                    }
                }
            }
        }
        #[cfg(not(windows))]
        {
            Self {
                status: DeviceStatus::Unsupported,
                frames_submitted: 0,
                starved: 0,
            }
        }
    }

    #[cfg(windows)]
    fn open_windows() -> Result<WindowsBackend> {
        let block_align = (OUTPUT_CHANNELS * 2) as u16;
        let format = winmm::WaveFormatEx {
            format_tag: winmm::WAVE_FORMAT_PCM,
            channels: OUTPUT_CHANNELS as u16,
            samples_per_sec: SAMPLE_RATE,
            avg_bytes_per_sec: SAMPLE_RATE * block_align as u32,
            block_align,
            bits_per_sample: 16,
            extra_size: 0,
        };

        let mut handle: winmm::Handle = core::ptr::null_mut();
        // SAFETY: `handle` is a valid out-pointer and `format` outlives the call.
        // A null callback with no flags means "no notifications"; buffer
        // completion is polled through WHDR_DONE instead, which avoids running
        // engine code on the OS audio thread.
        let result = unsafe {
            winmm::waveOutOpen(
                &mut handle,
                winmm::WAVE_MAPPER,
                &format,
                0,
                0,
                0,
            )
        };
        if result != winmm::MMSYSERR_NOERROR || handle.is_null() {
            return Err(EngineError::InvalidState(format!(
                "waveOutOpen failed with code {result} (no output device?)"
            )));
        }

        let samples_per_block = BUFFER_FRAMES * OUTPUT_CHANNELS;
        let mut blocks: Vec<Box<[i16]>> = Vec::with_capacity(BUFFER_COUNT);
        let mut headers: Vec<winmm::WaveHdr> = Vec::with_capacity(BUFFER_COUNT);
        for _ in 0..BUFFER_COUNT {
            blocks.push(vec![0i16; samples_per_block].into_boxed_slice());
        }
        for block in &mut blocks {
            let mut header = winmm::WaveHdr {
                data: block.as_mut_ptr() as *mut u8,
                buffer_length: (samples_per_block * 2) as u32,
                ..Default::default()
            };
            // SAFETY: `header` points at storage that lives in `blocks`, which is
            // moved into the backend and never reallocated afterwards.
            let result = unsafe {
                winmm::waveOutPrepareHeader(
                    handle,
                    &mut header,
                    core::mem::size_of::<winmm::WaveHdr>() as u32,
                )
            };
            if result != winmm::MMSYSERR_NOERROR {
                // SAFETY: the handle is open and this is the documented cleanup.
                unsafe {
                    winmm::waveOutClose(handle);
                }
                return Err(EngineError::InvalidState(format!(
                    "waveOutPrepareHeader failed with code {result}"
                )));
            }
            headers.push(header);
        }

        let submitted = vec![false; BUFFER_COUNT];
        Ok(WindowsBackend {
            handle,
            blocks,
            headers,
            submitted,
            next: 0,
        })
    }

    pub fn status(&self) -> DeviceStatus {
        self.status
    }

    pub fn frames_submitted(&self) -> u64 {
        self.frames_submitted
    }

    /// How many times the device ran dry while something was still playing.
    pub fn starved(&self) -> u64 {
        self.starved
    }

    /// Buffers currently queued with the device, out of `BUFFER_COUNT`.
    ///
    /// The useful health number: 0 while a voice is playing means an audible gap,
    /// and a queue that hovers at 1 means the frame rate is only just keeping up.
    pub fn queued_buffers(&self) -> usize {
        #[cfg(windows)]
        {
            self.backend
                .as_ref()
                .map(|b| {
                    b.headers
                        .iter()
                        .zip(b.submitted.iter())
                        .filter(|(h, submitted)| {
                            **submitted && h.flags & winmm::WHDR_DONE == 0
                        })
                        .count()
                })
                .unwrap_or(0)
        }
        #[cfg(not(windows))]
        {
            0
        }
    }

    /// Fill every free buffer from `mixer` and hand it to the device.
    ///
    /// Called once per rendered frame rather than from an audio callback. That is
    /// a deliberate trade: it keeps all engine state single-threaded — the mixer
    /// reads listener and voice state the render loop writes — at the cost of
    /// needing the frame rate to stay above the buffer rate. At 47 buffers per
    /// second and three queued buffers, the frame rate has to fall below ~16 FPS
    /// before audio starves, and `starved()` reports it when it does.
    pub fn submit(&mut self, mixer: &mut Mixer) {
        if !self.status.is_active() {
            return;
        }
        #[cfg(windows)]
        {
            let Some(backend) = self.backend.as_mut() else {
                return;
            };
            let mut scratch = [0.0f32; BUFFER_FRAMES * OUTPUT_CHANNELS];
            // Measured before filling anything: if the device has nothing queued
            // at the top of a frame and a voice is still playing, the gap has
            // already been heard.
            let queued_before = backend
                .headers
                .iter()
                .zip(backend.submitted.iter())
                .filter(|(h, submitted)| **submitted && h.flags & winmm::WHDR_DONE == 0)
                .count();
            if queued_before == 0 && mixer.voice_count() > 0 {
                self.starved += 1;
            }
            for _ in 0..BUFFER_COUNT {
                let index = backend.next;
                backend.next = (backend.next + 1) % BUFFER_COUNT;
                // Free if it has never been submitted, or if the device has
                // finished with it.
                let free = !backend.submitted[index]
                    || backend.headers[index].flags & winmm::WHDR_DONE != 0;
                if !free {
                    continue;
                }
                mixer.mix(&mut scratch, BUFFER_FRAMES);
                for (dst, src) in backend.blocks[index].iter_mut().zip(scratch.iter()) {
                    *dst = float_to_i16(*src);
                }
                backend.headers[index].flags &= !winmm::WHDR_DONE;
                // SAFETY: the header was prepared against this handle and its
                // data pointer still refers to the same never-moved block.
                let result = unsafe {
                    winmm::waveOutWrite(
                        backend.handle,
                        &mut backend.headers[index],
                        core::mem::size_of::<winmm::WaveHdr>() as u32,
                    )
                };
                if result != winmm::MMSYSERR_NOERROR {
                    // Leave `submitted` false so the block is retried rather than
                    // dropping out of the rotation.
                    backend.headers[index].flags |= winmm::WHDR_DONE;
                    continue;
                }
                backend.submitted[index] = true;
                self.frames_submitted += BUFFER_FRAMES as u64;
            }

        }
        #[cfg(not(windows))]
        {
            let _ = mixer;
        }
    }

    /// Stop playback immediately, discarding anything queued.
    pub fn stop(&mut self) {
        #[cfg(windows)]
        {
            if let Some(backend) = self.backend.as_mut() {
                // SAFETY: the handle is open; reset is the documented way to
                // discard queued buffers, and it sets WHDR_DONE on each.
                unsafe {
                    winmm::waveOutReset(backend.handle);
                }
                for header in &mut backend.headers {
                    header.flags |= winmm::WHDR_DONE;
                }
                backend.submitted.fill(false);
            }
        }
    }
}

/// Convert a normalized float sample to 16-bit PCM.
///
/// Clamped before scaling, and scaled by 32767 rather than 32768: a value of
/// exactly 1.0 times 32768 is 32768, which does not fit in an i16 and wraps to
/// -32768 — a full-scale positive sample becomes a full-scale negative one, heard
/// as a loud click on every peak.
pub fn float_to_i16(sample: f32) -> i16 {
    if !sample.is_finite() {
        return 0;
    }
    (sample.clamp(-1.0, 1.0) * 32767.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wrap-around this conversion exists to prevent: 1.0 * 32768 does not fit
    /// in an i16, so a full-scale positive sample would come out as full-scale
    /// negative and click on every peak.
    #[test]
    fn full_scale_does_not_wrap_to_negative() {
        assert_eq!(float_to_i16(1.0), 32767);
        assert_eq!(float_to_i16(-1.0), -32767);
        assert!(float_to_i16(1.0) > 0, "positive full scale must stay positive");
    }

    #[test]
    fn out_of_range_input_is_clamped() {
        assert_eq!(float_to_i16(5.0), 32767);
        assert_eq!(float_to_i16(-5.0), -32767);
    }

    #[test]
    fn non_finite_input_becomes_silence() {
        assert_eq!(float_to_i16(f32::NAN), 0);
        assert_eq!(float_to_i16(f32::INFINITY), 0);
        assert_eq!(float_to_i16(f32::NEG_INFINITY), 0);
    }

    #[test]
    fn silence_and_midpoints_convert_proportionally() {
        assert_eq!(float_to_i16(0.0), 0);
        let half = float_to_i16(0.5);
        assert!((half - 16383).abs() <= 2, "0.5 became {half}");
    }

    #[test]
    fn only_active_status_accepts_samples() {
        assert!(DeviceStatus::Active.is_active());
        assert!(!DeviceStatus::Unavailable.is_active());
        assert!(!DeviceStatus::Unsupported.is_active());
        for s in [
            DeviceStatus::Active,
            DeviceStatus::Unavailable,
            DeviceStatus::Unsupported,
        ] {
            assert!(!s.describe().is_empty());
            assert!(!s.tag().is_empty());
        }
    }

    /// The buffer geometry has to stay consistent with the mixer's channel count,
    /// or the scratch buffer and the device block would disagree about size.
    #[test]
    fn buffer_geometry_is_consistent() {
        assert_eq!(OUTPUT_CHANNELS, 2);
        assert!(BUFFER_FRAMES > 0);
        assert!(BUFFER_COUNT >= 2, "double buffering at minimum");
        // ~21 ms at 48 kHz: short enough not to feel detached from the key press
        // that triggered the sound.
        let latency_ms = BUFFER_FRAMES as f32 / SAMPLE_RATE as f32 * 1000.0;
        assert!(
            latency_ms < 40.0,
            "buffer latency of {latency_ms} ms would be noticeable"
        );
    }

    /// A device that failed to open must accept `submit` without doing anything,
    /// so a machine with no sound card runs the engine rather than crashing in it.
    #[test]
    fn submitting_to_an_inactive_device_is_a_no_op() {
        let mut device = AudioDevice {
            status: DeviceStatus::Unavailable,
            frames_submitted: 0,
            starved: 0,
            #[cfg(windows)]
            backend: None,
        };

        let mut mixer = Mixer::new(SAMPLE_RATE);
        device.submit(&mut mixer);
        device.stop();
        assert_eq!(device.frames_submitted(), 0);
        assert_eq!(device.queued_buffers(), 0);
        assert_eq!(
            device.starved(),
            0,
            "a device that was never open cannot starve"
        );
    }
}
