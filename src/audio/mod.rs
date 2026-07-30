// Audio: WAV decoding, software mixing with 3D spatialisation, and playback.
//
// Self-implemented per the project's "no external crates" rule. That rule shapes
// the module boundary: `wav` and `mixer` are pure computation over slices, so
// every claim about decoding, attenuation, panning and Doppler is checked by
// reading the buffers they produce, with no audio hardware in the test. Only
// `device` touches the OS, and it is the one part unit tests cannot reach.
//
// What is here and what is not:
// - WAV: PCM 8/16/24/32-bit and IEEE float 32/64-bit, including
//   WAVE_FORMAT_EXTENSIBLE. Compressed RIFF payloads are rejected by name.
// - OGG/Vorbis: not implemented. Vorbis needs a codebook decoder, a floor/residue
//   decoder and an MDCT — several thousand lines, and the roadmap item was written
//   before that was weighed against the rest of the phase. Recorded in the phase
//   notes rather than half-built.
// - Streaming: a long track is an ordinary voice, seeked into, rather than a
//   separate path. Duplicating the resampler and panner for "music" would buy
//   nothing.

pub mod device;
pub mod mixer;
pub mod synth;
pub mod wav;

pub use device::AudioDevice;
pub use mixer::{Attenuation, Listener, Mixer, Spatial, Voice};
pub use synth::demo_sounds;
pub use wav::AudioBuffer;
