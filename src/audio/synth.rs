// Procedural sound generation.
//
// The repository ships no audio assets, and inventing a `.wav` to commit would
// only test the decoder against a file this same code wrote. Generating sounds
// instead gives the mixer something real to play — with known frequency content,
// so a wrong sample rate or a broken resampler is audible as a wrong pitch rather
// than as vague noise — and keeps the audio subsystem demonstrable in a clone of
// the repository with nothing else downloaded.
//
// All three generators are short, tonal and enveloped, because those are the
// properties that make mixing errors audible: a click at the start means the
// envelope was skipped, a buzz means samples are being read as the wrong type,
// and a wobble means the resampler is drifting.

use crate::audio::wav::AudioBuffer;

/// Sample rate the generators produce at.
///
/// Deliberately *not* the device rate (48 kHz): a generated buffer at the output
/// rate would exercise the resampler's identity case only, and the resampler is
/// the part most likely to be subtly wrong. 44.1 kHz forces a real 0.91875 ratio.
pub const SYNTH_RATE: u32 = 44_100;

/// Linear attack/release envelope, in [0, 1].
///
/// Without one, a tone starts and ends on a discontinuity, which is heard as a
/// click at both ends — loud enough on a short sound to be most of what you hear.
fn envelope(frame: usize, total: usize, attack: f32, release: f32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let t = frame as f32 / total as f32;
    let attack = attack.clamp(0.001, 0.5);
    let release = release.clamp(0.001, 0.9);
    if t < attack {
        t / attack
    } else if t > 1.0 - release {
        ((1.0 - t) / release).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// A pure sine tone.
pub fn tone(frequency: f32, seconds: f32, amplitude: f32) -> AudioBuffer {
    let frames = ((seconds.max(0.0) * SYNTH_RATE as f32) as usize).max(1);
    let mut samples = Vec::with_capacity(frames);
    let step = std::f32::consts::TAU * frequency / SYNTH_RATE as f32;
    for i in 0..frames {
        let value = (step * i as f32).sin() * amplitude * envelope(i, frames, 0.02, 0.3);
        samples.push(value);
    }
    AudioBuffer {
        samples,
        channels: 1,
        sample_rate: SYNTH_RATE,
    }
}

/// A short percussive blip: a tone whose pitch falls as it decays.
pub fn blip(start_frequency: f32, seconds: f32, amplitude: f32) -> AudioBuffer {
    let frames = ((seconds.max(0.0) * SYNTH_RATE as f32) as usize).max(1);
    let mut samples = Vec::with_capacity(frames);
    // Phase is accumulated rather than computed from `i * step`, because the step
    // changes every sample: multiplying the *current* frequency by the frame index
    // would jump the phase and produce a buzz instead of a glide.
    let mut phase = 0.0f32;
    for i in 0..frames {
        let t = i as f32 / frames as f32;
        let frequency = start_frequency * (1.0 - 0.6 * t);
        phase += std::f32::consts::TAU * frequency / SYNTH_RATE as f32;
        let value = phase.sin() * amplitude * envelope(i, frames, 0.005, 0.6);
        samples.push(value);
    }
    AudioBuffer {
        samples,
        channels: 1,
        sample_rate: SYNTH_RATE,
    }
}

/// Filtered noise: a whoosh.
///
/// The noise is deterministic (a fixed-seed integer hash, the same one the tracer
/// uses) so two runs produce identical audio. A random seed would make any
/// measurement of the mix irreproducible.
pub fn noise(seconds: f32, amplitude: f32) -> AudioBuffer {
    let frames = ((seconds.max(0.0) * SYNTH_RATE as f32) as usize).max(1);
    let mut samples = Vec::with_capacity(frames);
    let mut state = 0x1234_5678u32;
    // One-pole low-pass state. Raw white noise is harsh and, more usefully for
    // debugging, indistinguishable from the buzz a sample-format error produces.
    let mut filtered = 0.0f32;
    for i in 0..frames {
        state = state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        let white = (state >> 8) as f32 / 8_388_608.0 - 1.0;
        filtered += (white - filtered) * 0.15;
        samples.push(filtered * amplitude * envelope(i, frames, 0.1, 0.5));
    }
    AudioBuffer {
        samples,
        channels: 1,
        sample_rate: SYNTH_RATE,
    }
}

/// The demo's sound set: three recognisably different sounds, so a listener can
/// tell which one played.
pub fn demo_sounds() -> Vec<std::sync::Arc<AudioBuffer>> {
    vec![
        std::sync::Arc::new(tone(440.0, 0.5, 0.6)),
        std::sync::Arc::new(blip(880.0, 0.25, 0.7)),
        std::sync::Arc::new(noise(0.6, 0.5)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tone_has_the_right_length_and_rate() {
        let buf = tone(440.0, 0.5, 0.5);
        assert_eq!(buf.sample_rate, SYNTH_RATE);
        assert_eq!(buf.channels, 1);
        assert!((buf.duration() - 0.5).abs() < 0.01, "{}", buf.duration());
    }

    /// The generated rate must differ from the output rate, or the resampler's
    /// non-identity path would never run in the demo — and that is the path most
    /// likely to be subtly wrong.
    #[test]
    fn the_synth_rate_is_not_the_output_rate() {
        assert_ne!(SYNTH_RATE, crate::audio::device::SAMPLE_RATE);
    }

    /// Counting zero crossings recovers the frequency. A generator that used the
    /// wrong rate in its phase step would produce a tone at the wrong pitch, which
    /// no length or amplitude check would notice.
    #[test]
    fn a_tone_really_oscillates_at_its_stated_frequency() {
        let frequency = 441.0;
        let buf = tone(frequency, 1.0, 0.9);
        let mut crossings = 0;
        // Skip the attack and release, where the envelope suppresses the signal.
        let start = buf.frame_count() / 10;
        let end = buf.frame_count() * 9 / 10;
        for i in start + 1..end {
            let (a, b) = (buf.samples[i - 1], buf.samples[i]);
            if (a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0) {
                crossings += 1;
            }
        }
        // Two crossings per cycle over 80% of a second.
        let measured = crossings as f32 / 2.0 / 0.8;
        assert!(
            (measured - frequency).abs() < frequency * 0.05,
            "measured {measured} Hz for a {frequency} Hz tone"
        );
    }

    /// A tone that starts on a discontinuity clicks, and on a half-second sound
    /// the click is most of what you hear.
    #[test]
    fn every_generator_fades_in_and_out() {
        for buf in [tone(440.0, 0.3, 1.0), blip(880.0, 0.3, 1.0), noise(0.3, 1.0)] {
            let first = buf.samples[0].abs();
            let last = buf.samples[buf.samples.len() - 1].abs();
            assert!(first < 0.05, "starts at {first}, which would click");
            assert!(last < 0.05, "ends at {last}, which would click");
        }
    }

    #[test]
    fn generated_samples_stay_in_range() {
        for buf in [tone(440.0, 0.2, 1.0), blip(1200.0, 0.2, 1.0), noise(0.2, 1.0)] {
            assert!(
                buf.samples.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
                "a generator produced an out-of-range or non-finite sample"
            );
        }
    }

    /// A blip's pitch must actually fall. Accumulating phase is what makes that
    /// work; computing it as `i * current_step` jumps the phase and buzzes.
    #[test]
    fn a_blip_glides_downward() {
        let buf = blip(1000.0, 0.5, 0.9);
        let crossings_in = |from: usize, to: usize| {
            let mut n = 0;
            for i in from + 1..to {
                let (a, b) = (buf.samples[i - 1], buf.samples[i]);
                if (a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0) {
                    n += 1;
                }
            }
            n
        };
        let quarter = buf.frame_count() / 4;
        let early = crossings_in(0, quarter);
        let late = crossings_in(quarter * 2, quarter * 3);
        assert!(
            early > late,
            "the pitch should fall: {early} crossings early vs {late} later"
        );
    }

    /// Deterministic noise: a random seed would make any measurement of the mix
    /// irreproducible between runs.
    #[test]
    fn noise_is_deterministic() {
        assert_eq!(noise(0.1, 0.5).samples, noise(0.1, 0.5).samples);
    }

    /// Filtered, not white: raw white noise is indistinguishable from the buzz a
    /// sample-format error produces, which would make it useless for debugging.
    #[test]
    fn noise_is_low_passed_rather_than_white() {
        let buf = noise(0.5, 0.9);
        // A one-pole low-pass makes consecutive samples correlated; white noise
        // does not. Measure the mean absolute difference between neighbours
        // against the mean absolute amplitude.
        let mid = buf.frame_count() / 2;
        let window = &buf.samples[mid - 1000..mid + 1000];
        let mean_step: f32 = window
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f32>()
            / (window.len() - 1) as f32;
        let mean_level: f32 =
            window.iter().map(|s| s.abs()).sum::<f32>() / window.len() as f32;
        assert!(
            mean_step < mean_level,
            "consecutive samples differ by {mean_step} against a level of {mean_level}, \
             which looks like white noise rather than filtered"
        );
    }

    #[test]
    fn the_demo_set_has_three_distinguishable_sounds() {
        let sounds = demo_sounds();
        assert_eq!(sounds.len(), 3);
        for s in &sounds {
            assert!(s.frame_count() > 100, "a sound too short to hear");
        }
        // Different content, not three copies.
        assert_ne!(sounds[0].samples, sounds[1].samples);
        assert_ne!(sounds[1].samples, sounds[2].samples);
    }

    #[test]
    fn a_zero_or_negative_duration_still_yields_a_valid_buffer() {
        for buf in [tone(440.0, 0.0, 0.5), blip(440.0, -1.0, 0.5), noise(0.0, 0.5)] {
            assert!(buf.frame_count() >= 1, "an empty buffer would divide by zero");
            assert!(buf.samples.iter().all(|s| s.is_finite()));
        }
    }
}
