// Software mixer: 3D spatialisation, streaming, and the sample loop that turns
// a set of playing voices into interleaved stereo output.
//
// Self-implemented, and deliberately decoupled from any output device. This
// module answers "what samples should the speakers receive?" and nothing else,
// which is what makes it testable: every claim below about attenuation, panning
// and Doppler is checked by reading the buffer it produces, with no audio
// hardware involved. The device backend (`audio::device`) does the platform call
// and calls in here for samples.
//
// Streaming and one-shot playback are the same code path, distinguished only by
// where the samples come from — a `Source` is either a whole decoded buffer or a
// windowed view of a longer one. A separate "music" path would duplicate the
// resampler, the panner and the mixing loop for no gain.

use std::sync::Arc;

use crate::audio::wav::AudioBuffer;
use crate::engine::math::Vec3;

/// Output channel count. Stereo: it is what spatialisation needs (a mono output
/// cannot pan) and what every desktop device has.
pub const OUTPUT_CHANNELS: usize = 2;

/// Speed of sound in air, m/s, for Doppler.
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Largest pitch shift Doppler may apply, as a ratio.
///
/// Uncapped, a source moving at or faster than the speed of sound divides by
/// zero or goes negative — a supersonic bullet would produce a NaN that silences
/// the whole mix. Clamping keeps the artefact local and audible instead.
const MAX_DOPPLER: f32 = 2.0;
const MIN_DOPPLER: f32 = 0.5;

/// How a voice's volume falls off with distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Attenuation {
    /// No falloff: UI sounds, narration, music.
    None,
    /// Inverse-distance falloff, silent beyond `max_distance`.
    ///
    /// Not true inverse-square: at game distances inverse-square drops to
    /// inaudibility within a couple of metres, which is why essentially no game
    /// uses it unmodified. `ref_distance` is where the volume is 1.0.
    Inverse {
        ref_distance: f32,
        max_distance: f32,
    },
    /// Linear falloff from `ref_distance` to `max_distance`. Reaches exactly zero,
    /// which inverse falloff never does — useful when a sound must be guaranteed
    /// inaudible past a boundary.
    Linear {
        ref_distance: f32,
        max_distance: f32,
    },
}

impl Default for Attenuation {
    fn default() -> Self {
        Self::Inverse {
            ref_distance: 1.0,
            max_distance: 60.0,
        }
    }
}

impl Attenuation {
    /// Gain multiplier at `distance`.
    pub fn gain(&self, distance: f32) -> f32 {
        match *self {
            Self::None => 1.0,
            Self::Inverse {
                ref_distance,
                max_distance,
            } => {
                let r = ref_distance.max(0.01);
                if distance >= max_distance {
                    return 0.0;
                }
                (r / distance.max(r)).clamp(0.0, 1.0)
            }
            Self::Linear {
                ref_distance,
                max_distance,
            } => {
                let r = ref_distance.max(0.0);
                let m = max_distance.max(r + 0.01);
                if distance <= r {
                    1.0
                } else if distance >= m {
                    0.0
                } else {
                    1.0 - (distance - r) / (m - r)
                }
            }
        }
    }
}

/// Where a voice sits in the world, or that it does not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spatial {
    /// Not positioned: plays at full volume, centred. Music and UI.
    None,
    /// Positioned in world space.
    At {
        position: Vec3,
        /// Velocity, for Doppler. Zero disables the shift for this voice.
        velocity: Vec3,
        attenuation: Attenuation,
    },
}

impl Spatial {
    /// A stationary positioned source with the default falloff.
    pub fn at(position: Vec3) -> Self {
        Self::At {
            position,
            velocity: Vec3::ZERO,
            attenuation: Attenuation::default(),
        }
    }
}

/// The listener: where the ears are and which way they face.
#[derive(Clone, Copy, Debug)]
pub struct Listener {
    pub position: Vec3,
    /// Unit forward direction.
    pub forward: Vec3,
    /// Unit up direction.
    pub up: Vec3,
    /// Listener velocity, for Doppler.
    pub velocity: Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::UNIT_Y,
            velocity: Vec3::ZERO,
        }
    }
}

impl Listener {
    /// The listener's right-hand direction, derived rather than stored so it can
    /// never disagree with `forward` and `up`.
    pub fn right(&self) -> Vec3 {
        let r = self.forward.cross(self.up);
        if r.length_squared() < 1e-8 {
            // Looking straight up or down: any perpendicular will do, but it must
            // be a *consistent* one or the stereo image would flip frame to frame.
            Vec3::UNIT_X
        } else {
            r.normalize()
        }
    }
}

/// Sample data a voice reads from.
///
/// `Arc` because a sound effect is played many times over and the samples must
/// not be copied per voice — a 3-second 48 kHz stereo buffer is 1.2 MB, and a
/// dozen simultaneous voices would be 14 MB of pointless duplication.
pub type SharedBuffer = Arc<AudioBuffer>;

/// One playing sound.
#[derive(Clone, Debug)]
pub struct Voice {
    buffer: SharedBuffer,
    /// Playback position in source frames, fractional because the source rate
    /// and the output rate rarely match.
    cursor: f64,
    /// Volume before spatialisation.
    pub gain: f32,
    /// Playback speed multiplier, before Doppler.
    pub speed: f32,
    /// Whether to wrap around at the end.
    pub looping: bool,
    /// World placement, or `None`.
    pub spatial: Spatial,
    /// Set when a non-looping voice has run out of samples.
    finished: bool,
}

impl Voice {
    /// A voice that plays `buffer` once, unpositioned, at full volume.
    pub fn new(buffer: SharedBuffer) -> Self {
        Self {
            buffer,
            cursor: 0.0,
            gain: 1.0,
            speed: 1.0,
            looping: false,
            spatial: Spatial::None,
            finished: false,
        }
    }

    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = gain.max(0.0);
        self
    }

    pub fn with_speed(mut self, speed: f32) -> Self {
        // A non-positive speed would never advance the cursor, so a non-looping
        // voice would occupy a slot forever.
        self.speed = speed.max(0.01);
        self
    }

    pub fn looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    pub fn spatial(mut self, spatial: Spatial) -> Self {
        self.spatial = spatial;
        self
    }

    /// Start playback from `seconds` into the buffer.
    ///
    /// This is what makes music streaming work without a second code path: a
    /// long track is the same kind of voice, seeked into.
    pub fn seek(&mut self, seconds: f32) {
        let frame = (seconds.max(0.0) as f64) * self.buffer.sample_rate as f64;
        self.cursor = frame.min(self.buffer.frame_count() as f64);
        self.finished = false;
    }

    /// Current playback position in seconds.
    pub fn position_seconds(&self) -> f32 {
        if self.buffer.sample_rate == 0 {
            return 0.0;
        }
        (self.cursor / self.buffer.sample_rate as f64) as f32
    }

    /// Whether this voice has finished and can be dropped.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

/// The mixer: a set of voices and the listener they are heard from.
pub struct Mixer {
    voices: Vec<Voice>,
    /// Output sample rate; voices are resampled to it.
    sample_rate: u32,
    /// Listener pose, for spatialisation.
    pub listener: Listener,
    /// Overall output gain.
    pub master_gain: f32,
    /// Voices dropped in the most recent `mix` because they finished.
    retired: usize,
    /// Peak absolute sample of the most recent `mix`, before clamping. Above 1.0
    /// means the mix was clipped, which is exactly the kind of thing that is
    /// obvious in the ear and invisible in the code.
    peak: f32,
}

/// Most voices mixed at once.
///
/// A cap rather than unbounded: mixing is per-sample per-voice, so an unbounded
/// count lets a spammed sound effect turn the audio thread into the frame's
/// bottleneck. Beyond the cap the quietest voice is replaced, so the loudest
/// (usually the most important) sounds survive.
pub const MAX_VOICES: usize = 32;

impl Mixer {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            voices: Vec::new(),
            sample_rate: sample_rate.max(1),
            listener: Listener::default(),
            master_gain: 1.0,
            retired: 0,
            peak: 0.0,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    pub fn retired(&self) -> usize {
        self.retired
    }

    /// Peak absolute sample of the last `mix`, before clamping.
    pub fn peak(&self) -> f32 {
        self.peak
    }

    /// Start a voice. Returns false if it was dropped because every slot was
    /// occupied by a louder sound.
    pub fn play(&mut self, voice: Voice) -> bool {
        if self.voices.len() < MAX_VOICES {
            self.voices.push(voice);
            return true;
        }
        // Full: replace the quietest voice, so a burst of distant chatter cannot
        // drown out the sound the player needs to hear.
        let incoming = self.effective_gain(&voice);
        let (index, quietest) = self
            .voices
            .iter()
            .enumerate()
            .map(|(i, v)| (i, self.effective_gain(v)))
            .fold((0usize, f32::INFINITY), |acc, (i, g)| {
                if g < acc.1 {
                    (i, g)
                } else {
                    acc
                }
            });
        if incoming > quietest {
            self.voices[index] = voice;
            true
        } else {
            false
        }
    }

    /// Stop everything.
    pub fn stop_all(&mut self) {
        self.voices.clear();
    }

    /// Stop every looping voice, leaving one-shots to finish.
    ///
    /// Separate from `stop_all` because a loop is the thing a caller turns off
    /// deliberately (ambience, music), while a one-shot is expected to run out on
    /// its own — cutting them all would truncate whatever had just been triggered.
    pub fn stop_looping(&mut self) {
        self.voices.retain(|v| !v.looping);
    }

    /// Move every looping voice to `spatial`.
    ///
    /// Exists because a positioned loop is the one voice a caller keeps steering
    /// after starting it (an orbiting source, a moving vehicle), and reaching into
    /// the voice list from outside would mean exposing it mutably — which would
    /// also let a caller invalidate a cursor mid-mix.
    pub fn set_spatial_all_looping(&mut self, spatial: Spatial) {
        for voice in &mut self.voices {
            if voice.looping {
                voice.spatial = spatial;
            }
        }
    }

    /// A voice's audible gain, used both for the eviction decision and as a
    /// cheap "is this worth mixing" test.
    fn effective_gain(&self, voice: &Voice) -> f32 {
        match voice.spatial {
            Spatial::None => voice.gain,
            Spatial::At {
                position,
                attenuation,
                ..
            } => voice.gain * attenuation.gain((position - self.listener.position).length()),
        }
    }

    /// Render `frames` frames of interleaved stereo into `out`.
    ///
    /// `out` must hold `frames * OUTPUT_CHANNELS` samples; it is overwritten, not
    /// added to, so a caller cannot accidentally accumulate the previous buffer.
    pub fn mix(&mut self, out: &mut [f32], frames: usize) {
        let needed = frames * OUTPUT_CHANNELS;
        let limit = needed.min(out.len());
        out[..limit].fill(0.0);
        self.retired = 0;
        self.peak = 0.0;
        if limit == 0 {
            return;
        }

        let listener_right = self.listener.right();
        let listener_position = self.listener.position;
        let listener_velocity = self.listener.velocity;
        let master = self.master_gain.max(0.0);

        for voice in &mut self.voices {
            let (gain, pan, doppler) = match voice.spatial {
                Spatial::None => (voice.gain, 0.0, 1.0),
                Spatial::At {
                    position,
                    velocity,
                    attenuation,
                } => {
                    let to_source = position - listener_position;
                    let distance = to_source.length();
                    let gain = voice.gain * attenuation.gain(distance);
                    // Pan is the component of the source direction along the
                    // listener's right: -1 hard left, +1 hard right. A source
                    // directly ahead or behind pans to centre, which is the known
                    // limitation of amplitude panning — front/back is genuinely
                    // ambiguous without HRTF filtering.
                    let pan = if distance > 1e-4 {
                        (to_source / distance).dot(listener_right).clamp(-1.0, 1.0)
                    } else {
                        // At the listener's exact position there is no direction;
                        // centring is the only stable answer.
                        0.0
                    };
                    let doppler = doppler_ratio(
                        to_source,
                        distance,
                        velocity,
                        listener_velocity,
                    );
                    (gain, pan, doppler)
                }
            };

            if gain <= 1e-5 {
                // Inaudible, but the cursor still has to advance or a looping
                // voice would resume mid-sample when it comes back in range, and
                // a one-shot would never finish.
                advance_silently(voice, frames, self.sample_rate, doppler);
                if voice.is_finished() {
                    self.retired += 1;
                }
                continue;
            }

            // Constant-power panning for positioned voices: the gains are the
            // cosine and sine of the pan angle, so total power is the same at
            // every pan position and a source does not dip as it crosses centre.
            //
            // An unpositioned voice passes through at unity instead. The pan law
            // gives 0.707 per channel at centre — correct for *power* when one
            // mono signal is being spread across two speakers, but wrong for a
            // stereo track that already has a left and a right channel: it would
            // play music and UI sounds 3 dB quieter than authored, for no reason
            // a listener could act on.
            let (left_gain, right_gain) = if matches!(voice.spatial, Spatial::None) {
                (1.0, 1.0)
            } else {
                let angle = (pan + 1.0) * 0.25 * std::f32::consts::PI;
                (angle.cos(), angle.sin())
            };

            let rate_ratio =
                voice.buffer.sample_rate as f64 / self.sample_rate as f64
                    * voice.speed as f64
                    * doppler as f64;
            let frame_count = voice.buffer.frame_count();

            for f in 0..frames {
                let base = f * OUTPUT_CHANNELS;
                if base + 1 >= limit {
                    break;
                }
                if voice.finished {
                    break;
                }
                let (sample_l, sample_r) = read_frame(voice, frame_count);
                // A spatialised voice is heard as one signal from one direction,
                // so its channels are folded together before panning; an
                // unpositioned one keeps its own stereo image.
                let (l, r) = if matches!(voice.spatial, Spatial::None) {
                    (sample_l, sample_r)
                } else {
                    let mono = (sample_l + sample_r) * 0.5;
                    (mono, mono)
                };
                out[base] += l * gain * left_gain * master;
                out[base + 1] += r * gain * right_gain * master;

                voice.cursor += rate_ratio;
                wrap_or_finish(voice, frame_count);
            }
            if voice.is_finished() {
                self.retired += 1;
            }
        }

        // Clamp, and record how far over the mix went. Wrapping instead of
        // clamping is the classic digital-audio failure: an overflow reads as a
        // loud crack rather than as distortion.
        for sample in out[..limit].iter_mut() {
            let magnitude = sample.abs();
            if magnitude > self.peak {
                self.peak = magnitude;
            }
            *sample = sample.clamp(-1.0, 1.0);
        }

        self.voices.retain(|v| !v.is_finished());
    }
}

/// Linearly interpolated stereo read at the voice's fractional cursor.
///
/// Interpolated rather than nearest-neighbour: at a fractional rate ratio,
/// nearest-neighbour resampling produces audible aliasing on anything tonal,
/// which is most sounds.
fn read_frame(voice: &Voice, frame_count: usize) -> (f32, f32) {
    if frame_count == 0 {
        return (0.0, 0.0);
    }
    let index = voice.cursor.floor().max(0.0) as usize;
    let fraction = (voice.cursor - voice.cursor.floor()) as f32;
    let next = if index + 1 < frame_count {
        index + 1
    } else if voice.looping {
        0
    } else {
        index
    };

    let channels = voice.buffer.channels;
    if channels >= 2 {
        let l0 = voice.buffer.sample(index, 0);
        let l1 = voice.buffer.sample(next, 0);
        let r0 = voice.buffer.sample(index, 1);
        let r1 = voice.buffer.sample(next, 1);
        (l0 + (l1 - l0) * fraction, r0 + (r1 - r0) * fraction)
    } else {
        let s0 = voice.buffer.sample(index, 0);
        let s1 = voice.buffer.sample(next, 0);
        let s = s0 + (s1 - s0) * fraction;
        (s, s)
    }
}

/// Wrap a looping voice's cursor, or mark a one-shot finished.
fn wrap_or_finish(voice: &mut Voice, frame_count: usize) {
    if frame_count == 0 {
        voice.finished = true;
        return;
    }
    if voice.cursor < frame_count as f64 {
        return;
    }
    if voice.looping {
        // Modulo rather than reset to zero: at a high playback speed the cursor
        // can overshoot the end by more than one frame, and resetting would drop
        // the remainder and drift the loop point every cycle.
        voice.cursor %= frame_count as f64;
    } else {
        voice.finished = true;
    }
}

/// Advance an inaudible voice's cursor without touching the output.
fn advance_silently(voice: &mut Voice, frames: usize, output_rate: u32, doppler: f32) {
    let frame_count = voice.buffer.frame_count();
    let ratio = voice.buffer.sample_rate as f64 / output_rate.max(1) as f64
        * voice.speed as f64
        * doppler as f64;
    voice.cursor += ratio * frames as f64;
    wrap_or_finish(voice, frame_count);
}

/// Doppler pitch ratio for a source/listener pair.
///
/// Only the components of velocity *along the line between them* matter — a
/// source moving in a circle around the listener has no Doppler shift at all,
/// however fast it goes, which is the property a naive `speed / SPEED_OF_SOUND`
/// gets wrong.
fn doppler_ratio(
    to_source: Vec3,
    distance: f32,
    source_velocity: Vec3,
    listener_velocity: Vec3,
) -> f32 {
    if distance < 1e-4 {
        return 1.0;
    }
    let direction = to_source / distance;
    // Positive when the listener moves toward the source.
    let listener_along = listener_velocity.dot(direction);
    // Positive when the source moves away from the listener.
    let source_along = source_velocity.dot(direction);
    let denominator = SPEED_OF_SOUND + source_along;
    if denominator.abs() < 1.0 {
        // At or beyond the speed of sound the formula breaks down; clamping keeps
        // the artefact audible instead of producing a NaN that silences the mix.
        return MAX_DOPPLER;
    }
    ((SPEED_OF_SOUND + listener_along) / denominator).clamp(MIN_DOPPLER, MAX_DOPPLER)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short buffer of constant amplitude, so gain and panning can be read off
    /// the output directly.
    fn tone(frames: usize, channels: u16, rate: u32, amplitude: f32) -> SharedBuffer {
        Arc::new(AudioBuffer {
            samples: vec![amplitude; frames * channels as usize],
            channels,
            sample_rate: rate,
        })
    }

    /// A ramp, for detecting resampling and loop-point errors, where a constant
    /// tone would look identical however wrongly it was read.
    fn ramp(frames: usize, rate: u32) -> SharedBuffer {
        Arc::new(AudioBuffer {
            samples: (0..frames).map(|i| i as f32 / frames as f32).collect(),
            channels: 1,
            sample_rate: rate,
        })
    }

    fn mix_once(mixer: &mut Mixer, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0; frames * OUTPUT_CHANNELS];
        mixer.mix(&mut out, frames);
        out
    }

    // --- attenuation ---

    #[test]
    fn no_attenuation_is_flat_at_every_distance() {
        let a = Attenuation::None;
        assert_eq!(a.gain(0.0), 1.0);
        assert_eq!(a.gain(1000.0), 1.0);
    }

    #[test]
    fn inverse_attenuation_falls_off_and_cuts_at_max_distance() {
        let a = Attenuation::Inverse {
            ref_distance: 1.0,
            max_distance: 10.0,
        };
        assert!((a.gain(0.5) - 1.0).abs() < 1e-6, "inside ref is full volume");
        assert!((a.gain(1.0) - 1.0).abs() < 1e-6);
        assert!((a.gain(2.0) - 0.5).abs() < 1e-6);
        assert!((a.gain(4.0) - 0.25).abs() < 1e-6);
        assert_eq!(a.gain(10.0), 0.0, "at max distance it must be silent");
        assert_eq!(a.gain(50.0), 0.0);
    }

    #[test]
    fn linear_attenuation_reaches_exactly_zero() {
        let a = Attenuation::Linear {
            ref_distance: 2.0,
            max_distance: 6.0,
        };
        assert_eq!(a.gain(1.0), 1.0);
        assert!((a.gain(4.0) - 0.5).abs() < 1e-6);
        assert_eq!(a.gain(6.0), 0.0);
    }

    /// Division by zero at the listener's exact position would produce infinity
    /// and then NaN through the whole mix.
    #[test]
    fn attenuation_at_zero_distance_is_finite() {
        for a in [
            Attenuation::default(),
            Attenuation::Linear {
                ref_distance: 0.0,
                max_distance: 10.0,
            },
            Attenuation::Inverse {
                ref_distance: 0.0,
                max_distance: 10.0,
            },
        ] {
            let g = a.gain(0.0);
            assert!(g.is_finite() && g <= 1.0, "gain at 0 distance was {g}");
        }
    }

    // --- panning ---

    /// A source to the listener's right must be louder in the right channel. A
    /// sign error here puts every sound on the wrong side, which is obvious in
    /// headphones and invisible in a code review.
    #[test]
    fn a_source_on_the_right_is_louder_on_the_right() {
        let mut mixer = Mixer::new(48000);
        // Default listener looks down -Z with +Y up, so its right is +X.
        mixer.play(
            Voice::new(tone(1000, 1, 48000, 0.5))
                .spatial(Spatial::at(Vec3::new(10.0, 0.0, 0.0))),
        );
        let out = mix_once(&mut mixer, 16);
        let (l, r) = (out[0].abs(), out[1].abs());
        assert!(r > l * 1.5, "expected right-dominant, got L {l} R {r}");
    }

    #[test]
    fn a_source_on_the_left_is_louder_on_the_left() {
        let mut mixer = Mixer::new(48000);
        mixer.play(
            Voice::new(tone(1000, 1, 48000, 0.5))
                .spatial(Spatial::at(Vec3::new(-10.0, 0.0, 0.0))),
        );
        let out = mix_once(&mut mixer, 16);
        let (l, r) = (out[0].abs(), out[1].abs());
        assert!(l > r * 1.5, "expected left-dominant, got L {l} R {r}");
    }

    #[test]
    fn a_source_straight_ahead_is_centred() {
        let mut mixer = Mixer::new(48000);
        mixer.play(
            Voice::new(tone(1000, 1, 48000, 0.5))
                .spatial(Spatial::at(Vec3::new(0.0, 0.0, -10.0))),
        );
        let out = mix_once(&mut mixer, 16);
        assert!(
            (out[0] - out[1]).abs() < 1e-5,
            "a source dead ahead must be centred: L {} R {}",
            out[0],
            out[1]
        );
    }

    /// The panning follows the *listener's* orientation, not the world axes. A
    /// mixer that used world +X as "right" would keep every sound on the same side
    /// as the player turned around.
    #[test]
    fn panning_follows_the_listener_orientation() {
        let source = Vec3::new(10.0, 0.0, 0.0);
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(1000, 1, 48000, 0.5)).spatial(Spatial::at(source)));
        let facing_away = mix_once(&mut mixer, 16);

        // Turn the listener 180 degrees: the same source is now on its left.
        let mut mixer = Mixer::new(48000);
        mixer.listener.forward = Vec3::new(0.0, 0.0, 1.0);
        mixer.play(Voice::new(tone(1000, 1, 48000, 0.5)).spatial(Spatial::at(source)));
        let turned = mix_once(&mut mixer, 16);

        assert!(
            facing_away[1] > facing_away[0],
            "before turning, the source is on the right"
        );
        assert!(
            turned[0] > turned[1],
            "after turning 180 degrees it must be on the left: L {} R {}",
            turned[0],
            turned[1]
        );
    }

    /// Constant-power panning: the total power must not dip as a source crosses
    /// centre, which is what linear panning does and what makes it sound wrong.
    #[test]
    fn panning_preserves_power_across_the_stereo_field() {
        let power_at = |x: f32| {
            let mut mixer = Mixer::new(48000);
            mixer.play(
                Voice::new(tone(1000, 1, 48000, 1.0)).spatial(Spatial::At {
                    position: Vec3::new(x, 0.0, 0.0),
                    velocity: Vec3::ZERO,
                    // Constant gain, so only the panning varies.
                    attenuation: Attenuation::None,
                }),
            );
            let out = mix_once(&mut mixer, 8);
            out[0] * out[0] + out[1] * out[1]
        };
        let hard_right = power_at(100.0);
        let centre = power_at(0.001);
        assert!(
            (hard_right - centre).abs() < hard_right * 0.05,
            "power varied with pan: hard {hard_right} vs centre {centre}"
        );
    }

    // --- distance ---

    #[test]
    fn a_distant_source_is_quieter_than_a_near_one() {
        let amplitude_at = |distance: f32| {
            let mut mixer = Mixer::new(48000);
            mixer.play(
                Voice::new(tone(1000, 1, 48000, 1.0))
                    .spatial(Spatial::at(Vec3::new(0.0, 0.0, -distance))),
            );
            let out = mix_once(&mut mixer, 8);
            out[0].abs() + out[1].abs()
        };
        let near = amplitude_at(1.0);
        let far = amplitude_at(20.0);
        assert!(far < near * 0.2, "near {near} vs far {far}");
        assert_eq!(amplitude_at(200.0), 0.0, "past max distance is silent");
    }

    #[test]
    fn an_unpositioned_voice_ignores_distance_entirely() {
        let mut mixer = Mixer::new(48000);
        mixer.listener.position = Vec3::new(1000.0, 0.0, 0.0);
        mixer.play(Voice::new(tone(1000, 1, 48000, 0.5)));
        let out = mix_once(&mut mixer, 8);
        assert!(out[0].abs() > 0.1, "music must not attenuate: {}", out[0]);
    }

    // --- doppler ---

    #[test]
    fn an_approaching_source_is_pitched_up_and_a_receding_one_down() {
        // Source ahead at -Z, moving toward the listener (+Z velocity).
        let to_source = Vec3::new(0.0, 0.0, -10.0);
        let approaching = doppler_ratio(to_source, 10.0, Vec3::new(0.0, 0.0, 30.0), Vec3::ZERO);
        let receding = doppler_ratio(to_source, 10.0, Vec3::new(0.0, 0.0, -30.0), Vec3::ZERO);
        assert!(approaching > 1.0, "approaching should raise pitch: {approaching}");
        assert!(receding < 1.0, "receding should lower pitch: {receding}");
    }

    /// Only motion *along the line* between source and listener shifts pitch. A
    /// source orbiting the listener has no Doppler at all, however fast — the
    /// property a naive speed-based formula gets wrong.
    #[test]
    fn tangential_motion_produces_no_doppler_shift() {
        let to_source = Vec3::new(0.0, 0.0, -10.0);
        // Moving along +X, perpendicular to the line of sight.
        let ratio = doppler_ratio(to_source, 10.0, Vec3::new(300.0, 0.0, 0.0), Vec3::ZERO);
        assert!(
            (ratio - 1.0).abs() < 1e-4,
            "perpendicular motion shifted the pitch by {ratio}"
        );
    }

    /// A supersonic source divides by zero or goes negative in the raw formula,
    /// producing a NaN that silences the entire mix.
    #[test]
    fn a_supersonic_source_is_clamped_rather_than_producing_nan() {
        let to_source = Vec3::new(0.0, 0.0, -10.0);
        for speed in [-SPEED_OF_SOUND, -400.0, 400.0, 10_000.0] {
            let ratio = doppler_ratio(to_source, 10.0, Vec3::new(0.0, 0.0, speed), Vec3::ZERO);
            assert!(ratio.is_finite(), "speed {speed} produced {ratio}");
            assert!(
                (MIN_DOPPLER..=MAX_DOPPLER).contains(&ratio),
                "speed {speed} produced an out-of-range ratio {ratio}"
            );
        }
    }

    // --- playback lifecycle ---

    #[test]
    fn a_one_shot_voice_retires_when_it_runs_out() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(8, 1, 48000, 0.5)));
        assert_eq!(mixer.voice_count(), 1);
        mix_once(&mut mixer, 32);
        assert_eq!(mixer.voice_count(), 0, "a finished voice must be dropped");
        assert_eq!(mixer.retired(), 1);
    }

    #[test]
    fn a_looping_voice_never_retires() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(8, 1, 48000, 0.5)).looping(true));
        for _ in 0..20 {
            mix_once(&mut mixer, 64);
        }
        assert_eq!(mixer.voice_count(), 1, "a loop must keep playing");
    }

    /// At a high playback speed the cursor overshoots the loop point by more than
    /// one frame. Resetting to zero instead of taking the modulo drops the
    /// remainder and drifts the loop a little every cycle.
    #[test]
    fn a_fast_loop_wraps_by_modulo_and_does_not_drift() {
        let mut mixer = Mixer::new(48000);
        // 7.3 rather than a round 7.5: at 7.5 the cursor's period against a
        // 10-frame buffer is exact (7.5 * 4 = 30 = 3 * 10), so it lands on 0.0
        // and a correct modulo is indistinguishable from a wrong reset.
        mixer.play(
            Voice::new(ramp(10, 48000))
                .looping(true)
                .with_speed(7.3),
        );
        // Enough frames for several wraps.
        mix_once(&mut mixer, 64);
        let voice = &mixer.voices[0];
        assert!(
            voice.cursor >= 0.0 && voice.cursor < 10.0,
            "cursor {} left the buffer",
            voice.cursor
        );
        // A modulo wrap leaves a fractional remainder; a reset-to-zero would land
        // exactly on an integer every time.
        assert!(
            voice.cursor.fract() > 1e-9,
            "cursor landed exactly on {}, which suggests a reset rather than a modulo",
            voice.cursor
        );
    }

    /// An inaudible voice must still advance, or a one-shot out of range would
    /// occupy a slot forever and a loop would resume mid-sample.
    #[test]
    fn an_inaudible_one_shot_still_finishes() {
        let mut mixer = Mixer::new(48000);
        mixer.play(
            Voice::new(tone(8, 1, 48000, 0.5))
                .spatial(Spatial::at(Vec3::new(0.0, 0.0, -10_000.0))),
        );
        mix_once(&mut mixer, 64);
        assert_eq!(
            mixer.voice_count(),
            0,
            "an out-of-range one-shot must still retire"
        );
    }

    #[test]
    fn a_zero_or_negative_speed_is_clamped_so_a_voice_cannot_stall() {
        let v = Voice::new(tone(8, 1, 48000, 0.5)).with_speed(0.0);
        assert!(v.speed > 0.0);
        let v = Voice::new(tone(8, 1, 48000, 0.5)).with_speed(-3.0);
        assert!(v.speed > 0.0);
    }

    #[test]
    fn seek_moves_the_cursor_and_is_clamped_to_the_buffer() {
        let mut v = Voice::new(tone(48_000, 1, 48000, 0.5));
        v.seek(0.5);
        assert!((v.position_seconds() - 0.5).abs() < 1e-4);
        v.seek(-10.0);
        assert_eq!(v.position_seconds(), 0.0, "a negative seek clamps to the start");
        v.seek(1000.0);
        assert!(
            v.position_seconds() <= 1.001,
            "a seek past the end clamps to it: {}",
            v.position_seconds()
        );
    }

    /// Streaming is the same path as one-shot playback, seeked into. This checks
    /// the property that makes that valid: seeking to the middle really does
    /// produce the middle of the buffer.
    #[test]
    fn seeking_into_a_long_buffer_plays_from_there() {
        let mut mixer = Mixer::new(48000);
        let mut voice = Voice::new(ramp(1000, 48000));
        voice.seek(500.0 / 48000.0);
        mixer.play(voice);
        let out = mix_once(&mut mixer, 4);
        // The ramp is i/1000, so frame 500 is ~0.5.
        assert!(
            (out[0] - 0.5).abs() < 0.02,
            "seeked playback gave {} instead of ~0.5",
            out[0]
        );
    }

    // --- mixing ---

    #[test]
    fn two_voices_sum() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(100, 1, 48000, 0.25)));
        mixer.play(Voice::new(tone(100, 1, 48000, 0.25)));
        let out = mix_once(&mut mixer, 4);
        let one = {
            let mut single = Mixer::new(48000);
            single.play(Voice::new(tone(100, 1, 48000, 0.25)));
            mix_once(&mut single, 4)[0]
        };
        assert!(
            (out[0] - one * 2.0).abs() < 1e-5,
            "two identical voices should double: {} vs {}",
            out[0],
            one * 2.0
        );
    }

    /// Output must clamp, not wrap. A wrapped overflow reads as a loud crack
    /// rather than as distortion.
    #[test]
    fn an_overloaded_mix_clamps_and_reports_its_peak() {
        let mut mixer = Mixer::new(48000);
        for _ in 0..8 {
            mixer.play(Voice::new(tone(100, 1, 48000, 1.0)));
        }
        let out = mix_once(&mut mixer, 4);
        assert!(
            out.iter().all(|s| (-1.0..=1.0).contains(s)),
            "output escaped [-1, 1]"
        );
        assert!(
            mixer.peak() > 1.0,
            "the mixer should report that it clipped, peak was {}",
            mixer.peak()
        );
    }

    /// `mix` overwrites rather than accumulating, so a caller cannot double a
    /// buffer by forgetting to clear it.
    #[test]
    fn mix_overwrites_the_output_buffer() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(100, 1, 48000, 0.25)).looping(true));
        let mut out = vec![99.0; 8];
        mixer.mix(&mut out, 4);
        assert!(
            out.iter().all(|s| s.abs() < 1.0),
            "the previous contents leaked through: {out:?}"
        );
    }

    /// An unpositioned voice must pass through at unity. The pan law's 0.707
    /// centre gain is right for spreading one mono signal across two speakers and
    /// wrong for a stereo track that already has two channels — it would play all
    /// music and UI 3 dB quieter than authored.
    #[test]
    fn an_unpositioned_voice_is_not_attenuated_by_the_pan_law() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(100, 1, 48000, 0.5)));
        let out = mix_once(&mut mixer, 4);
        assert!(
            (out[0] - 0.5).abs() < 1e-4,
            "a 0.5 sample should come out at 0.5, got {}",
            out[0]
        );
        assert!((out[1] - 0.5).abs() < 1e-4, "and in both channels: {}", out[1]);
    }

    /// An unpositioned *stereo* voice must keep its own image rather than being
    /// folded to mono. This is what separates music from a positioned effect.
    #[test]
    fn an_unpositioned_stereo_voice_keeps_its_channels_separate() {
        let buffer = Arc::new(AudioBuffer {
            // Frame after frame of (left = 0.5, right = -0.5).
            samples: (0..200)
                .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
                .collect(),
            channels: 2,
            sample_rate: 48000,
        });
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(buffer));
        let out = mix_once(&mut mixer, 4);
        assert!((out[0] - 0.5).abs() < 1e-4, "left = {}", out[0]);
        assert!((out[1] + 0.5).abs() < 1e-4, "right = {}", out[1]);
    }

    /// A positioned voice, by contrast, IS folded to mono and panned: it is one
    /// signal arriving from one direction, and keeping its authored stereo image
    /// would fight the spatialisation.
    #[test]
    fn a_positioned_stereo_voice_is_folded_and_panned() {
        let buffer = Arc::new(AudioBuffer {
            samples: (0..200)
                .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
                .collect(),
            channels: 2,
            sample_rate: 48000,
        });
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(buffer).spatial(Spatial::At {
            position: Vec3::new(10.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            attenuation: Attenuation::None,
        }));
        let out = mix_once(&mut mixer, 4);
        // L and R cancel when folded, so a hard-panned-in-the-file source becomes
        // silent once positioned — which is the correct consequence of folding.
        assert!(
            out[0].abs() < 1e-4 && out[1].abs() < 1e-4,
            "opposing channels should cancel when folded: L {} R {}",
            out[0],
            out[1]
        );
    }

    #[test]
    fn master_gain_scales_the_whole_mix() {
        let mut mixer = Mixer::new(48000);
        mixer.master_gain = 0.5;
        mixer.play(Voice::new(tone(100, 1, 48000, 0.5)));
        let loud = {
            let mut m = Mixer::new(48000);
            m.play(Voice::new(tone(100, 1, 48000, 0.5)));
            mix_once(&mut m, 4)[0]
        };
        let quiet = mix_once(&mut mixer, 4)[0];
        assert!((quiet - loud * 0.5).abs() < 1e-5, "{quiet} vs {loud}");
    }

    #[test]
    fn a_silent_mixer_produces_silence_not_noise() {
        let mut mixer = Mixer::new(48000);
        let out = mix_once(&mut mixer, 32);
        assert!(out.iter().all(|s| *s == 0.0));
        assert_eq!(mixer.peak(), 0.0);
    }

    /// Resampling must follow the ratio of the rates. A mixer that ignored the
    /// source rate would play a 22 kHz effect at double speed on a 44 kHz device.
    #[test]
    fn a_source_at_half_the_output_rate_advances_at_half_speed() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(ramp(1000, 24000)));
        mix_once(&mut mixer, 100);
        let cursor = mixer.voices[0].cursor;
        assert!(
            (cursor - 50.0).abs() < 1.0,
            "100 output frames at half rate should advance ~50 source frames, got {cursor}"
        );
    }

    // --- voice cap ---

    #[test]
    fn the_voice_cap_is_enforced() {
        let mut mixer = Mixer::new(48000);
        for _ in 0..MAX_VOICES {
            assert!(mixer.play(Voice::new(tone(1000, 1, 48000, 0.5)).looping(true)));
        }
        assert_eq!(mixer.voice_count(), MAX_VOICES);
        // One more at the same gain must be refused, not silently grow the list.
        assert!(!mixer.play(Voice::new(tone(1000, 1, 48000, 0.5)).looping(true)));
        assert_eq!(mixer.voice_count(), MAX_VOICES);
    }

    /// A burst of distant chatter must not be able to drown out the sound the
    /// player needs to hear, so a louder voice evicts the quietest.
    #[test]
    fn a_louder_voice_evicts_the_quietest_when_full() {
        let mut mixer = Mixer::new(48000);
        for _ in 0..MAX_VOICES {
            mixer.play(
                Voice::new(tone(1000, 1, 48000, 0.5))
                    .with_gain(0.1)
                    .looping(true),
            );
        }
        assert!(
            mixer.play(
                Voice::new(tone(1000, 1, 48000, 0.5))
                    .with_gain(1.0)
                    .looping(true)
            ),
            "a much louder voice should claim a slot"
        );
        assert_eq!(mixer.voice_count(), MAX_VOICES);
        let loudest = mixer
            .voices
            .iter()
            .map(|v| v.gain)
            .fold(0.0f32, f32::max);
        assert!((loudest - 1.0).abs() < 1e-6, "the loud voice is not in the mix");
    }

    /// `stop_looping` must leave one-shots alone: cutting them too would truncate
    /// whatever had just been triggered.
    #[test]
    fn stop_looping_spares_one_shots() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(48_000, 1, 48000, 0.5)).looping(true));
        mixer.play(Voice::new(tone(48_000, 1, 48000, 0.5)));
        assert_eq!(mixer.voice_count(), 2);
        mixer.stop_looping();
        assert_eq!(mixer.voice_count(), 1, "the one-shot should survive");
        assert!(!mixer.voices[0].looping);
    }

    #[test]
    fn set_spatial_all_looping_moves_loops_and_not_one_shots() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(48_000, 1, 48000, 0.5)).looping(true));
        mixer.play(Voice::new(tone(48_000, 1, 48000, 0.5)));
        let target = Spatial::at(Vec3::new(3.0, 0.0, 0.0));
        mixer.set_spatial_all_looping(target);
        assert_eq!(mixer.voices[0].spatial, target);
        assert_eq!(
            mixer.voices[1].spatial,
            Spatial::None,
            "a one-shot must keep its own placement"
        );
    }

    /// Steering a looping source really does move the stereo image — the property
    /// the orbiting demo source depends on.
    #[test]
    fn steering_a_looping_source_moves_the_stereo_image() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(48_000, 1, 48000, 0.8)).looping(true));

        mixer.set_spatial_all_looping(Spatial::at(Vec3::new(6.0, 0.0, 0.0)));
        let right = mix_once(&mut mixer, 8);
        mixer.set_spatial_all_looping(Spatial::at(Vec3::new(-6.0, 0.0, 0.0)));
        let left = mix_once(&mut mixer, 8);

        assert!(right[1] > right[0], "source at +X should favour the right channel");
        assert!(left[0] > left[1], "moved to -X it should favour the left");
    }

    #[test]
    fn stop_all_clears_every_voice() {
        let mut mixer = Mixer::new(48000);
        for _ in 0..4 {
            mixer.play(Voice::new(tone(1000, 1, 48000, 0.5)).looping(true));
        }
        mixer.stop_all();
        assert_eq!(mixer.voice_count(), 0);
    }

    #[test]
    fn a_degenerate_listener_orientation_gives_a_stable_right_vector() {
        // Looking straight up: forward and up are parallel, so the cross product
        // collapses. The answer must still be a usable, *consistent* vector.
        let listener = Listener {
            forward: Vec3::UNIT_Y,
            up: Vec3::UNIT_Y,
            ..Listener::default()
        };
        let a = listener.right();
        let b = listener.right();
        assert!(a.length() > 0.5, "right vector collapsed to {a}");
        assert_eq!(a, b, "it must be deterministic or the stereo image flips");
    }

    #[test]
    fn a_zero_length_output_request_is_a_no_op() {
        let mut mixer = Mixer::new(48000);
        mixer.play(Voice::new(tone(100, 1, 48000, 0.5)).looping(true));
        let mut out: Vec<f32> = Vec::new();
        mixer.mix(&mut out, 0);
        assert!(out.is_empty());
        assert_eq!(mixer.voice_count(), 1, "nothing should have been retired");
    }
}
