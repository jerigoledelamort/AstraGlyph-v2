// WAV (RIFF) decoder — self-implemented, per the project's "no external crates"
// rule.
//
// Scope: the PCM subset real files actually use. 8/16/24/32-bit integer and
// 32/64-bit float samples, any channel count, any sample rate, and both plain
// `WAVE_FORMAT_PCM` and `WAVE_FORMAT_EXTENSIBLE` containers. Compressed formats
// (ADPCM, mu-law, MP3-in-RIFF) are rejected with a message naming the tag rather
// than decoded as noise — a wrong-format file that plays as a burst of static is
// much harder to diagnose than one that refuses to load.
//
// Chunks are walked rather than assumed. The canonical layout is
// RIFF/fmt /data, but real files interleave `LIST`, `fact`, `cue ` and padding
// chunks freely, and a decoder that reads `data` at a fixed offset works on
// files exported by one tool and fails on another.

use crate::engine::core::{EngineError, Result};

/// Decoded PCM audio, normalized to interleaved `f32` in [-1, 1].
///
/// Normalizing at decode time rather than at mix time means the mixer has one
/// sample format to handle instead of six, and the conversion happens once per
/// file rather than once per frame.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioBuffer {
    /// Interleaved samples: frame 0 channel 0, frame 0 channel 1, frame 1...
    pub samples: Vec<f32>,
    /// Channels per frame.
    pub channels: u16,
    /// Frames per second.
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// An empty buffer with the given format.
    pub fn empty(channels: u16, sample_rate: u32) -> Self {
        Self {
            samples: Vec::new(),
            channels: channels.max(1),
            sample_rate: sample_rate.max(1),
        }
    }

    /// Frames (sample groups), not individual samples.
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frame_count() as f32 / self.sample_rate as f32
    }

    /// One channel's sample at `frame`, or 0.0 outside the buffer.
    ///
    /// Silence rather than a panic for out-of-range reads: a voice that has run
    /// past its end should go quiet, and making every mixer read a bounds check
    /// would put the check in the hot loop anyway.
    pub fn sample(&self, frame: usize, channel: u16) -> f32 {
        if channel >= self.channels {
            return 0.0;
        }
        let index = frame * self.channels as usize + channel as usize;
        self.samples.get(index).copied().unwrap_or(0.0)
    }

    /// Mono mixdown of `frame`: the average of its channels.
    ///
    /// Averaging rather than summing, so a stereo file does not come out twice as
    /// loud as a mono one — which would make 3D positioning inconsistent between
    /// sources depending on how they happened to be exported.
    pub fn mono_sample(&self, frame: usize) -> f32 {
        if self.channels == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for c in 0..self.channels {
            sum += self.sample(frame, c);
        }
        sum / self.channels as f32
    }
}

/// A byte cursor over the file, with bounds-checked reads.
///
/// Every read returns `Result` rather than panicking: the input is a file from
/// disk, so a truncated or malformed one is an expected condition, and an
/// out-of-bounds slice index on user data is a crash rather than an error message.
struct Cursor<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.remaining() < count {
            return Err(err(format!(
                "unexpected end of file: wanted {count} bytes at offset {}, {} remain",
                self.position,
                self.remaining()
            )));
        }
        let slice = &self.data[self.position..self.position + count];
        self.position += count;
        Ok(slice)
    }

    fn u16_le(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32_le(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn four_cc(&mut self) -> Result<[u8; 4]> {
        let b = self.take(4)?;
        Ok([b[0], b[1], b[2], b[3]])
    }

    fn skip(&mut self, count: usize) {
        self.position = (self.position + count).min(self.data.len());
    }
}

fn err(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidState(msg.into())
}

/// `WAVE_FORMAT_PCM`: integer samples.
const FORMAT_PCM: u16 = 0x0001;
/// `WAVE_FORMAT_IEEE_FLOAT`: floating-point samples.
const FORMAT_FLOAT: u16 = 0x0003;
/// `WAVE_FORMAT_EXTENSIBLE`: the real format tag lives in the extension block.
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// What the `fmt ` chunk described.
#[derive(Clone, Copy, Debug)]
struct Format {
    /// Resolved tag: `FORMAT_PCM` or `FORMAT_FLOAT`, never `FORMAT_EXTENSIBLE`.
    tag: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

/// Decode a WAV file from memory.
pub fn decode(bytes: &[u8]) -> Result<AudioBuffer> {
    let mut cursor = Cursor::new(bytes);

    if &cursor.four_cc()? != b"RIFF" {
        return Err(err("not a RIFF file (missing \"RIFF\" magic)"));
    }
    // Declared size, deliberately ignored: it is wrong in a surprising number of
    // real files (streamed exports leave it at 0 or 0xFFFFFFFF), and the chunk
    // walk below does not need it.
    let _declared_size = cursor.u32_le()?;
    if &cursor.four_cc()? != b"WAVE" {
        return Err(err("RIFF file is not a WAVE (missing \"WAVE\" form type)"));
    }

    let mut format: Option<Format> = None;
    let mut data: Option<&[u8]> = None;

    // Walk every chunk. Neither `fmt ` nor `data` is at a fixed offset in
    // practice, and a decoder that assumes otherwise works on files from one tool
    // and fails on another.
    while cursor.remaining() >= 8 {
        let id = cursor.four_cc()?;
        let size = cursor.u32_le()? as usize;
        // A chunk claiming more than the file holds is a truncated file. Clamp
        // rather than fail: the audio up to that point is still usable, which is
        // better than refusing a file that a player would happily play.
        let size = size.min(cursor.remaining());
        let body = cursor.take(size)?;

        match &id {
            b"fmt " => format = Some(parse_format(body)?),
            b"data" => data = Some(body),
            // Everything else — LIST, fact, cue , id3 — is metadata this decoder
            // has no use for.
            _ => {}
        }

        // Chunks are word-aligned: an odd size is followed by one pad byte that
        // is not counted in the size field. Missing this shifts every subsequent
        // chunk by one byte and turns the rest of the file into garbage.
        if size % 2 == 1 {
            cursor.skip(1);
        }
    }

    let format = format.ok_or_else(|| err("WAVE file has no \"fmt \" chunk"))?;
    let data = data.ok_or_else(|| err("WAVE file has no \"data\" chunk"))?;

    if format.channels == 0 {
        return Err(err("WAVE format declares 0 channels"));
    }
    if format.sample_rate == 0 {
        return Err(err("WAVE format declares a sample rate of 0"));
    }

    let samples = convert_samples(&format, data)?;
    Ok(AudioBuffer {
        samples,
        channels: format.channels,
        sample_rate: format.sample_rate,
    })
}

/// Read and validate a `fmt ` chunk.
fn parse_format(body: &[u8]) -> Result<Format> {
    let mut c = Cursor::new(body);
    let mut tag = c.u16_le()?;
    let channels = c.u16_le()?;
    let sample_rate = c.u32_le()?;
    let _byte_rate = c.u32_le()?;
    let _block_align = c.u16_le()?;
    let bits_per_sample = c.u16_le()?;

    if tag == FORMAT_EXTENSIBLE {
        // The extension block's `SubFormat` GUID carries the real tag in its
        // first two bytes. Without following it, every 24-bit and multichannel
        // file — which is what tools emit by default above 16 bits — would be
        // rejected as "unsupported format 65534".
        let _extension_size = c.u16_le()?;
        let _valid_bits = c.u16_le()?;
        let _channel_mask = c.u32_le()?;
        tag = c.u16_le()?;
    }

    match tag {
        FORMAT_PCM | FORMAT_FLOAT => {}
        other => {
            return Err(err(format!(
                "unsupported WAVE format tag 0x{other:04X} (only PCM 0x0001 and \
                 IEEE float 0x0003 are decoded)"
            )))
        }
    }

    match (tag, bits_per_sample) {
        (FORMAT_PCM, 8) | (FORMAT_PCM, 16) | (FORMAT_PCM, 24) | (FORMAT_PCM, 32) => {}
        (FORMAT_FLOAT, 32) | (FORMAT_FLOAT, 64) => {}
        (_, bits) => {
            return Err(err(format!(
                "unsupported sample width: {bits} bits for format tag 0x{tag:04X}"
            )))
        }
    }

    Ok(Format {
        tag,
        channels,
        sample_rate,
        bits_per_sample,
    })
}

/// Convert raw sample bytes into normalized `f32`.
fn convert_samples(format: &Format, data: &[u8]) -> Result<Vec<f32>> {
    let bytes_per_sample = (format.bits_per_sample / 8) as usize;
    if bytes_per_sample == 0 {
        return Err(err("sample width of 0 bits"));
    }
    // A trailing partial sample is dropped rather than read as a whole one.
    let count = data.len() / bytes_per_sample;
    let mut out = Vec::with_capacity(count);

    for i in 0..count {
        let b = &data[i * bytes_per_sample..(i + 1) * bytes_per_sample];
        let value = match (format.tag, format.bits_per_sample) {
            // 8-bit PCM is *unsigned*, centred on 128 — the one integer width in
            // WAV that is. Treating it as signed inverts every sample and turns
            // the file into a loud buzz.
            (FORMAT_PCM, 8) => (b[0] as f32 - 128.0) / 128.0,
            (FORMAT_PCM, 16) => {
                let v = i16::from_le_bytes([b[0], b[1]]);
                // 32768 rather than 32767: the range is asymmetric, and dividing
                // by the positive maximum lets the most negative sample exceed
                // -1.0 and clip.
                v as f32 / 32768.0
            }
            (FORMAT_PCM, 24) => {
                // Sign-extend a 24-bit two's-complement value into i32 by placing
                // it in the high bytes and shifting back down arithmetically.
                let v = i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8;
                v as f32 / 8_388_608.0
            }
            (FORMAT_PCM, 32) => {
                let v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                v as f32 / 2_147_483_648.0
            }
            (FORMAT_FLOAT, 32) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (FORMAT_FLOAT, 64) => f64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]) as f32,
            _ => return Err(err("unsupported sample format reached the converter")),
        };
        // A NaN or infinity in a float WAV would propagate through the whole mix
        // and silence everything; clamp it to something audible-but-wrong instead.
        out.push(if value.is_finite() { value } else { 0.0 });
    }
    Ok(out)
}

/// Read and decode a WAV file from disk.
pub fn load(path: impl AsRef<std::path::Path>) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    decode(&bytes).map_err(|e| err(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid WAV in memory. Used instead of a fixture file so the
    /// tests state the byte layout they depend on explicitly.
    fn build_wav(
        tag: u16,
        channels: u16,
        sample_rate: u32,
        bits: u16,
        sample_bytes: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        let fmt_size = 16u32;
        let data_size = sample_bytes.len() as u32;
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(4 + 8 + fmt_size + 8 + data_size).to_le_bytes());
        out.extend_from_slice(b"WAVE");

        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&fmt_size.to_le_bytes());
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        let block_align = channels * bits / 8;
        out.extend_from_slice(&(sample_rate * block_align as u32).to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());

        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_size.to_le_bytes());
        out.extend_from_slice(sample_bytes);
        out
    }

    fn i16_bytes(values: &[i16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn decodes_16_bit_mono() {
        let wav = build_wav(FORMAT_PCM, 1, 44100, 16, &i16_bytes(&[0, 16384, -16384, 32767]));
        let buf = decode(&wav).expect("should decode");
        assert_eq!(buf.channels, 1);
        assert_eq!(buf.sample_rate, 44100);
        assert_eq!(buf.frame_count(), 4);
        assert!((buf.samples[0] - 0.0).abs() < 1e-6);
        assert!((buf.samples[1] - 0.5).abs() < 1e-4, "{}", buf.samples[1]);
        assert!((buf.samples[2] + 0.5).abs() < 1e-4, "{}", buf.samples[2]);
    }

    /// 16-bit PCM is asymmetric: -32768..32767. Dividing by 32767 lets the most
    /// negative sample come out below -1.0 and clip on playback.
    #[test]
    fn the_most_negative_16_bit_sample_does_not_exceed_minus_one() {
        let wav = build_wav(FORMAT_PCM, 1, 8000, 16, &i16_bytes(&[i16::MIN, i16::MAX]));
        let buf = decode(&wav).unwrap();
        assert!(
            buf.samples[0] >= -1.0,
            "i16::MIN normalized to {} which is below -1",
            buf.samples[0]
        );
        assert!(buf.samples[1] <= 1.0);
    }

    /// 8-bit WAV is the one integer width that is *unsigned*. Reading it as signed
    /// inverts every sample and turns the file into a buzz.
    #[test]
    fn eight_bit_pcm_is_unsigned_and_centred_on_128() {
        let wav = build_wav(FORMAT_PCM, 1, 8000, 8, &[128, 255, 0, 192]);
        let buf = decode(&wav).unwrap();
        assert!(buf.samples[0].abs() < 1e-6, "128 is silence, got {}", buf.samples[0]);
        assert!(buf.samples[1] > 0.9, "255 is near +1, got {}", buf.samples[1]);
        assert!(buf.samples[2] < -0.9, "0 is -1, got {}", buf.samples[2]);
        assert!(buf.samples[3] > 0.4, "192 is +0.5, got {}", buf.samples[3]);
    }

    /// 24-bit samples must be sign-extended. Reading them as unsigned makes every
    /// negative sample a large positive one.
    #[test]
    fn twenty_four_bit_samples_are_sign_extended() {
        // -1 as 24-bit two's complement is 0xFFFFFF, little-endian FF FF FF.
        // +0x400000 is a quarter of full scale.
        let bytes = vec![0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x40];
        let wav = build_wav(FORMAT_PCM, 1, 48000, 24, &bytes);
        let buf = decode(&wav).unwrap();
        assert!(
            buf.samples[0] < 0.0 && buf.samples[0] > -1e-5,
            "0xFFFFFF should be a tiny negative value, got {}",
            buf.samples[0]
        );
        assert!(
            (buf.samples[1] - 0.5).abs() < 1e-4,
            "0x400000 should be 0.5, got {}",
            buf.samples[1]
        );
    }

    #[test]
    fn decodes_32_bit_float() {
        let bytes: Vec<u8> = [0.0f32, 0.75, -0.75]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let wav = build_wav(FORMAT_FLOAT, 1, 48000, 32, &bytes);
        let buf = decode(&wav).unwrap();
        assert!((buf.samples[1] - 0.75).abs() < 1e-6);
        assert!((buf.samples[2] + 0.75).abs() < 1e-6);
    }

    /// A NaN in a float WAV would propagate through the mix and silence
    /// everything downstream of it.
    #[test]
    fn non_finite_float_samples_become_silence() {
        let bytes: Vec<u8> = [f32::NAN, f32::INFINITY, 0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let wav = build_wav(FORMAT_FLOAT, 1, 48000, 32, &bytes);
        let buf = decode(&wav).unwrap();
        assert_eq!(buf.samples[0], 0.0);
        assert_eq!(buf.samples[1], 0.0);
        assert!((buf.samples[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn stereo_frames_interleave_and_count_correctly() {
        // Two frames of (left, right).
        let wav = build_wav(FORMAT_PCM, 2, 44100, 16, &i16_bytes(&[16384, -16384, 0, 32767]));
        let buf = decode(&wav).unwrap();
        assert_eq!(buf.channels, 2);
        assert_eq!(buf.frame_count(), 2);
        assert!((buf.sample(0, 0) - 0.5).abs() < 1e-4);
        assert!((buf.sample(0, 1) + 0.5).abs() < 1e-4);
        // Averaging, not summing: a hard-panned frame must not be louder than a
        // centred one.
        assert!(buf.mono_sample(0).abs() < 1e-4, "L and R cancel: {}", buf.mono_sample(0));
    }

    /// Chunks are word-aligned, and an odd-sized chunk is followed by a pad byte
    /// that the size field does not count. Missing it shifts every later chunk by
    /// one and turns the rest of the file into garbage.
    #[test]
    fn an_odd_sized_chunk_before_data_is_padded_correctly() {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&0u32.to_le_bytes()); // deliberately wrong, ignored
        wav.extend_from_slice(b"WAVE");
        // An odd-length metadata chunk plus its pad byte.
        wav.extend_from_slice(b"LIST");
        wav.extend_from_slice(&3u32.to_le_bytes());
        wav.extend_from_slice(&[1, 2, 3]);
        wav.push(0); // pad
        // fmt
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&FORMAT_PCM.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&8000u32.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        // data
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&4u32.to_le_bytes());
        wav.extend_from_slice(&i16_bytes(&[16384, -16384]));

        let buf = decode(&wav).expect("padding must be handled");
        assert_eq!(buf.frame_count(), 2);
        assert!((buf.samples[0] - 0.5).abs() < 1e-4);
    }

    /// `WAVE_FORMAT_EXTENSIBLE` is what tools emit by default above 16 bits.
    /// Rejecting it would refuse most 24-bit files.
    #[test]
    fn extensible_format_resolves_to_its_subformat() {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&0u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&40u32.to_le_bytes());
        wav.extend_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&48000u32.to_le_bytes());
        wav.extend_from_slice(&96000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        // Extension: size, valid bits, channel mask, then the SubFormat GUID
        // whose first two bytes carry the real tag.
        wav.extend_from_slice(&22u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(&0x3u32.to_le_bytes());
        wav.extend_from_slice(&FORMAT_PCM.to_le_bytes());
        wav.extend_from_slice(&[0u8; 14]); // rest of the GUID
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&2u32.to_le_bytes());
        wav.extend_from_slice(&i16_bytes(&[16384]));

        let buf = decode(&wav).expect("extensible PCM must decode");
        assert_eq!(buf.sample_rate, 48000);
        assert!((buf.samples[0] - 0.5).abs() < 1e-4);
    }

    /// A compressed file must say so rather than decode as static.
    #[test]
    fn a_compressed_format_is_rejected_with_its_tag() {
        // 0x0011 is IMA ADPCM.
        let wav = build_wav(0x0011, 1, 22050, 4, &[0, 1, 2, 3]);
        let e = decode(&wav).expect_err("ADPCM must be rejected");
        let message = e.to_string();
        assert!(
            message.contains("0x0011"),
            "the error should name the format tag: {message}"
        );
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        let full = build_wav(FORMAT_PCM, 1, 44100, 16, &i16_bytes(&[1, 2, 3, 4]));
        // Every prefix must either decode or error — never panic.
        for len in 0..full.len() {
            let _ = decode(&full[..len]);
        }
        // A file cut inside the header must be rejected outright.
        assert!(decode(&full[..8]).is_err());
    }

    #[test]
    fn a_non_riff_file_is_rejected() {
        assert!(decode(b"not a wav file at all").is_err());
        // Right magic, wrong form type.
        let mut fake = Vec::new();
        fake.extend_from_slice(b"RIFF");
        fake.extend_from_slice(&4u32.to_le_bytes());
        fake.extend_from_slice(b"AVI ");
        assert!(decode(&fake).is_err());
    }

    #[test]
    fn a_missing_data_chunk_is_an_error() {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&0u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&FORMAT_PCM.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&8000u32.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        assert!(decode(&wav).is_err());
    }

    #[test]
    fn zero_channels_or_sample_rate_is_rejected() {
        let wav = build_wav(FORMAT_PCM, 0, 44100, 16, &i16_bytes(&[1, 2]));
        assert!(decode(&wav).is_err(), "0 channels must be an error");
        let wav = build_wav(FORMAT_PCM, 1, 0, 16, &i16_bytes(&[1, 2]));
        assert!(decode(&wav).is_err(), "a 0 sample rate must be an error");
    }

    /// A chunk header claiming more bytes than the file holds is a truncated file.
    /// The audio up to that point is still usable, which beats refusing a file a
    /// player would happily play.
    #[test]
    fn an_oversized_data_chunk_is_clamped_to_what_exists() {
        let mut wav = build_wav(FORMAT_PCM, 1, 8000, 16, &i16_bytes(&[16384, -16384]));
        // Rewrite the data chunk's size to claim far more than is present.
        let data_pos = wav.len() - 4 - 4;
        wav[data_pos..data_pos + 4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let buf = decode(&wav).expect("a truncated data chunk should still decode");
        assert_eq!(buf.frame_count(), 2);
    }

    #[test]
    fn duration_follows_the_frame_count_and_rate() {
        let wav = build_wav(FORMAT_PCM, 2, 100, 16, &i16_bytes(&[0; 200]));
        let buf = decode(&wav).unwrap();
        assert_eq!(buf.frame_count(), 100);
        assert!((buf.duration() - 1.0).abs() < 1e-6, "{}", buf.duration());
    }

    #[test]
    fn out_of_range_reads_are_silence_not_panics() {
        let buf = AudioBuffer::empty(2, 44100);
        assert_eq!(buf.sample(0, 0), 0.0);
        assert_eq!(buf.sample(1000, 1), 0.0);
        assert_eq!(buf.sample(0, 99), 0.0);
        assert_eq!(buf.mono_sample(0), 0.0);
        assert_eq!(buf.duration(), 0.0);
    }
}
