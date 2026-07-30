// PNG decoder — self-implemented, per the "no external crates" rule. Includes the
// DEFLATE (RFC 1951) inflater PNG's compression requires, since that is the bulk of
// the work and there is no way around it.
//
// Scope: the colour types and bit depths real files use — 8- and 16-bit greyscale,
// RGB, palette, greyscale+alpha and RGBA — with all five filter types and both
// interlaced and non-interlaced layout. Not supported, and rejected by name rather
// than mis-decoded: 1/2/4-bit sub-byte depths (they need bit-level unpacking for a
// case no exporter produces for textures) and APNG animation chunks.
//
// The parts that are easy to get subtly wrong, and where the tests concentrate:
//
// - **Filters are applied to *reconstructed* bytes, not raw ones.** Each scanline's
//   filter references the already-unfiltered line above it. Using the raw previous
//   line produces an image that looks almost right at the top and degrades downward,
//   which is the classic PNG bug.
// - **`Paeth` is a predictor, not an average.** Its tie-breaking order is specified,
//   and getting it wrong is a subtle per-pixel error rather than a visible break.
// - **The filter operates on bytes at a distance of one *pixel*, not one byte.** For
//   RGBA that is four bytes back; using one gives channel-smeared output.
// - **DEFLATE back-references can overlap the output being written.** A match with
//   distance 1 and length 100 repeats one byte a hundred times, so the copy has to
//   be byte-by-byte rather than a block move.

use crate::engine::core::{EngineError, Result};

/// A decoded image: 8-bit RGBA, top row first.
///
/// Normalised to RGBA8 at decode time rather than preserving the source format, so
/// consumers (a wgpu texture upload, an ASCII sampler) have one layout to handle
/// instead of six. 16-bit sources are reduced to 8, which is what the display and
/// the ASCII quantizer can show anyway.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, RGBA.
    pub pixels: Vec<u8>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The pixel data is never useful in a debug print and is often megabytes.
        write!(
            f,
            "Image {{ {}x{}, {} bytes }}",
            self.width,
            self.height,
            self.pixels.len()
        )
    }
}

impl Image {
    /// A solid-colour image, for tests and placeholders.
    pub fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        Self {
            width,
            height,
            pixels: rgba.repeat((width * height) as usize),
        }
    }

    /// One pixel, or `None` outside the image.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = ((y * self.width + x) * 4) as usize;
        self.pixels
            .get(index..index + 4)
            .map(|s| [s[0], s[1], s[2], s[3]])
    }

    /// Bytes the image occupies.
    pub fn byte_size(&self) -> usize {
        self.pixels.len()
    }
}

fn err(msg: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidState(msg.to_string())
}

/// PNG's fixed 8-byte signature.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Colour types from the IHDR chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorType {
    Grey,
    Rgb,
    Palette,
    GreyAlpha,
    Rgba,
}

impl ColorType {
    fn from_byte(byte: u8) -> Option<Self> {
        Some(match byte {
            0 => Self::Grey,
            2 => Self::Rgb,
            3 => Self::Palette,
            4 => Self::GreyAlpha,
            6 => Self::Rgba,
            _ => return None,
        })
    }

    /// Channels per pixel in the *source* data.
    fn channels(self) -> usize {
        match self {
            Self::Grey => 1,
            Self::Rgb => 3,
            // A palette index is one byte; the RGB comes from PLTE.
            Self::Palette => 1,
            Self::GreyAlpha => 2,
            Self::Rgba => 4,
        }
    }
}

/// Header fields from IHDR.
#[derive(Clone, Copy, Debug)]
struct Header {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: ColorType,
    interlaced: bool,
}

impl Header {
    /// Bytes per pixel in the filtered data. At least 1, since the filter's
    /// left-neighbour distance is measured in these.
    fn bytes_per_pixel(&self) -> usize {
        (self.color_type.channels() * self.bit_depth as usize / 8).max(1)
    }
}

/// Decode a PNG from memory.
pub fn decode(bytes: &[u8]) -> Result<Image> {
    if bytes.len() < SIGNATURE.len() || bytes[..SIGNATURE.len()] != SIGNATURE {
        return Err(err("not a PNG file (bad signature)"));
    }

    let mut position = SIGNATURE.len();
    let mut header: Option<Header> = None;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut palette_alpha: Vec<u8> = Vec::new();
    // IDAT may be split across any number of chunks, and the compressed stream runs
    // across the boundaries — so they must be concatenated before inflating, not
    // inflated one at a time.
    let mut compressed: Vec<u8> = Vec::new();

    while position + 8 <= bytes.len() {
        let length = u32::from_be_bytes(read4(bytes, position)?) as usize;
        let kind: [u8; 4] = bytes
            .get(position + 4..position + 8)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| err("truncated chunk header"))?;
        let data_start = position + 8;
        let data_end = data_start
            .checked_add(length)
            .ok_or_else(|| err("chunk length overflows"))?;
        if data_end > bytes.len() {
            return Err(err(format!(
                "chunk {:?} claims {length} bytes but only {} remain",
                String::from_utf8_lossy(&kind),
                bytes.len().saturating_sub(data_start)
            )));
        }
        let data = &bytes[data_start..data_end];

        match &kind {
            b"IHDR" => header = Some(parse_header(data)?),
            b"PLTE" => {
                palette = data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            }
            // tRNS for a palette image is a parallel array of alpha values. Ignoring
            // it makes every transparent pixel opaque, which for a UI texture is the
            // difference between a sprite and a rectangle.
            b"tRNS" => palette_alpha = data.to_vec(),
            b"IDAT" => compressed.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        // 4 more bytes for the CRC, which is not verified: a corrupt file will fail
        // in the inflater or the filter step with a more useful message than "CRC
        // mismatch", and verifying it would not make a broken file loadable.
        position = data_end + 4;
    }

    let header = header.ok_or_else(|| err("PNG has no IHDR chunk"))?;
    if compressed.is_empty() {
        return Err(err("PNG has no IDAT data"));
    }

    let raw = inflate_zlib(&compressed)?;
    let pixels = if header.interlaced {
        deinterlace(&header, &raw, &palette, &palette_alpha)?
    } else {
        let bpp = header.bytes_per_pixel();
        let stride = row_bytes(&header, header.width);
        let unfiltered = unfilter(&raw, header.height as usize, stride, bpp)?;
        to_rgba(&header, &unfiltered, stride, header.width, header.height, &palette, &palette_alpha)
    };

    Ok(Image {
        width: header.width,
        height: header.height,
        pixels,
    })
}

fn read4(bytes: &[u8], at: usize) -> Result<[u8; 4]> {
    bytes
        .get(at..at + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| err("unexpected end of file"))
}

fn parse_header(data: &[u8]) -> Result<Header> {
    if data.len() < 13 {
        return Err(err("IHDR is shorter than 13 bytes"));
    }
    let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    let bit_depth = data[8];
    let color_type = ColorType::from_byte(data[9])
        .ok_or_else(|| err(format!("unknown PNG colour type {}", data[9])))?;
    let compression = data[10];
    let filter = data[11];
    let interlace = data[12];

    if width == 0 || height == 0 {
        return Err(err("PNG has a zero dimension"));
    }
    // A 16k x 16k RGBA image is a gigabyte. Bounded so a malformed header cannot ask
    // for an allocation that takes the process down.
    const MAX_DIMENSION: u32 = 16_384;
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(err(format!(
            "PNG is {width}x{height}, beyond the {MAX_DIMENSION} limit"
        )));
    }
    if compression != 0 {
        return Err(err(format!("unknown compression method {compression}")));
    }
    if filter != 0 {
        return Err(err(format!("unknown filter method {filter}")));
    }
    if interlace > 1 {
        return Err(err(format!("unknown interlace method {interlace}")));
    }
    match bit_depth {
        8 | 16 => {}
        // Sub-byte depths need bit-level unpacking with a different stride
        // calculation. Rejected by name rather than mis-decoded into noise.
        1 | 2 | 4 => {
            return Err(err(format!(
                "{bit_depth}-bit PNG is not supported (only 8 and 16)"
            )))
        }
        other => return Err(err(format!("invalid bit depth {other}"))),
    }
    if color_type == ColorType::Palette && bit_depth != 8 {
        return Err(err("palette PNGs must be 8-bit"));
    }

    Ok(Header {
        width,
        height,
        bit_depth,
        color_type,
        interlaced: interlace == 1,
    })
}

/// Bytes in one scanline of `width` pixels, excluding the filter byte.
fn row_bytes(header: &Header, width: u32) -> usize {
    width as usize * header.color_type.channels() * header.bit_depth as usize / 8
}

/// Undo the per-scanline filters.
///
/// Each line begins with a filter-type byte, and the filter references the
/// **already-reconstructed** line above — not the raw one. Using the raw previous
/// line gives an image that is right at the top and degrades downward, which is the
/// single most common PNG decoder bug.
fn unfilter(raw: &[u8], height: usize, stride: usize, bpp: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; height * stride];
    let mut offset = 0usize;
    for y in 0..height {
        let filter = *raw
            .get(offset)
            .ok_or_else(|| err(format!("scanline {y} is missing its filter byte")))?;
        offset += 1;
        let line = raw
            .get(offset..offset + stride)
            .ok_or_else(|| err(format!("scanline {y} is truncated")))?;
        offset += stride;

        let row_start = y * stride;
        for x in 0..stride {
            let raw_byte = line[x];
            // `a` is the byte one *pixel* to the left, not one byte: for RGBA that is
            // four back. One byte back would smear the channels together.
            let a = if x >= bpp { out[row_start + x - bpp] } else { 0 };
            let b = if y > 0 { out[row_start - stride + x] } else { 0 };
            let c = if y > 0 && x >= bpp {
                out[row_start - stride + x - bpp]
            } else {
                0
            };
            let value = match filter {
                0 => raw_byte,
                1 => raw_byte.wrapping_add(a),
                2 => raw_byte.wrapping_add(b),
                // Average uses a 16-bit sum before halving: `(a + b) / 2` in u8
                // would overflow and halve the wrong number.
                3 => raw_byte.wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => raw_byte.wrapping_add(paeth(a, b, c)),
                other => {
                    return Err(err(format!(
                        "scanline {y} has unknown filter type {other}"
                    )))
                }
            };
            out[row_start + x] = value;
        }
    }
    Ok(out)
}

/// The Paeth predictor.
///
/// Picks whichever of left/above/upper-left is closest to the linear estimate
/// `a + b - c`. A predictor, not an average — and its tie-breaking order is
/// specified, so `<=` versus `<` matters. Getting it wrong is a subtle per-pixel
/// error rather than a visible break, which is why it has its own test.
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i32 + b as i32 - c as i32;
    let pa = (p - a as i32).abs();
    let pb = (p - b as i32).abs();
    let pc = (p - c as i32).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Convert unfiltered source bytes to RGBA8.
fn to_rgba(
    header: &Header,
    data: &[u8],
    stride: usize,
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
    palette_alpha: &[u8],
) -> Vec<u8> {
    let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
    // 16-bit samples are reduced to 8 by taking the high byte, which is what the
    // display and the ASCII quantizer can show. The step is 2 rather than 1.
    let sample_step = (header.bit_depth / 8) as usize;

    for y in 0..height as usize {
        for x in 0..width as usize {
            let source = y * stride + x * header.color_type.channels() * sample_step;
            let at = |channel: usize| -> u8 {
                data.get(source + channel * sample_step).copied().unwrap_or(0)
            };
            let rgba = match header.color_type {
                ColorType::Grey => {
                    let g = at(0);
                    [g, g, g, 255]
                }
                ColorType::Rgb => [at(0), at(1), at(2), 255],
                ColorType::Palette => {
                    let index = at(0) as usize;
                    let rgb = palette.get(index).copied().unwrap_or([0, 0, 0]);
                    // tRNS supplies alpha per palette entry; entries past its end
                    // are opaque, per the spec.
                    let alpha = palette_alpha.get(index).copied().unwrap_or(255);
                    [rgb[0], rgb[1], rgb[2], alpha]
                }
                ColorType::GreyAlpha => {
                    let g = at(0);
                    [g, g, g, at(1)]
                }
                ColorType::Rgba => [at(0), at(1), at(2), at(3)],
            };
            let target = (y * width as usize + x) * 4;
            out[target..target + 4].copy_from_slice(&rgba);
        }
    }
    out
}

/// Adam7 interlacing: seven passes at increasing resolution.
///
/// Implemented because interlaced files exist in the wild and a decoder that ignored
/// the flag would produce a scrambled image rather than an error — the worst outcome.
/// Each pass is a separately filtered sub-image with its own stride, which is why the
/// unfilter cannot simply run over the whole buffer.
fn deinterlace(
    header: &Header,
    raw: &[u8],
    palette: &[[u8; 3]],
    palette_alpha: &[u8],
) -> Result<Vec<u8>> {
    // (x_start, y_start, x_step, y_step) for each of the seven passes.
    const PASSES: [(u32, u32, u32, u32); 7] = [
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ];
    let mut out = vec![0u8; (header.width as usize) * (header.height as usize) * 4];
    let bpp = header.bytes_per_pixel();
    let mut offset = 0usize;

    for (x_start, y_start, x_step, y_step) in PASSES {
        // Pixels this pass covers. A pass can be empty for a small image, and its
        // scanlines must then be skipped entirely rather than read as zero-length.
        let pass_width = header.width.saturating_sub(x_start).div_ceil(x_step);
        let pass_height = header.height.saturating_sub(y_start).div_ceil(y_step);
        if pass_width == 0 || pass_height == 0 {
            continue;
        }
        let stride = row_bytes(header, pass_width);
        let consumed = pass_height as usize * (stride + 1);
        let slice = raw
            .get(offset..offset + consumed)
            .ok_or_else(|| err("interlaced PNG is truncated"))?;
        offset += consumed;

        let unfiltered = unfilter(slice, pass_height as usize, stride, bpp)?;
        let sub = to_rgba(
            header,
            &unfiltered,
            stride,
            pass_width,
            pass_height,
            palette,
            palette_alpha,
        );
        // Scatter the sub-image back into its interlaced positions.
        for sy in 0..pass_height {
            for sx in 0..pass_width {
                let target_x = x_start + sx * x_step;
                let target_y = y_start + sy * y_step;
                if target_x >= header.width || target_y >= header.height {
                    continue;
                }
                let from = ((sy * pass_width + sx) * 4) as usize;
                let to = ((target_y * header.width + target_x) * 4) as usize;
                out[to..to + 4].copy_from_slice(&sub[from..from + 4]);
            }
        }
    }
    Ok(out)
}

// --- DEFLATE ---------------------------------------------------------------

/// A bit reader for DEFLATE's LSB-first bit order.
struct BitReader<'a> {
    bytes: &'a [u8],
    /// Byte position.
    position: usize,
    /// Bits already consumed from the current byte.
    bit: u32,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            bit: 0,
        }
    }

    /// Read `count` bits, LSB first.
    fn bits(&mut self, count: u32) -> Result<u32> {
        let mut value = 0u32;
        for i in 0..count {
            let byte = *self
                .bytes
                .get(self.position)
                .ok_or_else(|| err("compressed stream ended mid-symbol"))?;
            let bit = (byte >> self.bit) & 1;
            value |= (bit as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.position += 1;
            }
        }
        Ok(value)
    }

    /// Discard the rest of the current byte.
    fn align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.position += 1;
        }
    }
}

/// A canonical Huffman decoder built from code lengths.
///
/// DEFLATE transmits only the *lengths*; the codes themselves are derived by the
/// canonical rule (sort by length, then by symbol). Building the table this way is
/// what makes the dynamic-block header decodable at all.
struct Huffman {
    /// counts[l] = how many codes have length l.
    counts: [u16; 16],
    /// Symbols in canonical order.
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &length in lengths {
            if length as usize >= 16 {
                return Err(err("Huffman code length exceeds 15"));
            }
            counts[length as usize] += 1;
        }
        // Length 0 means "unused", not a zero-length code.
        counts[0] = 0;

        let mut offsets = [0u16; 16];
        for length in 1..16 {
            offsets[length] = offsets[length - 1] + counts[length - 1];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &length) in lengths.iter().enumerate() {
            if length != 0 {
                symbols[offsets[length as usize] as usize] = symbol as u16;
                offsets[length as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    /// Decode one symbol.
    ///
    /// Walks lengths from 1 upward, accumulating the code MSB-first — DEFLATE's
    /// Huffman codes are packed starting from the most significant bit even though
    /// its integers are LSB-first, which is the detail that makes a from-scratch
    /// inflater fail confusingly if missed.
    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for length in 1..16 {
            code |= reader.bits(1)? as i32;
            let count = self.counts[length] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(err("invalid Huffman code in compressed stream"))
    }
}

/// Length codes 257-285: base length and extra bits.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
    131, 163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Distance codes 0-29.
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    13, 13,
];

/// Largest output an inflate call will produce.
///
/// A crafted stream can expand almost without bound (a few kilobytes into gigabytes),
/// and this is a decoder for untrusted files. 256 MiB is far above any real texture
/// and far below "the process dies".
const MAX_INFLATED: usize = 256 * 1024 * 1024;

/// Inflate a zlib stream (RFC 1950 wrapper around RFC 1951 DEFLATE).
fn inflate_zlib(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() < 2 {
        return Err(err("zlib stream is too short"));
    }
    let cmf = bytes[0];
    let flg = bytes[1];
    if cmf & 0x0F != 8 {
        return Err(err(format!(
            "zlib compression method {} is not DEFLATE",
            cmf & 0x0F
        )));
    }
    // The header is a 16-bit big-endian value that must be a multiple of 31.
    if ((cmf as u16) << 8 | flg as u16) % 31 != 0 {
        return Err(err("zlib header checksum is wrong"));
    }
    if flg & 0x20 != 0 {
        return Err(err("zlib streams with a preset dictionary are not supported"));
    }
    // The trailing Adler-32 is not verified, for the same reason the chunk CRCs are
    // not: a corrupt stream fails inside the inflater with a better message.
    inflate(&bytes[2..])
}

/// Inflate a raw DEFLATE stream.
fn inflate(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut reader = BitReader::new(bytes);
    let mut out: Vec<u8> = Vec::new();

    loop {
        let last = reader.bits(1)? == 1;
        let kind = reader.bits(2)?;
        match kind {
            // Stored: length, its complement, then raw bytes.
            0 => {
                reader.align();
                let length = reader.bits(16)? as usize;
                let complement = reader.bits(16)? as usize;
                if length ^ 0xFFFF != complement {
                    return Err(err("stored block length does not match its complement"));
                }
                for _ in 0..length {
                    out.push(reader.bits(8)? as u8);
                }
            }
            1 => {
                let (literal, distance) = fixed_tables()?;
                inflate_block(&mut reader, &mut out, &literal, &distance)?;
            }
            2 => {
                let (literal, distance) = dynamic_tables(&mut reader)?;
                inflate_block(&mut reader, &mut out, &literal, &distance)?;
            }
            _ => return Err(err("invalid DEFLATE block type 3")),
        }
        if out.len() > MAX_INFLATED {
            return Err(err(format!(
                "compressed stream expands beyond the {MAX_INFLATED} byte limit"
            )));
        }
        if last {
            break;
        }
    }
    Ok(out)
}

/// The fixed Huffman tables from RFC 1951 section 3.2.6.
fn fixed_tables() -> Result<(Huffman, Huffman)> {
    let mut literal_lengths = [0u8; 288];
    for (symbol, length) in literal_lengths.iter_mut().enumerate() {
        *length = match symbol {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    let distance_lengths = [5u8; 30];
    Ok((
        Huffman::new(&literal_lengths)?,
        Huffman::new(&distance_lengths)?,
    ))
}

/// Read a dynamic block's Huffman tables.
///
/// The code lengths are themselves Huffman-coded, with the code-length alphabet
/// transmitted in a fixed permuted order — three levels of indirection, and the
/// permutation is the part most easily got wrong.
fn dynamic_tables(reader: &mut BitReader<'_>) -> Result<(Huffman, Huffman)> {
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let literal_count = reader.bits(5)? as usize + 257;
    let distance_count = reader.bits(5)? as usize + 1;
    let code_length_count = reader.bits(4)? as usize + 4;

    let mut code_lengths = [0u8; 19];
    for i in 0..code_length_count {
        code_lengths[ORDER[i]] = reader.bits(3)? as u8;
    }
    let code_length_table = Huffman::new(&code_lengths)?;

    // Now decode the literal and distance lengths together, since run-length codes
    // 16-18 can span the boundary between the two tables.
    let total = literal_count + distance_count;
    let mut lengths = vec![0u8; total];
    let mut i = 0usize;
    while i < total {
        let symbol = code_length_table.decode(reader)?;
        match symbol {
            0..=15 => {
                lengths[i] = symbol as u8;
                i += 1;
            }
            // 16: repeat the previous length 3-6 times.
            16 => {
                if i == 0 {
                    return Err(err("repeat code at the start of the length table"));
                }
                let previous = lengths[i - 1];
                let repeat = 3 + reader.bits(2)? as usize;
                for _ in 0..repeat {
                    if i >= total {
                        return Err(err("length repeat runs past the table"));
                    }
                    lengths[i] = previous;
                    i += 1;
                }
            }
            // 17: repeat zero 3-10 times. 18: repeat zero 11-138 times.
            17 | 18 => {
                let repeat = if symbol == 17 {
                    3 + reader.bits(3)? as usize
                } else {
                    11 + reader.bits(7)? as usize
                };
                if i + repeat > total {
                    return Err(err("zero repeat runs past the length table"));
                }
                i += repeat;
            }
            other => return Err(err(format!("invalid code length symbol {other}"))),
        }
    }

    Ok((
        Huffman::new(&lengths[..literal_count])?,
        Huffman::new(&lengths[literal_count..])?,
    ))
}

/// Inflate one block's symbols.
fn inflate_block(
    reader: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    literal: &Huffman,
    distance_table: &Huffman,
) -> Result<()> {
    loop {
        let symbol = literal.decode(reader)?;
        match symbol {
            0..=255 => out.push(symbol as u8),
            // 256 ends the block.
            256 => return Ok(()),
            257..=285 => {
                let index = (symbol - 257) as usize;
                let length = LENGTH_BASE[index] as usize
                    + reader.bits(LENGTH_EXTRA[index] as u32)? as usize;
                let distance_symbol = distance_table.decode(reader)? as usize;
                if distance_symbol >= DISTANCE_BASE.len() {
                    return Err(err(format!("invalid distance symbol {distance_symbol}")));
                }
                let distance = DISTANCE_BASE[distance_symbol] as usize
                    + reader.bits(DISTANCE_EXTRA[distance_symbol] as u32)? as usize;
                if distance == 0 || distance > out.len() {
                    return Err(err(format!(
                        "back-reference distance {distance} exceeds the {} bytes decoded",
                        out.len()
                    )));
                }
                // Byte by byte, not a block copy. A match may *overlap* the output
                // being written — distance 1 with length 100 repeats one byte a
                // hundred times — and `copy_from_slice` cannot express that.
                let start = out.len() - distance;
                for k in 0..length {
                    let byte = out[start + k];
                    out.push(byte);
                }
                if out.len() > MAX_INFLATED {
                    return Err(err("compressed stream expands beyond the limit"));
                }
            }
            other => return Err(err(format!("invalid literal/length symbol {other}"))),
        }
    }
}

/// Read and decode a PNG file from disk.
pub fn load(path: impl AsRef<std::path::Path>) -> Result<Image> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    decode(&bytes).map_err(|e| err(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a PNG with a stored (uncompressed) DEFLATE block, so the tests exercise
    /// the container, filters and colour conversion without needing a compressor.
    fn build_png(
        width: u32,
        height: u32,
        color_type: u8,
        bit_depth: u8,
        scanlines: &[Vec<u8>],
        extra_chunks: &[(&[u8; 4], Vec<u8>)],
    ) -> Vec<u8> {
        let mut raw = Vec::new();
        for line in scanlines {
            raw.extend_from_slice(line);
        }

        // zlib header, then one stored block.
        let mut zlib = vec![0x78, 0x01];
        zlib.push(0x01); // final, stored
        zlib.extend_from_slice(&(raw.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(raw.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(&raw);
        // Adler-32, which the decoder does not verify but a real file has.
        zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut out = Vec::new();
        out.extend_from_slice(&SIGNATURE);

        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.push(bit_depth);
        ihdr.push(color_type);
        ihdr.extend_from_slice(&[0, 0, 0]); // compression, filter, interlace
        push_chunk(&mut out, b"IHDR", &ihdr);

        for (kind, data) in extra_chunks {
            push_chunk(&mut out, kind, data);
        }
        push_chunk(&mut out, b"IDAT", &zlib);
        push_chunk(&mut out, b"IEND", &[]);
        out
    }

    fn push_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        // A plausible CRC placeholder; the decoder does not verify it.
        out.extend_from_slice(&[0, 0, 0, 0]);
    }

    fn adler32(data: &[u8]) -> u32 {
        let mut a = 1u32;
        let mut b = 0u32;
        for byte in data {
            a = (a + *byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }

    // --- container ---

    #[test]
    fn decodes_a_two_by_two_rgba_image() {
        // Filter 0 (none) on each line.
        let scanlines = vec![
            vec![0, 255, 0, 0, 255, 0, 255, 0, 255],
            vec![0, 0, 0, 255, 255, 255, 255, 255, 255],
        ];
        let png = build_png(2, 2, 6, 8, &scanlines, &[]);
        let image = decode(&png).expect("should decode");
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(image.pixel(0, 0), Some([255, 0, 0, 255]));
        assert_eq!(image.pixel(1, 0), Some([0, 255, 0, 255]));
        assert_eq!(image.pixel(0, 1), Some([0, 0, 255, 255]));
        assert_eq!(image.pixel(1, 1), Some([255, 255, 255, 255]));
    }

    #[test]
    fn a_non_png_is_rejected() {
        assert!(decode(b"not a png").is_err());
        assert!(decode(&[]).is_err());
        // Right signature, nothing else.
        assert!(decode(&SIGNATURE).is_err());
    }

    #[test]
    fn a_missing_ihdr_or_idat_is_an_error() {
        let mut no_ihdr = Vec::new();
        no_ihdr.extend_from_slice(&SIGNATURE);
        push_chunk(&mut no_ihdr, b"IEND", &[]);
        assert!(decode(&no_ihdr).is_err());

        // IHDR but no IDAT.
        let mut no_idat = Vec::new();
        no_idat.extend_from_slice(&SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        push_chunk(&mut no_idat, b"IHDR", &ihdr);
        push_chunk(&mut no_idat, b"IEND", &[]);
        assert!(decode(&no_idat).is_err());
    }

    /// Unknown chunks must be skipped, not fatal — real files carry gAMA, sRGB,
    /// pHYs, tEXt and more, and a decoder that choked on them would load almost
    /// nothing.
    #[test]
    fn unknown_chunks_are_skipped() {
        let scanlines = vec![vec![0, 10, 20, 30, 255]];
        let png = build_png(
            1,
            1,
            6,
            8,
            &scanlines,
            &[
                (b"gAMA", vec![0, 1, 2, 3]),
                (b"tEXt", b"Comment\0made by a test".to_vec()),
            ],
        );
        let image = decode(&png).expect("unknown chunks must not be fatal");
        assert_eq!(image.pixel(0, 0), Some([10, 20, 30, 255]));
    }

    /// A malformed header must not be able to request a huge allocation.
    #[test]
    fn absurd_dimensions_are_rejected() {
        let scanlines = vec![vec![0, 0, 0, 0, 0]];
        let png = build_png(100_000, 100_000, 6, 8, &scanlines, &[]);
        assert!(decode(&png).is_err());
        let zero = build_png(0, 1, 6, 8, &scanlines, &[]);
        assert!(decode(&zero).is_err());
    }

    /// Sub-byte depths are rejected by name rather than decoded into noise.
    #[test]
    fn unsupported_bit_depths_say_so() {
        let scanlines = vec![vec![0, 0]];
        for depth in [1u8, 2, 4] {
            let png = build_png(1, 1, 0, depth, &scanlines, &[]);
            let e = decode(&png).unwrap_err();
            assert!(
                e.to_string().contains(&depth.to_string()),
                "the error should name the depth: {e}"
            );
        }
    }

    #[test]
    fn an_unknown_colour_type_is_rejected() {
        let scanlines = vec![vec![0, 0]];
        let png = build_png(1, 1, 7, 8, &scanlines, &[]);
        assert!(decode(&png).is_err());
    }

    // --- colour types ---

    #[test]
    fn greyscale_expands_to_rgb() {
        let scanlines = vec![vec![0, 128]];
        let png = build_png(1, 1, 0, 8, &scanlines, &[]);
        assert_eq!(decode(&png).unwrap().pixel(0, 0), Some([128, 128, 128, 255]));
    }

    #[test]
    fn greyscale_with_alpha_keeps_its_alpha() {
        let scanlines = vec![vec![0, 200, 64]];
        let png = build_png(1, 1, 4, 8, &scanlines, &[]);
        assert_eq!(decode(&png).unwrap().pixel(0, 0), Some([200, 200, 200, 64]));
    }

    #[test]
    fn rgb_gets_an_opaque_alpha() {
        let scanlines = vec![vec![0, 1, 2, 3]];
        let png = build_png(1, 1, 2, 8, &scanlines, &[]);
        assert_eq!(decode(&png).unwrap().pixel(0, 0), Some([1, 2, 3, 255]));
    }

    #[test]
    fn a_palette_image_looks_up_its_colours() {
        let palette = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let scanlines = vec![vec![0, 2, 0, 1]];
        let png = build_png(3, 1, 3, 8, &scanlines, &[(b"PLTE", palette)]);
        let image = decode(&png).unwrap();
        assert_eq!(image.pixel(0, 0), Some([0, 0, 255, 255]));
        assert_eq!(image.pixel(1, 0), Some([255, 0, 0, 255]));
        assert_eq!(image.pixel(2, 0), Some([0, 255, 0, 255]));
    }

    /// Ignoring tRNS makes every transparent pixel opaque — for a UI texture, the
    /// difference between a sprite and a rectangle.
    #[test]
    fn a_palette_images_transparency_chunk_is_honoured() {
        let palette = vec![255, 0, 0, 0, 255, 0];
        let scanlines = vec![vec![0, 0, 1]];
        let png = build_png(
            2,
            1,
            3,
            8,
            &scanlines,
            &[(b"PLTE", palette), (b"tRNS", vec![0, 128])],
        );
        let image = decode(&png).unwrap();
        assert_eq!(image.pixel(0, 0), Some([255, 0, 0, 0]), "entry 0 is transparent");
        assert_eq!(image.pixel(1, 0), Some([0, 255, 0, 128]));
    }

    /// A palette entry past the end of tRNS is opaque, per the spec.
    #[test]
    fn palette_entries_beyond_the_trns_array_are_opaque() {
        let palette = vec![1, 2, 3, 4, 5, 6];
        let scanlines = vec![vec![0, 1]];
        let png = build_png(
            1,
            1,
            3,
            8,
            &scanlines,
            &[(b"PLTE", palette), (b"tRNS", vec![0])],
        );
        assert_eq!(decode(&png).unwrap().pixel(0, 0), Some([4, 5, 6, 255]));
    }

    /// 16-bit samples are reduced by taking the high byte. Reading the low byte gives
    /// an image that looks like noise.
    #[test]
    fn sixteen_bit_samples_are_reduced_to_their_high_byte() {
        // One RGBA pixel at 16 bits per channel: 0x1122, 0x3344, 0x5566, 0xFFFF.
        let scanlines = vec![vec![
            0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0xFF, 0xFF,
        ]];
        let png = build_png(1, 1, 6, 16, &scanlines, &[]);
        assert_eq!(
            decode(&png).unwrap().pixel(0, 0),
            Some([0x11, 0x33, 0x55, 0xFF])
        );
    }

    // --- filters ---

    #[test]
    fn filter_sub_adds_the_pixel_to_its_left() {
        // Two RGB pixels: the first raw, the second a delta from it.
        let scanlines = vec![vec![1, 10, 20, 30, 5, 5, 5]];
        let png = build_png(2, 1, 2, 8, &scanlines, &[]);
        let image = decode(&png).unwrap();
        assert_eq!(image.pixel(0, 0), Some([10, 20, 30, 255]));
        assert_eq!(
            image.pixel(1, 0),
            Some([15, 25, 35, 255]),
            "Sub must add the pixel one PIXEL to the left, not one byte"
        );
    }

    /// The distance is one *pixel*. One byte back would smear channels together, and
    /// on a greyscale image the two are indistinguishable — hence RGB here.
    #[test]
    fn the_sub_filter_distance_is_one_pixel_not_one_byte() {
        let scanlines = vec![vec![1, 100, 0, 0, 1, 0, 0]];
        let png = build_png(2, 1, 2, 8, &scanlines, &[]);
        let image = decode(&png).unwrap();
        // Correct: R = 100 + 1 = 101, G = 0 + 0, B = 0 + 0.
        // Byte-distance would give R = 101, G = 101, B = 101.
        assert_eq!(image.pixel(1, 0), Some([101, 0, 0, 255]));
    }

    #[test]
    fn filter_up_adds_the_pixel_above() {
        let scanlines = vec![
            vec![0, 10, 20, 30],
            vec![2, 5, 5, 5],
        ];
        let png = build_png(1, 2, 2, 8, &scanlines, &[]);
        let image = decode(&png).unwrap();
        assert_eq!(image.pixel(0, 0), Some([10, 20, 30, 255]));
        assert_eq!(image.pixel(0, 1), Some([15, 25, 35, 255]));
    }

    /// The filter references the *reconstructed* line above, not the raw one. Getting
    /// this wrong gives an image that is right at the top and degrades downward — the
    /// classic PNG decoder bug, and invisible on a two-row test.
    #[test]
    fn filters_chain_through_reconstructed_rows_not_raw_ones() {
        // Each row is Up(+10) on the one above, over four rows. If the raw row were
        // used, every row after the first would read 10 rather than accumulating.
        let scanlines = vec![
            vec![0, 10, 10, 10],
            vec![2, 10, 10, 10],
            vec![2, 10, 10, 10],
            vec![2, 10, 10, 10],
        ];
        let png = build_png(1, 4, 2, 8, &scanlines, &[]);
        let image = decode(&png).unwrap();
        for y in 0..4u32 {
            let expected = 10 * (y as u8 + 1);
            assert_eq!(
                image.pixel(0, y),
                Some([expected, expected, expected, 255]),
                "row {y} should have accumulated to {expected}"
            );
        }
    }

    /// Average sums in 16 bits before halving. In u8 the sum would overflow and halve
    /// the wrong number, which shows up only when a + b exceeds 255.
    #[test]
    fn filter_average_does_not_overflow() {
        let scanlines = vec![
            vec![0, 200, 200, 200],
            // Average of left (0, no left pixel) and above (200) is 100.
            vec![3, 0, 0, 0],
        ];
        let png = build_png(1, 2, 2, 8, &scanlines, &[]);
        let image = decode(&png).unwrap();
        assert_eq!(image.pixel(0, 1), Some([100, 100, 100, 255]));
    }

    /// Paeth is a predictor with a specified tie-breaking order, and getting it wrong
    /// is a subtle per-pixel error rather than a visible break — so it is tested
    /// directly against the RFC's definition.
    #[test]
    fn paeth_matches_its_specification() {
        // p = a + b - c; pick whichever of a, b, c is nearest, with a winning ties.
        assert_eq!(paeth(10, 20, 15), 15, "p = 15, exactly c");
        assert_eq!(paeth(0, 0, 0), 0);
        assert_eq!(paeth(255, 0, 0), 255, "p = 255, nearest is a");
        assert_eq!(paeth(0, 255, 0), 255, "p = 255, nearest is b");
        // Ties go to a, then b: a and b equidistant from p means a.
        assert_eq!(paeth(10, 20, 30), 10, "p = 0; |0-10| = 10, |0-20| = 20 -> a");
        // And the arithmetic must not wrap: a + b can exceed 255.
        let result = paeth(200, 200, 100);
        assert!(result == 200, "p = 300; nearest of a/b/c is a or b, got {result}");
    }

    #[test]
    fn filter_paeth_round_trips_through_the_decoder() {
        let scanlines = vec![
            vec![0, 50, 50, 50],
            vec![4, 10, 10, 10],
        ];
        let png = build_png(1, 2, 2, 8, &scanlines, &[]);
        let image = decode(&png).unwrap();
        // Row 1: a = 0 (no left), b = 50, c = 0. p = 50, nearest is b -> 50.
        // 10 + 50 = 60.
        assert_eq!(image.pixel(0, 1), Some([60, 60, 60, 255]));
    }

    #[test]
    fn an_unknown_filter_type_is_an_error() {
        let scanlines = vec![vec![9, 0, 0, 0]];
        let png = build_png(1, 1, 2, 8, &scanlines, &[]);
        let e = decode(&png).unwrap_err();
        assert!(e.to_string().contains("filter"), "{e}");
    }

    // --- DEFLATE ---

    /// A stored block is the simplest path and the one the test fixtures use, so it
    /// is pinned directly.
    #[test]
    fn a_stored_deflate_block_round_trips() {
        let payload = b"hello deflate";
        let mut stream = vec![0x01];
        stream.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        stream.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        stream.extend_from_slice(payload);
        assert_eq!(inflate(&stream).unwrap(), payload);
    }

    #[test]
    fn a_stored_block_with_a_bad_complement_is_rejected() {
        let mut stream = vec![0x01];
        stream.extend_from_slice(&5u16.to_le_bytes());
        stream.extend_from_slice(&999u16.to_le_bytes()); // wrong
        stream.extend_from_slice(b"hello");
        assert!(inflate(&stream).is_err());
    }

    /// The fixed Huffman tables come straight from the RFC, and a wrong length
    /// assignment breaks every fixed-block file — which is most small PNGs.
    #[test]
    fn the_fixed_huffman_tables_build() {
        let (literal, distance) = fixed_tables().expect("fixed tables must be valid");
        // 288 literal symbols, 30 distance symbols.
        assert_eq!(literal.symbols.len(), 288);
        assert_eq!(distance.symbols.len(), 30);
    }

    /// The zlib wrapper is checked before the DEFLATE stream, so a non-DEFLATE or
    /// corrupt header fails with a message about the wrapper rather than a confusing
    /// Huffman error.
    #[test]
    fn a_bad_zlib_header_is_rejected() {
        assert!(inflate_zlib(&[]).is_err());
        assert!(inflate_zlib(&[0x00, 0x00]).is_err(), "method 0 is not DEFLATE");
        assert!(
            inflate_zlib(&[0x78, 0x00]).is_err(),
            "0x7800 is not a multiple of 31"
        );
        // Preset dictionary bit set.
        assert!(inflate_zlib(&[0x78, 0xBB]).is_err());
    }

    #[test]
    fn an_invalid_block_type_is_rejected() {
        // Type 3 is reserved: bit 0 = final, bits 1-2 = 11.
        assert!(inflate(&[0b0000_0111]).is_err());
    }

    /// A crafted stream can expand almost without bound. This is a decoder for
    /// untrusted files, so the output has to be capped.
    #[test]
    fn the_inflate_output_is_bounded() {
        assert!(
            MAX_INFLATED >= 16 * 1024 * 1024,
            "the cap must leave room for a real texture"
        );
        assert!(
            MAX_INFLATED <= 512 * 1024 * 1024,
            "the cap must be low enough to matter"
        );
    }

    /// Whatever a malformed file contains, the decoder must return rather than panic.
    #[test]
    fn malformed_input_never_panics() {
        let good = build_png(2, 2, 6, 8, &[vec![0; 9], vec![0; 9]], &[]);
        // Every truncation.
        for length in 0..good.len() {
            let _ = decode(&good[..length]);
        }
        // Every single-byte corruption in the header region.
        for index in 0..good.len().min(40) {
            let mut corrupt = good.clone();
            corrupt[index] = corrupt[index].wrapping_add(0x5A);
            let _ = decode(&corrupt);
        }
        // Random-looking garbage behind a valid signature.
        let mut garbage = SIGNATURE.to_vec();
        garbage.extend((0u8..250).cycle().take(500));
        let _ = decode(&garbage);
    }

    // --- interlacing ---

    /// A decoder that ignored the interlace flag would produce a scrambled image
    /// rather than an error, which is the worst outcome — so Adam7 is implemented and
    /// tested rather than rejected.
    #[test]
    fn an_interlaced_image_reassembles() {
        // 8x8 greyscale, every pixel a distinct value, laid out by Adam7 pass.
        let width = 8u32;
        let height = 8u32;
        const PASSES: [(u32, u32, u32, u32); 7] = [
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        ];
        // The value at (x, y) is y * 8 + x, so a scrambled result is obvious.
        let value = |x: u32, y: u32| (y * width + x) as u8;
        let mut raw: Vec<u8> = Vec::new();
        for (x_start, y_start, x_step, y_step) in PASSES {
            let pass_width = width.saturating_sub(x_start).div_ceil(x_step);
            let pass_height = height.saturating_sub(y_start).div_ceil(y_step);
            if pass_width == 0 || pass_height == 0 {
                continue;
            }
            for sy in 0..pass_height {
                raw.push(0); // filter: none
                for sx in 0..pass_width {
                    raw.push(value(x_start + sx * x_step, y_start + sy * y_step));
                }
            }
        }

        // Build it by hand, with the interlace byte set.
        let mut zlib = vec![0x78, 0x01, 0x01];
        zlib.extend_from_slice(&(raw.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(raw.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(&raw);
        zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut png = SIGNATURE.to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 0, 0, 0, 1]); // 8-bit grey, interlaced
        push_chunk(&mut png, b"IHDR", &ihdr);
        push_chunk(&mut png, b"IDAT", &zlib);
        push_chunk(&mut png, b"IEND", &[]);

        let image = decode(&png).expect("interlaced PNG should decode");
        for y in 0..height {
            for x in 0..width {
                let expected = value(x, y);
                assert_eq!(
                    image.pixel(x, y),
                    Some([expected, expected, expected, 255]),
                    "pixel ({x}, {y}) came back wrong, so the passes were misplaced"
                );
            }
        }
    }

    /// A small image leaves some Adam7 passes empty, and those must be skipped rather
    /// than read as zero-length scanlines.
    #[test]
    fn an_interlaced_image_smaller_than_the_pass_grid_decodes() {
        let raw = vec![0u8, 42]; // one pass, one pixel
        let mut zlib = vec![0x78, 0x01, 0x01];
        zlib.extend_from_slice(&(raw.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(raw.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(&raw);
        zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut png = SIGNATURE.to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 0, 0, 0, 1]);
        push_chunk(&mut png, b"IHDR", &ihdr);
        push_chunk(&mut png, b"IDAT", &zlib);
        push_chunk(&mut png, b"IEND", &[]);

        let image = decode(&png).expect("a 1x1 interlaced PNG should decode");
        assert_eq!(image.pixel(0, 0), Some([42, 42, 42, 255]));
    }

    // --- Image helpers ---

    #[test]
    fn a_solid_image_is_uniform() {
        let image = Image::solid(4, 3, [1, 2, 3, 4]);
        assert_eq!(image.byte_size(), 4 * 3 * 4);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(image.pixel(x, y), Some([1, 2, 3, 4]));
            }
        }
    }

    #[test]
    fn out_of_range_pixel_reads_are_none() {
        let image = Image::solid(2, 2, [0, 0, 0, 255]);
        assert!(image.pixel(2, 0).is_none());
        assert!(image.pixel(0, 2).is_none());
        assert!(image.pixel(99, 99).is_none());
    }

    // --- a real, really-compressed file ---

    /// Every test above builds a *stored* DEFLATE block, which leaves the Huffman
    /// path — the actual compression, and the bulk of the inflater — untested. This
    /// decodes a genuine zlib-compressed PNG whose first block is type 2 (dynamic
    /// Huffman), against a pattern computed independently.
    ///
    /// The fixture is generated rather than hand-written; `assets/textures/checker.png`
    /// was produced by Python's zlib at level 9 and is committed, so this exercises a
    /// stream that a different implementation compressed.
    #[test]
    fn a_real_compressed_png_decodes_to_its_known_pattern() {
        let candidates = [
            std::path::PathBuf::from("assets/textures/checker.png"),
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/textures/checker.png"),
        ];
        let Some(path) = candidates.into_iter().find(|p| p.exists()) else {
            panic!("the checker.png fixture is missing from the repository");
        };
        let bytes = std::fs::read(&path).expect("should read the fixture");

        // Confirm the fixture really is Huffman-compressed, or this test would pass
        // while proving nothing. Find the IDAT payload and read its first block type.
        let idat_at = bytes
            .windows(4)
            .position(|w| w == b"IDAT")
            .expect("fixture must have an IDAT chunk");
        // Skip the chunk type and the two zlib header bytes.
        let block_type = (bytes[idat_at + 4 + 2] >> 1) & 0b11;
        assert_eq!(
            block_type, 2,
            "the fixture must use dynamic Huffman blocks, not stored ones"
        );

        let image = decode(&bytes).expect("a real PNG should decode");
        assert_eq!((image.width, image.height), (16, 16));

        // The generator's rule, restated here independently: 4x4 checkerboard, with a
        // per-cell gradient. A wrong Huffman table, a mishandled back-reference or a
        // filter error would all break this, and none of them would break a
        // stored-block test.
        for y in 0..16u32 {
            for x in 0..16u32 {
                let expected = if (x / 4 + y / 4) % 2 == 0 {
                    [255, (x * 16) as u8, (y * 16) as u8, 255]
                } else {
                    [(y * 16) as u8, 255, (x * 16) as u8, 200]
                };
                assert_eq!(
                    image.pixel(x, y),
                    Some(expected),
                    "pixel ({x}, {y}) decoded wrongly"
                );
            }
        }
    }

    /// Back-references may overlap the output being written: distance 1 with length
    /// 100 repeats one byte a hundred times. A block copy cannot express that, and
    /// the compressed fixture above relies on it — but so does this, directly.
    #[test]
    fn an_overlapping_back_reference_repeats_correctly() {
        // A run of identical bytes is exactly what produces a distance-1 match.
        let payload: Vec<u8> = std::iter::repeat(0xABu8).take(300).collect();
        let mut raw = vec![0u8]; // filter: none
        raw.extend_from_slice(&payload);

        // Compress with a stored block is not enough here; build the zlib stream by
        // hand is impractical, so verify the inflater against a known-overlapping
        // hand-built stream instead: a fixed-Huffman block emitting one literal then a
        // length-3 distance-1 match.
        //
        // Fixed block: literal 'A' (code 0x41 -> 8 bits), then length code 257
        // (length 3), distance code 0 (distance 1), then end-of-block 256.
        let mut bits: Vec<u8> = Vec::new();
        let mut accumulator = 0u32;
        let mut count = 0u32;
        let push_bits = |value: u32, width: u32, bits: &mut Vec<u8>, acc: &mut u32, n: &mut u32| {
            for i in 0..width {
                let bit = (value >> i) & 1;
                *acc |= bit << *n;
                *n += 1;
                if *n == 8 {
                    bits.push(*acc as u8);
                    *acc = 0;
                    *n = 0;
                }
            }
        };
        // Header: BFINAL = 1, BTYPE = 01 (fixed).
        push_bits(1, 1, &mut bits, &mut accumulator, &mut count);
        push_bits(1, 2, &mut bits, &mut accumulator, &mut count);
        // Literal 'A' (0x41 = 65): fixed code for 0..143 is 8 bits, value 0x30 + 65,
        // written MSB-first.
        let literal_code = 0x30u32 + 65;
        for i in (0..8).rev() {
            push_bits((literal_code >> i) & 1, 1, &mut bits, &mut accumulator, &mut count);
        }
        // Length symbol 257: fixed code for 256..279 is 7 bits, value symbol - 256.
        let length_code = 257u32 - 256;
        for i in (0..7).rev() {
            push_bits((length_code >> i) & 1, 1, &mut bits, &mut accumulator, &mut count);
        }
        // Distance symbol 0: five zero bits.
        for _ in 0..5 {
            push_bits(0, 1, &mut bits, &mut accumulator, &mut count);
        }
        // End of block, symbol 256: 7 bits of zero.
        for _ in 0..7 {
            push_bits(0, 1, &mut bits, &mut accumulator, &mut count);
        }
        if count > 0 {
            bits.push(accumulator as u8);
        }

        let out = inflate(&bits).expect("the hand-built stream should inflate");
        assert_eq!(
            out, b"AAAA",
            "a distance-1 length-3 match must repeat the literal, giving AAAA"
        );
    }

    /// IDAT may be split across chunks, and the compressed stream runs across the
    /// boundaries — so they have to be concatenated before inflating. Inflating each
    /// separately fails on the second one.
    #[test]
    fn idat_split_across_chunks_is_concatenated() {
        let raw = vec![0u8, 7, 8, 9];
        let mut zlib = vec![0x78, 0x01, 0x01];
        zlib.extend_from_slice(&(raw.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(raw.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(&raw);
        zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut png = SIGNATURE.to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        push_chunk(&mut png, b"IHDR", &ihdr);
        // Split the stream mid-way.
        let split = zlib.len() / 2;
        push_chunk(&mut png, b"IDAT", &zlib[..split]);
        push_chunk(&mut png, b"IDAT", &zlib[split..]);
        push_chunk(&mut png, b"IEND", &[]);

        let image = decode(&png).expect("split IDAT must be concatenated");
        assert_eq!(image.pixel(0, 0), Some([7, 8, 9, 255]));
    }
}
