// Minimal JSON parser — replaces `serde_json`, foundation for file-based scene loading.
//
// Design notes:
// - Recursive descent over the raw bytes of the input `&str`. Byte-level scanning keeps the
//   hot loop branch-simple; every structural token in JSON is ASCII, so slicing at token
//   boundaries can never split a multi-byte UTF-8 sequence.
// - Objects are `Vec<(String, JsonValue)>` instead of a map: scene files are small, document
//   order matters for diagnostics, and this avoids pulling in hashing machinery. Lookup is a
//   linear scan, which beats hashing for the handful of keys we expect.
// - Recursion depth is hard-capped (`MAX_DEPTH`), so hostile or machine-generated input cannot
//   blow the stack — it gets a clean `Err` instead.
// - Errors always carry the byte offset plus what was expected and what was found, because a
//   hand-written parser is only debuggable if its errors are precise.
// - Numbers are validated against the JSON grammar by hand before being handed to
//   `str::parse::<f64>()`; that rejects JSON5-isms (`+1`, `.5`, `01`, `1.`) which `parse`
//   would otherwise happily accept.

// Note: this is a Phase 2.2 foundation, so not every accessor has a caller yet — the crate
// root already carries `#![allow(dead_code)]`, so no extra attribute is needed here.

use crate::engine::core::error::{EngineError, Result};

/// Maximum nesting depth of arrays/objects. Beyond this the parser errors out instead of
/// recursing further, which bounds stack usage for arbitrary input.
pub const MAX_DEPTH: usize = 128;

/// A parsed JSON value.
///
/// `Object` keeps its pairs in document order; duplicate keys are preserved and lookups
/// return the first match (the same observable behaviour as most JSON libraries).
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// Any JSON number, always stored as `f64`.
    Number(f64),
    /// A string with all escapes already decoded.
    String(String),
    /// `[...]`
    Array(Vec<JsonValue>),
    /// `{...}` as ordered key/value pairs.
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Human-readable name of this value's kind, for caller-side error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "bool",
            JsonValue::Number(_) => "number",
            JsonValue::String(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        }
    }

    /// True if this is `null`.
    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    /// Look up `key` in an object. `None` if this is not an object or the key is absent.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Element `i` of an array. `None` if this is not an array or `i` is out of bounds.
    pub fn index(&self, i: usize) -> Option<&JsonValue> {
        match self {
            JsonValue::Array(items) => items.get(i),
            _ => None,
        }
    }

    /// Borrow the string payload.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Numeric payload as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Numeric payload narrowed to `f32` (the engine's native float width).
    ///
    /// The parser already rejects any literal that does not narrow to a finite `f32`
    /// (with a byte offset, which this accessor could never provide), so for parsed
    /// documents the check below never fires. It stays because `JsonValue` can also be
    /// constructed by hand: a synthetic `Number(1e300)` or `Number(NaN)` must not smuggle
    /// an infinity into a transform. Merely losing precision (`1e-300` flushing to zero)
    /// is still fine, that is ordinary float behaviour.
    pub fn as_f32(&self) -> Option<f32> {
        let narrowed = self.as_f64()? as f32;
        if narrowed.is_finite() {
            Some(narrowed)
        } else {
            None
        }
    }

    /// Numeric payload as `u32`. Rejects negative, non-integral and out-of-range values so
    /// that counts and indices in scene files fail loudly instead of silently wrapping.
    pub fn as_u32(&self) -> Option<u32> {
        let n = self.as_f64()?;
        if n.fract() != 0.0 || n < 0.0 || n > u32::MAX as f64 {
            return None;
        }
        Some(n as u32)
    }

    /// Boolean payload.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Elements of an array.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    /// Ordered key/value pairs of an object.
    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(pairs) => Some(pairs.as_slice()),
            _ => None,
        }
    }

    /// A fixed-size array of numbers, e.g. `[0.0, 1.0, 0.0]`. The length must match exactly,
    /// so a truncated vector in a scene file is an error rather than a silent zero.
    pub fn as_f32_array<const N: usize>(&self) -> Option<[f32; N]> {
        let items = self.as_array()?;
        if items.len() != N {
            return None;
        }
        let mut out = [0.0f32; N];
        for (slot, item) in out.iter_mut().zip(items) {
            *slot = item.as_f32()?;
        }
        Some(out)
    }

    /// This value read as a 3-component vector array (`[x, y, z]`).
    pub fn get_vec3_array(&self) -> Option<[f32; 3]> {
        self.as_f32_array::<3>()
    }

    /// This value read as a 4-component vector array (`[x, y, z, w]`).
    pub fn get_vec4_array(&self) -> Option<[f32; 4]> {
        self.as_f32_array::<4>()
    }

    /// Object field as a string.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    /// Object field as `f64`.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }

    /// Object field as `f32` — the workhorse for scene descriptions.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f32()
    }

    /// Object field as `u32`.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.as_u32()
    }

    /// Object field as a bool.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    /// Object field as an array.
    pub fn get_array(&self, key: &str) -> Option<&[JsonValue]> {
        self.get(key)?.as_array()
    }

    /// Object field as `[x, y, z]`.
    pub fn get_vec3(&self, key: &str) -> Option<[f32; 3]> {
        self.get(key)?.get_vec3_array()
    }

    /// Object field as `[x, y, z, w]`.
    pub fn get_vec4(&self, key: &str) -> Option<[f32; 4]> {
        self.get(key)?.get_vec4_array()
    }
}

/// Parse a complete JSON document. Trailing content after the top-level value is an error.
///
/// A single leading UTF-8 byte-order mark is tolerated. Windows editors add one silently and
/// RFC 8259 lets parsers ignore it; without this, a hand-edited scene file fails with a
/// baffling "found byte 0xef" at offset 0. Nothing else about the strictness changes.
pub fn parse(input: &str) -> Result<JsonValue> {
    let mut parser = Parser::new(input);
    parser.skip_bom();
    parser.skip_whitespace();
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.pos < parser.bytes.len() {
        return parser.error_at(parser.pos, "end of input after the top-level value");
    }
    Ok(value)
}

/// Cursor over the input document.
struct Parser<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, bytes: input.as_bytes(), pos: 0, depth: 0 }
    }

    // --- error helpers -------------------------------------------------------------------

    /// Build a parse error. Associated (not a method) so it can also be used to report on a
    /// position other than the cursor, with a hand-written "found" description.
    fn error<T>(offset: usize, expected: &str, found: &str) -> Result<T> {
        Err(EngineError::InvalidState(format!(
            "json parse error at byte offset {offset}: expected {expected}, found {found}"
        )))
    }

    fn error_at<T>(&self, offset: usize, expected: &str) -> Result<T> {
        Self::error(offset, expected, &self.describe(offset))
    }

    /// Describe the byte at `offset` for an error message. Non-graphic bytes (and the lead
    /// bytes of multi-byte sequences) are shown in hex rather than mangled into a char.
    fn describe(&self, offset: usize) -> String {
        match self.bytes.get(offset) {
            None => String::from("end of input"),
            Some(&b) if b.is_ascii_graphic() => format!("`{}`", b as char),
            Some(&b) => format!("byte 0x{b:02x}"),
        }
    }

    // --- scanning primitives -------------------------------------------------------------

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    /// Consume a leading UTF-8 BOM (`EF BB BF`) if the document opens with one. Only ever
    /// called at offset 0, so a BOM anywhere else still fails as an unexpected byte. The
    /// cursor is advanced rather than the input re-sliced, which keeps every reported offset
    /// an absolute byte offset into the original file.
    fn skip_bom(&mut self) {
        if self.pos == 0 && self.bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            self.pos = 3;
        }
    }

    /// JSON whitespace is exactly space, tab, LF and CR.
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')) {
            self.pos += 1;
        }
    }

    fn skip_digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
    }

    /// Enter a nested container, enforcing the depth cap.
    fn enter(&mut self) -> Result<()> {
        if self.depth >= MAX_DEPTH {
            return Err(EngineError::InvalidState(format!(
                "json parse error at byte offset {}: nesting depth exceeds the limit of {}",
                self.pos, MAX_DEPTH
            )));
        }
        self.depth += 1;
        Ok(())
    }

    // --- grammar -------------------------------------------------------------------------

    /// Parse one value. The cursor must already be at the first non-whitespace byte.
    fn parse_value(&mut self) -> Result<JsonValue> {
        match self.peek() {
            None => self.error_at(self.pos, "a JSON value"),
            Some(b'n') => self.parse_literal("null", JsonValue::Null),
            Some(b't') => self.parse_literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal("false", JsonValue::Bool(false)),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(_) => self.error_at(self.pos, "a JSON value"),
        }
    }

    fn parse_literal(&mut self, word: &str, value: JsonValue) -> Result<JsonValue> {
        let start = self.pos;
        let expected = word.as_bytes();
        let end = start + expected.len();
        if end <= self.bytes.len() && &self.bytes[start..end] == expected {
            self.pos = end;
            Ok(value)
        } else {
            self.error_at(start, &format!("the literal `{word}`"))
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.enter()?;
        self.pos += 1; // consume '['
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            self.skip_whitespace();
            // A trailing comma lands here and fails as "expected a JSON value" — deliberate.
            items.push(self.parse_value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return self.error_at(self.pos, "`,` or `]` in array"),
            }
        }
        self.depth -= 1;
        Ok(JsonValue::Array(items))
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.enter()?;
        self.pos += 1; // consume '{'
        let mut pairs = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Object(pairs));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return self.error_at(self.pos, "`\"` starting an object key");
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return self.error_at(self.pos, "`:` after an object key");
            }
            self.pos += 1;
            self.skip_whitespace();
            let value = self.parse_value()?;
            pairs.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return self.error_at(self.pos, "`,` or `}` in object"),
            }
        }
        self.depth -= 1;
        Ok(JsonValue::Object(pairs))
    }

    /// Parse a string literal; the cursor must be on the opening quote.
    fn parse_string(&mut self) -> Result<String> {
        let open = self.pos;
        self.pos += 1; // consume '"'
        // Decoded bytes, assembled as UTF-8 and validated once at the end.
        let mut out: Vec<u8> = Vec::new();
        loop {
            let byte = match self.peek() {
                Some(b) => b,
                None => {
                    return Self::error(
                        open,
                        "a closing `\"` for the string starting here",
                        "end of input",
                    )
                }
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    break;
                }
                b'\\' => {
                    self.pos += 1;
                    self.parse_escape(&mut out)?;
                }
                // Raw control characters are illegal inside JSON strings.
                0x00..=0x1F => {
                    return self
                        .error_at(self.pos, "an escape sequence instead of a raw control character")
                }
                _ => {
                    out.push(byte);
                    self.pos += 1;
                }
            }
        }
        // Can only fail if the input itself held invalid UTF-8, which `&str` forbids — but we
        // report instead of unwrapping so a future byte-slice entry point stays panic-free.
        String::from_utf8(out).map_err(|_| {
            EngineError::InvalidState(format!(
                "json parse error at byte offset {open}: string decoded to invalid UTF-8"
            ))
        })
    }

    /// Decode the escape sequence following a backslash; the cursor is on the escape
    /// character itself.
    fn parse_escape(&mut self, out: &mut Vec<u8>) -> Result<()> {
        let at = self.pos;
        let byte = match self.peek() {
            Some(b) => b,
            None => return Self::error(at, "an escape character after `\\`", "end of input"),
        };
        self.pos += 1;
        let decoded = match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{0008}',
            b'f' => '\u{000C}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => return self.parse_unicode_escape(out, at),
            _ => {
                return self.error_at(at, "one of `\"` `\\` `/` `b` `f` `n` `r` `t` `u` after `\\`")
            }
        };
        push_char(out, decoded);
        Ok(())
    }

    /// Decode a `u`-escape, joining surrogate pairs into a single scalar value.
    /// The cursor sits just past the `u`; `at` points at that `u` for error reporting.
    fn parse_unicode_escape(&mut self, out: &mut Vec<u8>, at: usize) -> Result<()> {
        let unit = self.read_hex4()?;
        if (0xD800..0xDC00).contains(&unit) {
            // High surrogate: a low surrogate escape must follow, otherwise the codepoint is
            // unrepresentable as a Rust `char`.
            if self.peek() != Some(b'\\') || self.peek_at(1) != Some(b'u') {
                return self.error_at(self.pos, "a low surrogate escape completing the pair");
            }
            let low_at = self.pos;
            self.pos += 2; // consume the backslash and the `u`
            let low = self.read_hex4()?;
            if !(0xDC00..0xE000).contains(&low) {
                return self.error_at(low_at, "a low surrogate in the range DC00-DFFF");
            }
            let cp = 0x1_0000u32 + ((unit as u32 - 0xD800) << 10) + (low as u32 - 0xDC00);
            match char::from_u32(cp) {
                Some(ch) => push_char(out, ch),
                None => return self.error_at(at, "a valid Unicode scalar value"),
            }
        } else if (0xDC00..0xE000).contains(&unit) {
            return self.error_at(at, "a Unicode escape that is not a lone low surrogate");
        } else {
            match char::from_u32(unit as u32) {
                Some(ch) => push_char(out, ch),
                None => return self.error_at(at, "a valid Unicode scalar value"),
            }
        }
        Ok(())
    }

    /// Read exactly four hexadecimal digits and advance past them.
    fn read_hex4(&mut self) -> Result<u16> {
        let start = self.pos;
        if start + 4 > self.bytes.len() {
            return Self::error(start, "4 hexadecimal digits in a unicode escape", "end of input");
        }
        let mut value: u16 = 0;
        for i in 0..4 {
            let byte = self.bytes[start + i];
            let digit = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => return self.error_at(start + i, "a hexadecimal digit in a unicode escape"),
            };
            value = (value << 4) | digit as u16;
        }
        self.pos = start + 4;
        Ok(value)
    }

    /// Validate a number against the JSON grammar, then convert it with `str::parse`.
    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return self.error_at(self.pos, "no further digits after a leading zero");
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                self.skip_digits();
            }
            _ => return self.error_at(self.pos, "a digit in a number"),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return self.error_at(self.pos, "at least one digit after `.`");
            }
            self.skip_digits();
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return self.error_at(self.pos, "at least one digit in the exponent");
            }
            self.skip_digits();
        }
        // Slice boundaries are digits or ASCII punctuation, so this can never split a char.
        let text = &self.input[start..self.pos];
        match text.parse::<f64>() {
            // Reject values that are not representable as a *finite f32*, not just
            // a finite f64. The engine's native float is f32, and every consumer
            // narrows eventually; `1e300` is a perfectly finite f64 that `as f32`
            // turns into infinity. Rejecting it here, with a byte offset, is the
            // difference between "scene.json: parse error at byte offset 1042" and
            // an optional field silently falling back to its default because the
            // accessor returned `None` — the field did not "go missing", the file
            // is wrong, and only the parser knows where. Losing precision (`1e-300`
            // flushing to zero) stays accepted: that is ordinary float narrowing.
            Ok(n) if (n as f32).is_finite() => Ok(JsonValue::Number(n)),
            Ok(_) => self.error_at(start, "a number representable as a finite f32"),
            Err(_) => self.error_at(start, "a valid number"),
        }
    }
}

/// Append a `char` to a UTF-8 byte buffer.
fn push_char(out: &mut Vec<u8>, ch: char) {
    let mut buf = [0u8; 4];
    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse and unwrap, surfacing the error text on failure.
    fn ok(input: &str) -> JsonValue {
        match parse(input) {
            Ok(v) => v,
            Err(e) => panic!("expected `{input}` to parse, got: {e}"),
        }
    }

    /// Assert the input is rejected and hand back the message for inspection.
    fn err(input: &str) -> String {
        match parse(input) {
            Ok(v) => panic!("expected `{input}` to be rejected, got: {v:?}"),
            Err(e) => e.to_string(),
        }
    }

    /// A JSON unicode escape (`backslash` + `u` + hex), assembled at runtime. Built this way
    /// on purpose: a literal escape in this source risks being decoded by tooling before it
    /// ever reaches the parser, which would silently gut these tests.
    fn uesc(hex: &str) -> String {
        let mut s = String::from("\\");
        s.push('u');
        s.push_str(hex);
        s
    }

    /// Wrap a raw string body in JSON quotes.
    fn quoted(body: &str) -> String {
        format!("\"{body}\"")
    }

    /// `depth` nested containers of one kind, e.g. `[[[]]]`.
    fn nested(open: char, close: char, depth: usize) -> String {
        let mut s = String::with_capacity(depth * 2);
        for _ in 0..depth {
            s.push(open);
        }
        for _ in 0..depth {
            s.push(close);
        }
        s
    }

    /// Well-formed alternating array/object nest of `depth` levels wrapped around a `1`.
    fn mixed_nest(depth: usize) -> String {
        let mut open = String::new();
        let mut close = String::new();
        for i in 0..depth {
            if i % 2 == 0 {
                open.push('[');
                close.insert(0, ']');
            } else {
                open.push_str("{\"k\":");
                close.insert(0, '}');
            }
        }
        format!("{open}1{close}")
    }

    // --- primitives ----------------------------------------------------------------------

    #[test]
    fn parses_literals() {
        assert_eq!(ok("null"), JsonValue::Null);
        assert_eq!(ok("true"), JsonValue::Bool(true));
        assert_eq!(ok("false"), JsonValue::Bool(false));
    }

    #[test]
    fn parses_empty_containers() {
        assert_eq!(ok("[]"), JsonValue::Array(vec![]));
        assert_eq!(ok("{}"), JsonValue::Object(vec![]));
        assert_eq!(ok("[ ]").as_array().expect("array").len(), 0);
        assert_eq!(ok("{ }").as_object().expect("object").len(), 0);
    }

    #[test]
    fn parses_plain_string() {
        assert_eq!(ok("\"hello world\"").as_str(), Some("hello world"));
        assert_eq!(ok("\"\"").as_str(), Some(""));
    }

    #[test]
    fn parses_non_ascii_string_verbatim() {
        // Multi-byte UTF-8 must survive the byte-level scanner untouched.
        let src = "\"\u{043F}\u{0440}\u{0438} \u{4E2D} \u{1F600}\"";
        assert_eq!(ok(src).as_str(), Some("\u{043F}\u{0440}\u{0438} \u{4E2D} \u{1F600}"));
    }

    // --- numbers -------------------------------------------------------------------------

    #[test]
    fn parses_number_forms() {
        assert_eq!(ok("0").as_f64(), Some(0.0));
        assert_eq!(ok("-0").as_f64(), Some(-0.0));
        assert_eq!(ok("-1").as_f64(), Some(-1.0));
        assert_eq!(ok("42").as_f64(), Some(42.0));
        assert_eq!(ok("1.5").as_f64(), Some(1.5));
        assert_eq!(ok("1e10").as_f64(), Some(1e10));
        assert_eq!(ok("1E10").as_f64(), Some(1e10));
        assert_eq!(ok("1e+10").as_f64(), Some(1e10));
        assert_eq!(ok("-2.5e-3").as_f64(), Some(-2.5e-3));
        assert_eq!(ok("0.0").as_f64(), Some(0.0));
        assert_eq!(ok("-0.125").as_f64(), Some(-0.125));
        assert_eq!(ok("123456789012345").as_f64(), Some(123456789012345.0));
    }

    #[test]
    fn rejects_malformed_numbers() {
        for input in ["01", "-", "-x", ".5", "1.", "1e", "1e+", "1.2.3", "+1", "0.", "--1", "1e--2"]
        {
            assert!(parse(input).is_err(), "`{input}` should be rejected");
        }
    }

    #[test]
    fn rejects_non_finite_numbers() {
        // Overflow to infinity would silently corrupt downstream math.
        let message = err("1e999");
        assert!(message.contains("finite"), "message should explain the overflow: {message}");
    }

    #[test]
    fn rejects_numbers_outside_f32_range_at_parse_time() {
        // Finite as f64, infinite as f32. Rejected by the *parser*, not the
        // accessor, so a bad literal in a scene file fails with a byte offset
        // instead of surfacing as "optional field silently used its default".
        for input in ["1e300", "-1e300", "1e39", "3.5e38"] {
            let message = err(input);
            assert!(
                message.contains("offset") && message.contains("f32"),
                "`{input}` should be rejected with an offset and a reason: {message}"
            );
        }
        // The offset must point at the number, not the start of the document.
        let message = err(r#"{ "fov": 1e300 }"#);
        assert!(message.contains("offset 9"), "wrong offset: {message}");
    }

    #[test]
    fn numeric_accessors_narrow_correctly() {
        assert_eq!(ok("1.5").as_f32(), Some(1.5f32));
        assert_eq!(ok("7").as_u32(), Some(7));
        assert_eq!(ok("7.5").as_u32(), None, "non-integral must not truncate silently");
        assert_eq!(ok("-1").as_u32(), None, "negative must not wrap");
        assert_eq!(ok("1e20").as_u32(), None, "out of range must be rejected");
        assert_eq!(ok("\"7\"").as_f64(), None, "strings are not numbers");
        assert_eq!(ok("true").as_f64(), None);
        assert_eq!(ok("4294967295").as_u32(), Some(u32::MAX), "the exact upper bound fits");
        assert_eq!(ok("4294967296").as_u32(), None, "one past the upper bound does not");
    }

    #[test]
    fn as_f32_rejects_values_outside_f32_range() {
        // The parser now rejects these at parse time (see
        // `rejects_numbers_outside_f32_range_at_parse_time`), so the only way such
        // values reach the accessor is a hand-constructed document — which must
        // still not smuggle an infinity through.
        assert_eq!(JsonValue::Number(1e300).as_f32(), None);
        assert_eq!(JsonValue::Number(-1e300).as_f32(), None);
        assert_eq!(JsonValue::Number(f64::NAN).as_f32(), None);
        assert_eq!(
            JsonValue::Number(1e300).as_f64(),
            Some(1e300),
            "as_f64 still reports it faithfully"
        );
        // The rejection propagates through the aggregate accessors a scene loader uses.
        let vec = JsonValue::Array(vec![
            JsonValue::Number(1e300),
            JsonValue::Number(0.0),
            JsonValue::Number(0.0),
        ]);
        assert_eq!(vec.get_vec3_array(), None);

        // Values that merely lose precision stay accepted.
        assert_eq!(ok("1e-300").as_f32(), Some(0.0), "underflow to zero is ordinary");
        assert_eq!(ok("3.4e38").as_f32().map(f32::is_finite), Some(true), "inside f32 range");
        assert_eq!(ok("[1, 2, 3]").get_vec3_array(), Some([1.0, 2.0, 3.0]));
    }

    #[test]
    fn skips_a_single_leading_utf8_bom() {
        assert_eq!(ok("\u{FEFF}{\"a\": 1}").get_f64("a"), Some(1.0));
        assert_eq!(ok("\u{FEFF} [1]").as_array().map(|a| a.len()), Some(1));
        assert_eq!(ok("\u{FEFF}null"), JsonValue::Null);

        assert!(parse("\u{FEFF}").is_err(), "a BOM alone is not a document");
        assert!(parse("\u{FEFF}\u{FEFF}1").is_err(), "only the first BOM is tolerated");
        assert!(parse("[\u{FEFF}1]").is_err(), "a BOM inside the document is not whitespace");
        assert!(parse("1\u{FEFF}").is_err(), "a trailing BOM is still trailing content");

        // Offsets must keep counting the BOM bytes so they still index into the real file.
        let message = err("\u{FEFF}x");
        assert!(message.contains("offset 3"), "offset must include the BOM bytes: {message}");
    }

    // --- escapes -------------------------------------------------------------------------

    #[test]
    fn parses_all_simple_escapes() {
        let value = ok(r#""\"\\\/\b\f\n\r\t""#);
        assert_eq!(value.as_str(), Some("\"\\/\u{0008}\u{000C}\n\r\t"));
    }

    #[test]
    fn parses_unicode_escapes() {
        assert_eq!(ok(&quoted(&uesc("0041"))).as_str(), Some("A"));
        assert_eq!(ok(&quoted(&uesc("00e9"))).as_str(), Some("\u{00E9}"));
        assert_eq!(ok(&quoted(&uesc("002F"))).as_str(), Some("/"), "uppercase hex");
        assert_eq!(ok(&quoted(&uesc("0000"))).as_str(), Some("\u{0000}"));
        assert_eq!(ok(&quoted(&uesc("07FF"))).as_str(), Some("\u{07FF}"), "2-byte boundary");
        assert_eq!(ok(&quoted(&uesc("0800"))).as_str(), Some("\u{0800}"), "3-byte boundary");

        let two = format!("{}{}", uesc("4e2d"), uesc("6587"));
        assert_eq!(ok(&quoted(&two)).as_str(), Some("\u{4E2D}\u{6587}"));

        // Escapes interleaved with literal text.
        let mixed = format!("a{}c", uesc("0062"));
        assert_eq!(ok(&quoted(&mixed)).as_str(), Some("abc"));
    }

    #[test]
    fn parses_surrogate_pairs() {
        let grin = format!("{}{}", uesc("D83D"), uesc("DE00"));
        assert_eq!(ok(&quoted(&grin)).as_str(), Some("\u{1F600}"));

        let grin_lower = format!("{}{}", uesc("d83d"), uesc("de00"));
        assert_eq!(ok(&quoted(&grin_lower)).as_str(), Some("\u{1F600}"), "lowercase hex");

        let embedded = format!("x{}{}y", uesc("D83D"), uesc("DE00"));
        assert_eq!(ok(&quoted(&embedded)).as_str(), Some("x\u{1F600}y"));

        // Both ends of the supplementary planes.
        let lowest = format!("{}{}", uesc("D800"), uesc("DC00"));
        assert_eq!(ok(&quoted(&lowest)).as_str(), Some("\u{10000}"));
        let highest = format!("{}{}", uesc("DBFF"), uesc("DFFF"));
        assert_eq!(ok(&quoted(&highest)).as_str(), Some("\u{10FFFF}"));
    }

    #[test]
    fn rejects_bad_escapes() {
        assert!(parse("\"\\q\"").is_err(), "unknown escape letter");
        assert!(parse("\"\\N\"").is_err(), "unknown escape letter");
        assert!(parse("\"\\\"").is_err(), "escaped quote leaves the string unterminated");
        assert!(parse("\"\\").is_err(), "backslash at end of input");
        assert!(parse("\"\\u\"").is_err(), "unicode escape without digits");
        assert!(parse(&quoted(&uesc("12"))).is_err(), "too few hex digits");
        assert!(parse(&quoted(&uesc("ZZZZ"))).is_err(), "non-hex digits");
        assert!(parse(&quoted(&uesc("00G1"))).is_err(), "partially non-hex");
    }

    #[test]
    fn rejects_lone_surrogates() {
        assert!(parse(&quoted(&uesc("D800"))).is_err(), "lone high surrogate");
        assert!(parse(&quoted(&uesc("DBFF"))).is_err(), "lone high surrogate at range end");
        assert!(parse(&quoted(&uesc("DC00"))).is_err(), "lone low surrogate");
        assert!(parse(&quoted(&uesc("DFFF"))).is_err(), "lone low surrogate at range end");

        let then_text = quoted(&format!("{}x", uesc("D800")));
        assert!(parse(&then_text).is_err(), "high surrogate followed by plain text");

        let then_bmp = quoted(&format!("{}{}", uesc("D800"), uesc("0041")));
        assert!(parse(&then_bmp).is_err(), "high surrogate followed by a non-surrogate escape");

        let two_highs = quoted(&format!("{}{}", uesc("D800"), uesc("D800")));
        assert!(parse(&two_highs).is_err(), "two high surrogates in a row");

        let truncated = quoted(&format!("{}\\u", uesc("D800")));
        assert!(parse(&truncated).is_err(), "high surrogate then a truncated escape");
    }

    #[test]
    fn rejects_raw_control_characters_in_strings() {
        assert!(parse("\"a\nb\"").is_err(), "raw newline");
        assert!(parse("\"a\tb\"").is_err(), "raw tab");
        assert!(parse("\"a\u{0000}b\"").is_err(), "raw NUL");
    }

    // --- structure -----------------------------------------------------------------------

    #[test]
    fn parses_flat_array() {
        let value = ok("[1, 2, 3]");
        assert_eq!(value.as_array().expect("array").len(), 3);
        assert_eq!(value.index(0).and_then(JsonValue::as_f64), Some(1.0));
        assert_eq!(value.index(2).and_then(JsonValue::as_f64), Some(3.0));
        assert_eq!(value.index(3), None, "out of bounds");
    }

    #[test]
    fn parses_heterogeneous_array() {
        let value = ok("[null, true, 1, \"s\", [], {}]");
        let items = value.as_array().expect("array");
        assert_eq!(items.len(), 6);
        assert!(items[0].is_null());
        assert_eq!(items[1].as_bool(), Some(true));
        assert_eq!(items[2].as_f64(), Some(1.0));
        assert_eq!(items[3].as_str(), Some("s"));
        assert_eq!(items[4].type_name(), "array");
        assert_eq!(items[5].type_name(), "object");
    }

    #[test]
    fn parses_nested_object_graph() {
        let src = r#"{
            "name": "cornell",
            "camera": { "position": [0.0, 1.0, -3.5], "fov": 60, "perspective": true },
            "entities": [
                { "mesh": "sphere", "segments": 32,
                  "material": { "type": "mirror", "reflectivity": 0.5 } },
                { "mesh": "plane",
                  "material": { "type": "matte", "color": [1, 1, 1, 1] } }
            ]
        }"#;
        let doc = ok(src);
        assert_eq!(doc.get_str("name"), Some("cornell"));
        assert_eq!(doc.as_object().expect("object").len(), 3);

        let camera = doc.get("camera").expect("camera");
        assert_eq!(camera.get_vec3("position"), Some([0.0, 1.0, -3.5]));
        assert_eq!(camera.get_f32("fov"), Some(60.0));
        assert_eq!(camera.get_bool("perspective"), Some(true));
        assert_eq!(camera.get_vec3("missing"), None);

        let entities = doc.get_array("entities").expect("entities");
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].get_str("mesh"), Some("sphere"));
        assert_eq!(entities[0].get_u32("segments"), Some(32));
        assert_eq!(
            entities[0].get("material").and_then(|m| m.get_f32("reflectivity")),
            Some(0.5)
        );
        assert_eq!(entities[1].get("material").and_then(|m| m.get_str("type")), Some("matte"));
        assert_eq!(entities[1].get("material").and_then(|m| m.get_vec4("color")), Some([1.0; 4]));
        assert_eq!(doc.get("missing"), None);
    }

    #[test]
    fn object_preserves_order_and_first_duplicate_wins() {
        let doc = ok(r#"{ "z": 1, "a": 2, "z": 3 }"#);
        let pairs = doc.as_object().expect("object");
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "z");
        assert_eq!(pairs[1].0, "a");
        assert_eq!(pairs[2].0, "z");
        assert_eq!(doc.get_f64("z"), Some(1.0), "lookup returns the first occurrence");
    }

    #[test]
    fn accessors_return_none_on_type_mismatch() {
        let object = ok(r#"{ "a": 1 }"#);
        assert_eq!(object.as_array(), None);
        assert_eq!(object.as_str(), None);
        assert_eq!(object.as_bool(), None);
        assert_eq!(object.index(0), None);
        assert_eq!(object.get_str("a"), None, "number is not a string");

        let array = ok("[1, 2, 3]");
        assert_eq!(array.get("a"), None);
        assert_eq!(array.as_object(), None);
        assert_eq!(array.get_vec3_array(), Some([1.0, 2.0, 3.0]));

        assert_eq!(ok("[1, 2]").get_vec3_array(), None, "too few components");
        assert_eq!(ok("[1, 2, 3, 4]").get_vec3_array(), None, "too many components");
        assert_eq!(ok(r#"[1, "x", 3]"#).get_vec3_array(), None, "non-numeric component");
        assert_eq!(ok("[1, 2, 3, 4]").get_vec4_array(), Some([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(ok("null").get_vec3_array(), None);
    }

    #[test]
    fn type_name_reports_every_kind() {
        assert_eq!(ok("null").type_name(), "null");
        assert_eq!(ok("true").type_name(), "bool");
        assert_eq!(ok("1").type_name(), "number");
        assert_eq!(ok("\"s\"").type_name(), "string");
        assert_eq!(ok("[]").type_name(), "array");
        assert_eq!(ok("{}").type_name(), "object");
        assert!(!ok("0").is_null());
    }

    // --- whitespace ----------------------------------------------------------------------

    #[test]
    fn tolerates_arbitrary_whitespace() {
        let src = " \t\r\n { \n \"a\" \t : \r\n [ 1 , 2 ] \n } \t\r\n ";
        let doc = ok(src);
        assert_eq!(doc.get_array("a").map(|a| a.len()), Some(2));
        assert_eq!(ok("\n\t 7 \n").as_f64(), Some(7.0));
        assert_eq!(ok(" [ ] ").type_name(), "array");
    }

    // --- malformed input -----------------------------------------------------------------

    #[test]
    fn rejects_empty_and_whitespace_only_input() {
        assert!(parse("").is_err());
        assert!(parse("   \n\t ").is_err());
        let message = err("");
        assert!(message.contains("end of input"), "message should say why: {message}");
    }

    #[test]
    fn rejects_unterminated_string() {
        assert!(parse("\"abc").is_err());
        assert!(parse("\"").is_err());
        assert!(parse(r#"{ "key": "value }"#).is_err());
        let message = err("\"abc");
        assert!(message.contains("closing"), "message should name the cause: {message}");
    }

    #[test]
    fn rejects_unterminated_containers() {
        for input in
            ["[", "[1", "[1,", "[[1]", "[1 2]", "{", "{\"a\"", "{\"a\":", "{\"a\":1", "{\"a\":1,"]
        {
            assert!(parse(input).is_err(), "`{input}` should be rejected");
        }
    }

    #[test]
    fn rejects_trailing_commas() {
        assert!(parse("[1,]").is_err());
        assert!(parse("[,]").is_err());
        assert!(parse("[1,,2]").is_err());
        assert!(parse(r#"{"a":1,}"#).is_err());
        assert!(parse(r#"{,}"#).is_err());
    }

    #[test]
    fn rejects_trailing_content() {
        for input in ["1 2", "{} {}", "null x", "[] []", "\"a\" \"b\"", "truefalse", "[1][2]"] {
            assert!(parse(input).is_err(), "`{input}` should be rejected");
        }
    }

    #[test]
    fn rejects_broken_object_syntax() {
        for input in [r#"{a: 1}"#, r#"{"a" 1}"#, r#"{"a": }"#, r#"{1: 2}"#, r#"{"a"::1}"#] {
            assert!(parse(input).is_err(), "`{input}` should be rejected");
        }
    }

    #[test]
    fn rejects_partial_literals() {
        for input in ["nul", "tru", "fals", "None", "NULL", "n", "undefined"] {
            assert!(parse(input).is_err(), "`{input}` should be rejected");
        }
    }

    #[test]
    fn errors_carry_offset_and_expectation() {
        // The unexpected `"` sits at index 8: `{"a": 1 "b": 2}`.
        let message = err(r#"{"a": 1 "b": 2}"#);
        assert!(message.contains("offset"), "message should locate the failure: {message}");
        assert!(message.contains("expected"), "message should state expectation: {message}");
        assert!(message.contains("offset 8"), "wrong offset reported: {message}");
    }

    // --- depth limit ---------------------------------------------------------------------

    #[test]
    fn accepts_nesting_up_to_the_limit() {
        assert!(parse(&nested('[', ']', MAX_DEPTH - 1)).is_ok());
        assert!(parse(&nested('[', ']', MAX_DEPTH)).is_ok());
        assert!(parse(&mixed_nest(MAX_DEPTH)).is_ok());
    }

    #[test]
    fn rejects_nesting_past_the_limit() {
        let src = nested('[', ']', MAX_DEPTH + 1);
        let message = match parse(&src) {
            Ok(_) => panic!("depth {} should be rejected", MAX_DEPTH + 1),
            Err(e) => e.to_string(),
        };
        assert!(message.contains("depth"), "message should mention depth: {message}");
        assert!(parse(&mixed_nest(MAX_DEPTH + 1)).is_err(), "arrays and objects share the budget");
    }

    #[test]
    fn survives_pathologically_deep_input_without_overflowing() {
        // Errors out at the cap instead of recursing a hundred thousand frames deep.
        assert!(parse(&nested('[', ']', 100_000)).is_err());

        let mut deep_objects = String::new();
        for _ in 0..100_000 {
            deep_objects.push_str("{\"a\":");
        }
        assert!(parse(&deep_objects).is_err());
    }
}
