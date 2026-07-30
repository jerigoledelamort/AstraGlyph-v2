// Lua lexer: source text to tokens.
//
// Self-implemented per the "no external crates" rule — no mlua, no rlua.
//
// Scope: the Lua 5.1 lexical grammar minus the parts a game-logic script does not
// need. What is here: all keywords and operators, decimal and hex numbers,
// single/double-quoted strings with escapes, long strings and long comments
// (`[[...]]`, `[==[...]==]`), line comments. What is not: string coercion of
// numeric literals with exponents in odd bases, and `goto`/labels (Lua 5.2+).
//
// Every token carries its line number. A script error that says "line 12" is
// actionable; one that says "syntax error" is a bug report with no information in
// it, and hot-reloaded scripts are edited constantly.

use crate::engine::core::{EngineError, Result};

/// A lexical token.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    // Literals
    Number(f64),
    Str(String),
    Name(String),

    // Keywords
    And,
    Break,
    Do,
    Else,
    Elseif,
    End,
    False,
    For,
    Function,
    If,
    In,
    Local,
    Nil,
    Not,
    Or,
    Repeat,
    Return,
    Then,
    True,
    Until,
    While,

    // Operators and punctuation
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Hash,
    Equal,
    NotEqual,
    LessEqual,
    GreaterEqual,
    Less,
    Greater,
    Assign,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semicolon,
    Colon,
    Comma,
    Dot,
    Concat,
    Ellipsis,

    /// End of input.
    Eof,
}

impl Token {
    /// Human-readable name, for error messages.
    pub fn describe(&self) -> String {
        match self {
            Self::Number(n) => format!("number {n}"),
            Self::Str(s) => format!("string {s:?}"),
            Self::Name(n) => format!("name {n:?}"),
            Self::Eof => "end of input".to_string(),
            other => format!("{}", keyword_text(other).unwrap_or("token")),
        }
    }
}

/// Source text of a fixed token, for error messages.
fn keyword_text(token: &Token) -> Option<&'static str> {
    Some(match token {
        Token::And => "and",
        Token::Break => "break",
        Token::Do => "do",
        Token::Else => "else",
        Token::Elseif => "elseif",
        Token::End => "end",
        Token::False => "false",
        Token::For => "for",
        Token::Function => "function",
        Token::If => "if",
        Token::In => "in",
        Token::Local => "local",
        Token::Nil => "nil",
        Token::Not => "not",
        Token::Or => "or",
        Token::Repeat => "repeat",
        Token::Return => "return",
        Token::Then => "then",
        Token::True => "true",
        Token::Until => "until",
        Token::While => "while",
        Token::Plus => "+",
        Token::Minus => "-",
        Token::Star => "*",
        Token::Slash => "/",
        Token::Percent => "%",
        Token::Caret => "^",
        Token::Hash => "#",
        Token::Equal => "==",
        Token::NotEqual => "~=",
        Token::LessEqual => "<=",
        Token::GreaterEqual => ">=",
        Token::Less => "<",
        Token::Greater => ">",
        Token::Assign => "=",
        Token::LParen => "(",
        Token::RParen => ")",
        Token::LBrace => "{",
        Token::RBrace => "}",
        Token::LBracket => "[",
        Token::RBracket => "]",
        Token::Semicolon => ";",
        Token::Colon => ":",
        Token::Comma => ",",
        Token::Dot => ".",
        Token::Concat => "..",
        Token::Ellipsis => "...",
        _ => return None,
    })
}

/// A token and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Spanned {
    pub token: Token,
    pub line: u32,
}

fn err(line: u32, msg: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidState(format!("line {line}: {msg}"))
}

/// Tokenize Lua source.
pub fn tokenize(source: &str) -> Result<Vec<Spanned>> {
    // Chars, not bytes: a string literal or comment may contain any UTF-8, and
    // indexing bytes would split a multi-byte character.
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let mut line = 1u32;

    while i < chars.len() {
        let c = chars[i];

        // Whitespace.
        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Comments. Checked before the `-` operator, since `--` starts a comment
        // and `-` is subtraction.
        if c == '-' && chars.get(i + 1) == Some(&'-') {
            i += 2;
            // A long comment is `--[[ ... ]]` or `--[==[ ... ]==]`.
            if let Some(level) = long_bracket_level(&chars, i) {
                let (_, next, lines) = read_long_bracket(&chars, i, level, line)?;
                i = next;
                line += lines;
            } else {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            continue;
        }

        // Long strings.
        if c == '[' {
            if let Some(level) = long_bracket_level(&chars, i) {
                let (text, next, lines) = read_long_bracket(&chars, i, level, line)?;
                tokens.push(Spanned {
                    token: Token::Str(text),
                    line,
                });
                i = next;
                line += lines;
                continue;
            }
        }

        // Names and keywords.
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            tokens.push(Spanned {
                token: keyword_or_name(word),
                line,
            });
            continue;
        }

        // Numbers. A leading '.' followed by a digit is also a number (`.5`).
        if c.is_ascii_digit() || (c == '.' && matches!(chars.get(i + 1), Some(d) if d.is_ascii_digit()))
        {
            let (value, next) = read_number(&chars, i, line)?;
            tokens.push(Spanned {
                token: Token::Number(value),
                line,
            });
            i = next;
            continue;
        }

        // Quoted strings.
        if c == '"' || c == '\'' {
            let (text, next, lines) = read_quoted(&chars, i, c, line)?;
            tokens.push(Spanned {
                token: Token::Str(text),
                line,
            });
            i = next;
            line += lines;
            continue;
        }

        // Operators. Longest match first, or `==` would lex as two `=`.
        let (token, width) = match (c, chars.get(i + 1), chars.get(i + 2)) {
            ('.', Some('.'), Some('.')) => (Token::Ellipsis, 3),
            ('.', Some('.'), _) => (Token::Concat, 2),
            ('=', Some('='), _) => (Token::Equal, 2),
            ('~', Some('='), _) => (Token::NotEqual, 2),
            ('<', Some('='), _) => (Token::LessEqual, 2),
            ('>', Some('='), _) => (Token::GreaterEqual, 2),
            ('+', _, _) => (Token::Plus, 1),
            ('-', _, _) => (Token::Minus, 1),
            ('*', _, _) => (Token::Star, 1),
            ('/', _, _) => (Token::Slash, 1),
            ('%', _, _) => (Token::Percent, 1),
            ('^', _, _) => (Token::Caret, 1),
            ('#', _, _) => (Token::Hash, 1),
            ('<', _, _) => (Token::Less, 1),
            ('>', _, _) => (Token::Greater, 1),
            ('=', _, _) => (Token::Assign, 1),
            ('(', _, _) => (Token::LParen, 1),
            (')', _, _) => (Token::RParen, 1),
            ('{', _, _) => (Token::LBrace, 1),
            ('}', _, _) => (Token::RBrace, 1),
            ('[', _, _) => (Token::LBracket, 1),
            (']', _, _) => (Token::RBracket, 1),
            (';', _, _) => (Token::Semicolon, 1),
            (':', _, _) => (Token::Colon, 1),
            (',', _, _) => (Token::Comma, 1),
            ('.', _, _) => (Token::Dot, 1),
            _ => return Err(err(line, format!("unexpected character {c:?}"))),
        };
        tokens.push(Spanned { token, line });
        i += width;
    }

    tokens.push(Spanned {
        token: Token::Eof,
        line,
    });
    Ok(tokens)
}

fn keyword_or_name(word: String) -> Token {
    match word.as_str() {
        "and" => Token::And,
        "break" => Token::Break,
        "do" => Token::Do,
        "else" => Token::Else,
        "elseif" => Token::Elseif,
        "end" => Token::End,
        "false" => Token::False,
        "for" => Token::For,
        "function" => Token::Function,
        "if" => Token::If,
        "in" => Token::In,
        "local" => Token::Local,
        "nil" => Token::Nil,
        "not" => Token::Not,
        "or" => Token::Or,
        "repeat" => Token::Repeat,
        "return" => Token::Return,
        "then" => Token::Then,
        "true" => Token::True,
        "until" => Token::Until,
        "while" => Token::While,
        _ => Token::Name(word),
    }
}

/// If a long bracket opens at `i`, its level: `[[` is 0, `[=[` is 1, and so on.
///
/// The level exists so a long string can contain `]]` — which is exactly what
/// happens when a script embeds another script or a chunk of Lua-ish text.
fn long_bracket_level(chars: &[char], i: usize) -> Option<usize> {
    if chars.get(i) != Some(&'[') {
        return None;
    }
    let mut level = 0;
    let mut j = i + 1;
    while chars.get(j) == Some(&'=') {
        level += 1;
        j += 1;
    }
    if chars.get(j) == Some(&'[') {
        Some(level)
    } else {
        None
    }
}

/// Read a long bracket, returning its contents, the index after it, and how many
/// newlines it spanned.
fn read_long_bracket(
    chars: &[char],
    open: usize,
    level: usize,
    line: u32,
) -> Result<(String, usize, u32)> {
    // Skip `[`, the `=`s and the second `[`.
    let mut i = open + 2 + level;
    // A newline immediately after the opening bracket is skipped, per the Lua
    // manual — so `[[\nfoo]]` is "foo", not "\nfoo".
    let mut newlines = 0;
    if chars.get(i) == Some(&'\n') {
        i += 1;
        newlines += 1;
    }
    let start = i;
    while i < chars.len() {
        if chars[i] == ']' {
            // A closing bracket of the *same* level, and no other.
            let mut j = i + 1;
            let mut found = 0;
            while chars.get(j) == Some(&'=') {
                found += 1;
                j += 1;
            }
            if found == level && chars.get(j) == Some(&']') {
                let text: String = chars[start..i].iter().collect();
                return Ok((text, j + 1, newlines));
            }
        }
        if chars[i] == '\n' {
            newlines += 1;
        }
        i += 1;
    }
    Err(err(line, "unterminated long string or comment"))
}

/// Read a decimal or hexadecimal number literal.
fn read_number(chars: &[char], start: usize, line: u32) -> Result<(f64, usize)> {
    let mut i = start;

    // Hexadecimal.
    if chars[i] == '0' && matches!(chars.get(i + 1), Some('x') | Some('X')) {
        i += 2;
        let digits_start = i;
        while i < chars.len() && chars[i].is_ascii_hexdigit() {
            i += 1;
        }
        if i == digits_start {
            return Err(err(line, "malformed hexadecimal number"));
        }
        let text: String = chars[digits_start..i].iter().collect();
        let value = u64::from_str_radix(&text, 16)
            .map_err(|_| err(line, format!("hexadecimal number 0x{text} does not fit")))?;
        return Ok((value as f64, i));
    }

    // Decimal, with an optional fraction and exponent.
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if chars.get(i) == Some(&'.') {
        i += 1;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
    }
    if matches!(chars.get(i), Some('e') | Some('E')) {
        let exponent_start = i;
        i += 1;
        if matches!(chars.get(i), Some('+') | Some('-')) {
            i += 1;
        }
        if !matches!(chars.get(i), Some(d) if d.is_ascii_digit()) {
            // `1e` with no digits is not an exponent. Back out rather than fail:
            // `1e` followed by a name is a syntax error at the parser level, and
            // reporting it there gives a better message.
            i = exponent_start;
        } else {
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
        }
    }

    let text: String = chars[start..i].iter().collect();
    let value = text
        .parse::<f64>()
        .map_err(|_| err(line, format!("malformed number {text:?}")))?;
    Ok((value, i))
}

/// Read a single- or double-quoted string, applying escapes.
fn read_quoted(chars: &[char], start: usize, quote: char, line: u32) -> Result<(String, usize, u32)> {
    let mut out = String::new();
    let mut i = start + 1;
    let mut newlines = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == quote {
            return Ok((out, i + 1, newlines));
        }
        if c == '\n' {
            // An unescaped newline inside a short string is an error in Lua, and
            // reporting it here catches a missing closing quote at the line it
            // was forgotten rather than at the end of the file.
            return Err(err(line, "unterminated string (newline in a short string)"));
        }
        if c == '\\' {
            i += 1;
            let escape = *chars
                .get(i)
                .ok_or_else(|| err(line, "unterminated escape sequence"))?;
            match escape {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'a' => out.push('\x07'),
                'b' => out.push('\x08'),
                'f' => out.push('\x0C'),
                'v' => out.push('\x0B'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                '\'' => out.push('\''),
                '\n' => {
                    out.push('\n');
                    newlines += 1;
                }
                // `\ddd`: up to three decimal digits, a byte value.
                d if d.is_ascii_digit() => {
                    let mut value = 0u32;
                    let mut digits = 0;
                    while digits < 3 {
                        match chars.get(i) {
                            Some(c) if c.is_ascii_digit() => {
                                value = value * 10 + c.to_digit(10).unwrap_or(0);
                                i += 1;
                                digits += 1;
                            }
                            _ => break,
                        }
                    }
                    // Step back one: the loop below advances past the last digit.
                    i -= 1;
                    if value > 255 {
                        return Err(err(line, format!("escape \\{value} is out of range")));
                    }
                    out.push(value as u8 as char);
                }
                other => return Err(err(line, format!("unknown escape sequence \\{other}"))),
            }
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    Err(err(line, "unterminated string"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_of(source: &str) -> Vec<Token> {
        tokenize(source)
            .expect("should tokenize")
            .into_iter()
            .map(|s| s.token)
            .collect()
    }

    #[test]
    fn tokenizes_an_assignment() {
        assert_eq!(
            tokens_of("local x = 42"),
            vec![
                Token::Local,
                Token::Name("x".into()),
                Token::Assign,
                Token::Number(42.0),
                Token::Eof
            ]
        );
    }

    /// Longest-match first. `==` lexing as two `=` would silently turn every
    /// comparison into a chain of assignments.
    #[test]
    fn multi_character_operators_win_over_single_ones() {
        assert_eq!(
            tokens_of("a == b ~= c <= d >= e .. f ..."),
            vec![
                Token::Name("a".into()),
                Token::Equal,
                Token::Name("b".into()),
                Token::NotEqual,
                Token::Name("c".into()),
                Token::LessEqual,
                Token::Name("d".into()),
                Token::GreaterEqual,
                Token::Name("e".into()),
                Token::Concat,
                Token::Name("f".into()),
                Token::Ellipsis,
                Token::Eof
            ]
        );
    }

    #[test]
    fn keywords_are_distinct_from_names() {
        assert_eq!(tokens_of("if")[0], Token::If);
        // A keyword is only a keyword when it is the whole word.
        assert_eq!(tokens_of("iffy")[0], Token::Name("iffy".into()));
        assert_eq!(tokens_of("_end")[0], Token::Name("_end".into()));
        assert_eq!(tokens_of("end2")[0], Token::Name("end2".into()));
    }

    #[test]
    fn numbers_cover_decimals_fractions_exponents_and_hex() {
        assert_eq!(tokens_of("42")[0], Token::Number(42.0));
        assert_eq!(tokens_of("3.5")[0], Token::Number(3.5));
        assert_eq!(tokens_of(".5")[0], Token::Number(0.5));
        assert_eq!(tokens_of("1e3")[0], Token::Number(1000.0));
        assert_eq!(tokens_of("1.5e-2")[0], Token::Number(0.015));
        assert_eq!(tokens_of("0xff")[0], Token::Number(255.0));
        assert_eq!(tokens_of("0X10")[0], Token::Number(16.0));
    }

    /// `--` starts a comment, so it must be tested before `-` as subtraction.
    /// Getting this wrong turns every comment into an expression.
    #[test]
    fn a_line_comment_is_not_two_minus_signs() {
        assert_eq!(
            tokens_of("local a = 1 -- a comment with - and -- inside\nlocal b = 2"),
            vec![
                Token::Local,
                Token::Name("a".into()),
                Token::Assign,
                Token::Number(1.0),
                Token::Local,
                Token::Name("b".into()),
                Token::Assign,
                Token::Number(2.0),
                Token::Eof
            ]
        );
        // Actual subtraction still works.
        assert_eq!(
            tokens_of("a - b"),
            vec![
                Token::Name("a".into()),
                Token::Minus,
                Token::Name("b".into()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn a_long_comment_spans_lines() {
        assert_eq!(
            tokens_of("--[[ this\nspans\nlines ]] local x"),
            vec![Token::Local, Token::Name("x".into()), Token::Eof]
        );
    }

    #[test]
    fn strings_apply_escapes() {
        assert_eq!(tokens_of(r#""a\nb""#)[0], Token::Str("a\nb".into()));
        assert_eq!(tokens_of(r#""tab\there""#)[0], Token::Str("tab\there".into()));
        assert_eq!(tokens_of(r#""quote\"inside""#)[0], Token::Str("quote\"inside".into()));
        assert_eq!(tokens_of(r"'single'")[0], Token::Str("single".into()));
        assert_eq!(tokens_of(r#""\65""#)[0], Token::Str("A".into()));
    }

    #[test]
    fn a_long_string_keeps_its_contents_verbatim() {
        assert_eq!(
            tokens_of("[[no \\n escapes here]]")[0],
            Token::Str("no \\n escapes here".into())
        );
    }

    /// The level exists so a long string can contain `]]`, which happens whenever
    /// a script embeds another script.
    #[test]
    fn a_levelled_long_string_can_contain_closing_brackets() {
        assert_eq!(
            tokens_of("[==[ contains ]] inside ]==]")[0],
            Token::Str(" contains ]] inside ".into())
        );
    }

    /// Per the Lua manual, a newline directly after `[[` is skipped — so a
    /// multi-line string does not begin with a blank line.
    #[test]
    fn a_long_string_skips_a_leading_newline() {
        assert_eq!(tokens_of("[[\nfoo]]")[0], Token::Str("foo".into()));
    }

    /// Line numbers must be right, or a hot-reload error message points at the
    /// wrong line and sends the author looking in the wrong place.
    #[test]
    fn line_numbers_survive_comments_and_multi_line_strings() {
        let spans = tokenize("local a\n-- comment\n\nlocal b\n[[one\ntwo]]\nlocal c").unwrap();
        let line_of = |name: &str| {
            spans
                .iter()
                .find(|s| s.token == Token::Name(name.into()))
                .map(|s| s.line)
                .unwrap_or(0)
        };
        assert_eq!(line_of("a"), 1);
        assert_eq!(line_of("b"), 4);
        assert_eq!(line_of("c"), 7, "the long string's newline must be counted");
    }

    #[test]
    fn an_unterminated_string_is_an_error_naming_its_line() {
        let e = tokenize("local a = 1\nlocal s = \"oops").unwrap_err();
        assert!(e.to_string().contains("line 2"), "{e}");
    }

    /// A missing closing quote should be reported where the quote was forgotten,
    /// not at the end of the file — which is what allowing a raw newline would do.
    #[test]
    fn a_newline_inside_a_short_string_is_rejected() {
        assert!(tokenize("local s = \"unclosed\nlocal t = 1").is_err());
    }

    #[test]
    fn an_unterminated_long_string_is_an_error() {
        assert!(tokenize("local s = [[never closed").is_err());
        assert!(
            tokenize("local s = [==[ closed at the wrong level ]=]").is_err(),
            "a closing bracket of a different level must not terminate the string"
        );
    }

    #[test]
    fn an_unknown_escape_is_rejected() {
        assert!(tokenize(r#""\q""#).is_err());
    }

    #[test]
    fn an_unexpected_character_names_itself() {
        let e = tokenize("local a = $").unwrap_err();
        assert!(e.to_string().contains('$'), "{e}");
    }

    #[test]
    fn empty_input_is_just_eof() {
        assert_eq!(tokens_of(""), vec![Token::Eof]);
        assert_eq!(tokens_of("   \n\t\n  "), vec![Token::Eof]);
        assert_eq!(tokens_of("-- only a comment"), vec![Token::Eof]);
    }

    /// A script may legitimately contain non-ASCII text in a string. Byte-indexing
    /// would split a multi-byte character and produce garbage or a panic.
    #[test]
    fn multi_byte_characters_in_strings_survive() {
        assert_eq!(
            tokens_of("\"привет мир\"")[0],
            Token::Str("привет мир".into())
        );
        assert_eq!(tokens_of("[[日本語]]")[0], Token::Str("日本語".into()));
    }

    #[test]
    fn describe_produces_something_readable_for_every_token() {
        for token in [
            Token::Number(1.5),
            Token::Str("x".into()),
            Token::Name("y".into()),
            Token::If,
            Token::Concat,
            Token::Eof,
        ] {
            assert!(!token.describe().is_empty());
        }
    }
}
