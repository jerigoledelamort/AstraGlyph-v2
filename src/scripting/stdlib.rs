// Lua standard library subset.
//
// What a game script actually reaches for: `print`, `type`, `tostring`,
// `tonumber`, `pairs`, `ipairs`, `math`, `string`, `table`, `setmetatable` (for
// `__index` only). Deliberately not `io` or `os` — a hot-reloaded script is a
// file the user is editing, and giving it the ability to read the filesystem or
// spawn a process would make an editing mistake dangerous rather than annoying.
//
// `print` writes into the interpreter's own output buffer rather than to stdout,
// which is what makes it testable and what lets the in-engine console show script
// output alongside its own.

use std::cell::RefCell;
use std::rc::Rc;

use crate::scripting::interp::{compare, tostring, Interpreter};
use crate::scripting::value::{format_number, Key, NativeFunction, Table, Value};

/// Shared output sink, so `print` can reach the interpreter's buffer from inside a
/// closure that cannot borrow the interpreter.
///
/// A closure installed as a native function is `Fn(&[Value])` — it has no path
/// back to the `Interpreter` that owns it, and giving it one would mean handing
/// the interpreter a reference to itself. An `Rc<RefCell<Vec<String>>>` shared
/// between the two sides is the smaller answer.
type OutputSink = Rc<RefCell<Vec<String>>>;

/// Install the standard library into an interpreter.
pub fn install(interp: &mut Interpreter) {
    let sink: OutputSink = Rc::new(RefCell::new(Vec::new()));
    install_base(interp, sink.clone());
    install_math(interp);
    install_string(interp);
    install_table(interp);
    interp.attach_output_sink(sink);
}

fn native(name: &str, f: impl Fn(&[Value]) -> Result<Vec<Value>, String> + 'static) -> Value {
    Value::Native(Rc::new(NativeFunction {
        name: name.to_string(),
        function: Box::new(f),
    }))
}

/// Build a table from `(name, value)` pairs and return it as a value.
fn module(entries: Vec<(&str, Value)>) -> Value {
    let table = Rc::new(RefCell::new(Table::new()));
    for (name, value) in entries {
        table.borrow_mut().set(Key::Str(Rc::from(name)), value);
    }
    Value::Table(table)
}

fn arg(args: &[Value], index: usize) -> Value {
    args.get(index).cloned().unwrap_or(Value::Nil)
}

/// Read an argument as a number, with a message naming the position and function.
fn number_arg(args: &[Value], index: usize, function: &str) -> Result<f64, String> {
    arg(args, index).as_number().ok_or_else(|| {
        format!(
            "bad argument #{} to '{function}' (number expected, got {})",
            index + 1,
            arg(args, index).type_name()
        )
    })
}

fn string_arg(args: &[Value], index: usize, function: &str) -> Result<String, String> {
    arg(args, index).as_str_coerced().ok_or_else(|| {
        format!(
            "bad argument #{} to '{function}' (string expected, got {})",
            index + 1,
            arg(args, index).type_name()
        )
    })
}

fn table_arg(
    args: &[Value],
    index: usize,
    function: &str,
) -> Result<Rc<RefCell<Table>>, String> {
    match arg(args, index) {
        Value::Table(t) => Ok(t),
        other => Err(format!(
            "bad argument #{} to '{function}' (table expected, got {})",
            index + 1,
            other.type_name()
        )),
    }
}

// --- base ---

fn install_base(interp: &mut Interpreter, sink: OutputSink) {
    let print_sink = sink.clone();
    interp.set_global(
        "print",
        native("print", move |args| {
            // Tab-separated, like Lua's print.
            let line = args
                .iter()
                .map(tostring)
                .collect::<Vec<_>>()
                .join("\t");
            print_sink.borrow_mut().push(line);
            Ok(vec![])
        }),
    );

    interp.set_global(
        "type",
        native("type", |args| {
            Ok(vec![Value::str(arg(args, 0).type_name())])
        }),
    );

    interp.set_global(
        "tostring",
        native("tostring", |args| {
            Ok(vec![Value::str(tostring(&arg(args, 0)))])
        }),
    );

    interp.set_global(
        "tonumber",
        native("tonumber", |args| {
            // With a base, only a string is accepted and the digits are parsed in
            // that radix — which is the documented behaviour and the reason
            // tonumber takes a second argument at all.
            if let Some(base) = args.get(1).and_then(|v| v.as_number()) {
                let base = base as u32;
                if !(2..=36).contains(&base) {
                    return Err(format!("bad argument #2 to 'tonumber' (base out of range: {base})"));
                }
                let text = arg(args, 0).as_str_coerced().unwrap_or_default();
                return Ok(vec![match i64::from_str_radix(text.trim(), base) {
                    Ok(v) => Value::Number(v as f64),
                    Err(_) => Value::Nil,
                }]);
            }
            // Without a base, a failed conversion is nil rather than an error:
            // `tonumber(x) or default` is the standard idiom and it needs nil.
            Ok(vec![match arg(args, 0).as_number() {
                Some(n) => Value::Number(n),
                None => Value::Nil,
            }])
        }),
    );

    interp.set_global(
        "assert",
        native("assert", |args| {
            if arg(args, 0).truthy() {
                Ok(args.to_vec())
            } else {
                let message = args
                    .get(1)
                    .and_then(|v| v.as_str_coerced())
                    .unwrap_or_else(|| "assertion failed!".to_string());
                Err(message)
            }
        }),
    );

    interp.set_global(
        "error",
        native("error", |args| {
            Err(arg(args, 0)
                .as_str_coerced()
                .unwrap_or_else(|| tostring(&arg(args, 0))))
        }),
    );

    // `next(t, key)`: the primitive `pairs` is built on.
    interp.set_global("next", native("next", table_next));

    interp.set_global(
        "pairs",
        native("pairs", |args| {
            let table = table_arg(args, 0, "pairs")?;
            // The key list is snapshotted here, once, and the returned iterator
            // walks it by position.
            //
            // `next(t, key)` — which re-derives the key list on every call and
            // finds the successor of `key` — is the obvious implementation and it
            // is subtly broken over a HashMap: writing to an existing field inside
            // the loop can change the map's iteration order, so the next call
            // finds `key` at a different position and skips or repeats entries.
            // Measured on `{a=1,b=2,c=3}` with `t[k] = v * 10` in the body: the
            // sum came out 42 instead of 60, because `b` was never visited. Lua
            // explicitly permits assigning to an existing field during a `pairs`
            // loop, so this had to work.
            let keys = Rc::new(RefCell::new(table.borrow().keys()));
            let position = Rc::new(RefCell::new(0usize));
            let iterator = native("pairs_iterator", move |args| {
                let table = table_arg(args, 0, "pairs")?;
                let keys = keys.borrow();
                let mut position = position.borrow_mut();
                while *position < keys.len() {
                    let key = keys[*position].clone();
                    *position += 1;
                    // A key removed during the loop reads back as nil and is
                    // skipped, which is what Lua does for a field set to nil.
                    let value = table.borrow().get_own(&key);
                    if !matches!(value, Value::Nil) {
                        return Ok(vec![key.to_value(), value]);
                    }
                }
                Ok(vec![Value::Nil])
            });
            Ok(vec![iterator, Value::Table(table), Value::Nil])
        }),
    );

    interp.set_global(
        "ipairs",
        native("ipairs", |args| {
            let table = table_arg(args, 0, "ipairs")?;
            let iterator = native("ipairs_iterator", |args| {
                let table = table_arg(args, 0, "ipairs")?;
                let index = arg(args, 1).as_number().unwrap_or(0.0) as i64 + 1;
                let value = table.borrow().get(&Key::Int(index));
                // ipairs stops at the first nil, which is why it only ever walks
                // the array part.
                if matches!(value, Value::Nil) {
                    Ok(vec![Value::Nil])
                } else {
                    Ok(vec![Value::Number(index as f64), value])
                }
            });
            Ok(vec![iterator, Value::Table(table), Value::Number(0.0)])
        }),
    );

    // Only `__index` is honoured — see `value::Table::index_fallback` for why.
    interp.set_global(
        "setmetatable",
        native("setmetatable", |args| {
            let table = table_arg(args, 0, "setmetatable")?;
            match arg(args, 1) {
                Value::Table(meta) => {
                    let index = meta.borrow().get(&Key::Str(Rc::from("__index")));
                    table.borrow_mut().set_index_fallback(index);
                }
                Value::Nil => table.borrow_mut().set_index_fallback(Value::Nil),
                other => {
                    return Err(format!(
                        "bad argument #2 to 'setmetatable' (table expected, got {})",
                        other.type_name()
                    ))
                }
            }
            Ok(vec![Value::Table(table)])
        }),
    );

    interp.set_global(
        "getmetatable",
        native("getmetatable", |args| {
            let table = table_arg(args, 0, "getmetatable")?;
            // Reconstructed rather than stored: only `__index` is kept, so the
            // metatable a script gets back is a fresh one carrying that field.
            // Honest about the subset instead of returning nil.
            match table.borrow().index_fallback() {
                Some(index) => Ok(vec![module(vec![("__index", index.clone())])]),
                None => Ok(vec![Value::Nil]),
            }
        }),
    );
}

/// `next(t, key)`: the key after `key`, or nil at the end.
///
/// Exposed because scripts call it directly, but *not* what `pairs` is built on —
/// see the comment there. Position is recovered by searching the current key list,
/// which is stateless and therefore correct only if the table is not written to
/// between calls. A script driving iteration by hand with `next` and mutating as
/// it goes gets Lua's own "undefined behaviour" answer; `pairs` does not, because
/// it snapshots.
fn table_next(args: &[Value]) -> Result<Vec<Value>, String> {
    let table = table_arg(args, 0, "next")?;
    let keys = table.borrow().keys();
    let current = arg(args, 1);

    let start = if matches!(current, Value::Nil) {
        0
    } else {
        let current_key = Key::from_value(&current)
            .ok_or_else(|| "invalid key to 'next'".to_string())?;
        match keys.iter().position(|k| *k == current_key) {
            Some(position) => position + 1,
            // The key is gone — removed during iteration. Ending the loop is the
            // safe answer; restarting it would loop forever.
            None => keys.len(),
        }
    };

    for key in keys.iter().skip(start) {
        let value = table.borrow().get_own(key);
        if !matches!(value, Value::Nil) {
            return Ok(vec![key.to_value(), value]);
        }
    }
    Ok(vec![Value::Nil])
}

// --- math ---

fn install_math(interp: &mut Interpreter) {
    let unary = |name: &'static str, f: fn(f64) -> f64| {
        native(name, move |args| {
            Ok(vec![Value::Number(f(number_arg(args, 0, name)?))])
        })
    };

    interp.set_global(
        "math",
        module(vec![
            ("pi", Value::Number(std::f64::consts::PI)),
            ("huge", Value::Number(f64::INFINITY)),
            ("abs", unary("abs", f64::abs)),
            ("floor", unary("floor", f64::floor)),
            ("ceil", unary("ceil", f64::ceil)),
            ("sqrt", unary("sqrt", f64::sqrt)),
            ("sin", unary("sin", f64::sin)),
            ("cos", unary("cos", f64::cos)),
            ("tan", unary("tan", f64::tan)),
            ("asin", unary("asin", f64::asin)),
            ("acos", unary("acos", f64::acos)),
            ("exp", unary("exp", f64::exp)),
            (
                "atan",
                native("atan", |args| {
                    let y = number_arg(args, 0, "atan")?;
                    // Two-argument form is atan2; Lua 5.3 merged them and scripts
                    // written either way are common.
                    match args.get(1).and_then(|v| v.as_number()) {
                        Some(x) => Ok(vec![Value::Number(y.atan2(x))]),
                        None => Ok(vec![Value::Number(y.atan())]),
                    }
                }),
            ),
            (
                "log",
                native("log", |args| {
                    let x = number_arg(args, 0, "log")?;
                    match args.get(1).and_then(|v| v.as_number()) {
                        Some(base) => Ok(vec![Value::Number(x.log(base))]),
                        None => Ok(vec![Value::Number(x.ln())]),
                    }
                }),
            ),
            (
                "pow",
                native("pow", |args| {
                    let x = number_arg(args, 0, "pow")?;
                    let y = number_arg(args, 1, "pow")?;
                    Ok(vec![Value::Number(x.powf(y))])
                }),
            ),
            (
                "fmod",
                native("fmod", |args| {
                    let x = number_arg(args, 0, "fmod")?;
                    let y = number_arg(args, 1, "fmod")?;
                    // C's fmod, which truncates toward zero — deliberately NOT
                    // Lua's `%` operator, which floors. Lua exposes both, and
                    // conflating them is a classic source of off-by-a-modulus.
                    Ok(vec![Value::Number(x % y)])
                }),
            ),
            (
                "min",
                native("min", |args| {
                    if args.is_empty() {
                        return Err("bad argument #1 to 'min' (value expected)".to_string());
                    }
                    let mut best = number_arg(args, 0, "min")?;
                    for i in 1..args.len() {
                        best = best.min(number_arg(args, i, "min")?);
                    }
                    Ok(vec![Value::Number(best)])
                }),
            ),
            (
                "max",
                native("max", |args| {
                    if args.is_empty() {
                        return Err("bad argument #1 to 'max' (value expected)".to_string());
                    }
                    let mut best = number_arg(args, 0, "max")?;
                    for i in 1..args.len() {
                        best = best.max(number_arg(args, i, "max")?);
                    }
                    Ok(vec![Value::Number(best)])
                }),
            ),
            ("random", make_random()),
        ]),
    );
}

/// `math.random`, over a deterministic generator.
///
/// Deterministic on purpose, and seeded from a fixed constant rather than the
/// clock: a script that misbehaves must misbehave the same way on the next run, or
/// it cannot be debugged and a test cannot assert on it. A script that wants
/// variety can seed from a value the host passes in.
fn make_random() -> Value {
    let state = Rc::new(RefCell::new(0x2545_F491_4F6C_DD1Du64));
    native("random", move |args| {
        let next = {
            let mut s = state.borrow_mut();
            // xorshift64*: short, well-distributed, and self-implemented.
            let mut x = *s;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *s = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        // 53 bits, the mantissa width, so the result is uniform in [0, 1).
        let unit = (next >> 11) as f64 / (1u64 << 53) as f64;
        match (args.first(), args.get(1)) {
            (None, _) => Ok(vec![Value::Number(unit)]),
            (Some(m), None) => {
                let m = m.as_number().unwrap_or(1.0).floor().max(1.0);
                Ok(vec![Value::Number((unit * m).floor() + 1.0)])
            }
            (Some(lo), Some(hi)) => {
                let lo = lo.as_number().unwrap_or(1.0).floor();
                let hi = hi.as_number().unwrap_or(1.0).floor();
                if hi < lo {
                    return Err("bad argument #2 to 'random' (interval is empty)".to_string());
                }
                Ok(vec![Value::Number(lo + (unit * (hi - lo + 1.0)).floor())])
            }
        }
    })
}

// --- string ---

fn install_string(interp: &mut Interpreter) {
    interp.set_global(
        "string",
        module(vec![
            (
                "len",
                native("len", |args| {
                    let s = string_arg(args, 0, "len")?;
                    Ok(vec![Value::Number(s.chars().count() as f64)])
                }),
            ),
            (
                "upper",
                native("upper", |args| {
                    Ok(vec![Value::str(string_arg(args, 0, "upper")?.to_uppercase())])
                }),
            ),
            (
                "lower",
                native("lower", |args| {
                    Ok(vec![Value::str(string_arg(args, 0, "lower")?.to_lowercase())])
                }),
            ),
            (
                "rep",
                native("rep", |args| {
                    let s = string_arg(args, 0, "rep")?;
                    let n = number_arg(args, 1, "rep")? as i64;
                    // Bounded: `("x"):rep(1e9)` would otherwise allocate a
                    // gigabyte from one line of script.
                    const MAX: i64 = 1_000_000;
                    if n > MAX {
                        return Err(format!("string.rep count {n} exceeds the limit of {MAX}"));
                    }
                    Ok(vec![Value::str(s.repeat(n.max(0) as usize))])
                }),
            ),
            ("sub", native("sub", string_sub)),
            (
                "byte",
                native("byte", |args| {
                    let s = string_arg(args, 0, "byte")?;
                    let index = args.get(1).and_then(|v| v.as_number()).unwrap_or(1.0) as usize;
                    Ok(vec![match s.chars().nth(index.saturating_sub(1)) {
                        Some(c) => Value::Number(c as u32 as f64),
                        None => Value::Nil,
                    }])
                }),
            ),
            (
                "char",
                native("char", |args| {
                    let mut out = String::new();
                    for i in 0..args.len() {
                        let code = number_arg(args, i, "char")? as u32;
                        out.push(
                            char::from_u32(code)
                                .ok_or_else(|| format!("string.char: {code} is not a character"))?,
                        );
                    }
                    Ok(vec![Value::str(out)])
                }),
            ),
            (
                "find",
                native("find", |args| {
                    // Plain substring search, not Lua patterns: implementing the
                    // pattern engine is a project in itself, and a script that
                    // passes a pattern would get a wrong answer silently. So this
                    // is documented as plain-only rather than half-matching.
                    let haystack = string_arg(args, 0, "find")?;
                    let needle = string_arg(args, 1, "find")?;
                    match haystack.find(&needle) {
                        Some(byte_index) => {
                            // Lua indices are 1-based and count characters.
                            let start = haystack[..byte_index].chars().count() + 1;
                            let end = start + needle.chars().count() - 1;
                            Ok(vec![Value::Number(start as f64), Value::Number(end as f64)])
                        }
                        None => Ok(vec![Value::Nil]),
                    }
                }),
            ),
            (
                "format",
                native("format", |args| {
                    let template = string_arg(args, 0, "format")?;
                    Ok(vec![Value::str(format_string(&template, &args[1..])?)])
                }),
            ),
        ]),
    );
}

/// `string.sub(s, i, j)` with Lua's index rules.
fn string_sub(args: &[Value]) -> Result<Vec<Value>, String> {
    let s = string_arg(args, 0, "sub")?;
    let chars: Vec<char> = s.chars().collect();
    let length = chars.len() as i64;

    // Negative indices count from the end: -1 is the last character. This is the
    // part most often got wrong, and it is what `s:sub(-3)` relies on.
    let resolve = |index: i64, default: i64| -> i64 {
        if index == 0 {
            default
        } else if index < 0 {
            (length + index + 1).max(1)
        } else {
            index
        }
    };

    let i = resolve(args.get(1).and_then(|v| v.as_number()).unwrap_or(1.0) as i64, 1);
    let j = resolve(
        args.get(2).and_then(|v| v.as_number()).unwrap_or(-1.0) as i64,
        length,
    );
    let j = j.min(length);
    if i > j {
        return Ok(vec![Value::str("")]);
    }
    let slice: String = chars[(i - 1) as usize..j as usize].iter().collect();
    Ok(vec![Value::str(slice)])
}

/// `string.format`, supporting the specifiers a script actually uses.
///
/// `%d %i %f %g %s %q %x %X %%`, with an optional precision for `%f`. Not the full
/// printf grammar: an unsupported specifier is an error naming itself rather than
/// being copied through, because silently emitting `%5.2z` into a log is worse
/// than refusing it.
fn format_string(template: &str, args: &[Value]) -> Result<String, String> {
    let mut out = String::new();
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0;
    let mut next_arg = 0;

    while i < chars.len() {
        if chars[i] != '%' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i >= chars.len() {
            return Err("string.format: trailing '%'".to_string());
        }
        if chars[i] == '%' {
            out.push('%');
            i += 1;
            continue;
        }
        // Optional precision, as in `%.2f`.
        let mut precision: Option<usize> = None;
        if chars[i] == '.' {
            i += 1;
            let mut digits = String::new();
            while i < chars.len() && chars[i].is_ascii_digit() {
                digits.push(chars[i]);
                i += 1;
            }
            precision = digits.parse::<usize>().ok();
        }
        if i >= chars.len() {
            return Err("string.format: incomplete specifier".to_string());
        }
        let specifier = chars[i];
        i += 1;

        let value = args.get(next_arg).cloned().unwrap_or(Value::Nil);
        next_arg += 1;

        let piece = match specifier {
            'd' | 'i' => {
                let n = value
                    .as_number()
                    .ok_or_else(|| format!("string.format: %{specifier} needs a number"))?;
                format!("{}", n.trunc() as i64)
            }
            'f' => {
                let n = value
                    .as_number()
                    .ok_or_else(|| "string.format: %f needs a number".to_string())?;
                // Lua's default is six decimal places, like C's printf.
                format!("{:.*}", precision.unwrap_or(6), n)
            }
            'g' => {
                let n = value
                    .as_number()
                    .ok_or_else(|| "string.format: %g needs a number".to_string())?;
                format_number(n)
            }
            'x' => {
                let n = value
                    .as_number()
                    .ok_or_else(|| "string.format: %x needs a number".to_string())?;
                format!("{:x}", n.trunc() as i64)
            }
            'X' => {
                let n = value
                    .as_number()
                    .ok_or_else(|| "string.format: %X needs a number".to_string())?;
                format!("{:X}", n.trunc() as i64)
            }
            's' => tostring(&value),
            // `%q` quotes for re-reading by Lua, which is what makes it useful for
            // serialising a table.
            'q' => format!("{:?}", tostring(&value)),
            other => {
                return Err(format!(
                    "string.format: unsupported specifier %{other}"
                ))
            }
        };
        out.push_str(&piece);
    }
    Ok(out)
}

// --- table ---

fn install_table(interp: &mut Interpreter) {
    interp.set_global(
        "table",
        module(vec![
            (
                "insert",
                native("insert", |args| {
                    let table = table_arg(args, 0, "insert")?;
                    if args.len() >= 3 {
                        // Three-argument form inserts at a position.
                        let position = number_arg(args, 1, "insert")? as i64;
                        let value = arg(args, 2);
                        let mut borrowed = table.borrow_mut();
                        let length = borrowed.length() as i64;
                        if position < 1 || position > length + 1 {
                            return Err(format!(
                                "bad argument #2 to 'insert' (position {position} out of bounds)"
                            ));
                        }
                        borrowed
                            .array_part_mut()
                            .insert((position - 1) as usize, value);
                    } else {
                        table.borrow_mut().push(arg(args, 1));
                    }
                    Ok(vec![])
                }),
            ),
            (
                "remove",
                native("remove", |args| {
                    let table = table_arg(args, 0, "remove")?;
                    let mut borrowed = table.borrow_mut();
                    let length = borrowed.length() as i64;
                    if length == 0 {
                        return Ok(vec![Value::Nil]);
                    }
                    let position = match args.get(1).and_then(|v| v.as_number()) {
                        Some(p) => p as i64,
                        None => length,
                    };
                    if position < 1 || position > length {
                        return Err(format!(
                            "bad argument #2 to 'remove' (position {position} out of bounds)"
                        ));
                    }
                    Ok(vec![borrowed.array_part_mut().remove((position - 1) as usize)])
                }),
            ),
            (
                "concat",
                native("concat", |args| {
                    let table = table_arg(args, 0, "concat")?;
                    let separator = args
                        .get(1)
                        .and_then(|v| v.as_str_coerced())
                        .unwrap_or_default();
                    let borrowed = table.borrow();
                    let mut pieces = Vec::with_capacity(borrowed.length());
                    for value in borrowed.array_part() {
                        pieces.push(value.as_str_coerced().ok_or_else(|| {
                            format!(
                                "invalid value ({}) at index in table for 'concat'",
                                value.type_name()
                            )
                        })?);
                    }
                    Ok(vec![Value::str(pieces.join(&separator))])
                }),
            ),
            (
                "sort",
                native("sort", |args| {
                    let table = table_arg(args, 0, "sort")?;
                    // Only the default ordering: a comparator would have to call
                    // back into the interpreter from inside `sort_by`, and the
                    // interpreter is not reachable from a native closure (see
                    // OutputSink above for the same constraint). A script needing
                    // a custom order can sort in Lua.
                    if args.len() > 1 && args[1].is_callable() {
                        return Err(
                            "table.sort with a comparator is not supported; sort in Lua instead"
                                .to_string(),
                        );
                    }
                    let mut error = None;
                    table.borrow_mut().array_part_mut().sort_by(|a, b| {
                        match compare(a, b) {
                            Ok(ordering) => ordering,
                            Err(e) => {
                                // `sort_by` cannot fail, so the error is captured
                                // and reported after the sort rather than lost.
                                if error.is_none() {
                                    error = Some(e);
                                }
                                std::cmp::Ordering::Equal
                            }
                        }
                    });
                    match error {
                        Some(e) => Err(format!("table.sort: {e}")),
                        None => Ok(vec![]),
                    }
                }),
            ),
            (
                "getn",
                native("getn", |args| {
                    let table = table_arg(args, 0, "getn")?;
                    Ok(vec![Value::Number(table.borrow().length() as f64)])
                }),
            ),
        ]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(source: &str) -> Interpreter {
        let mut interp = Interpreter::new();
        interp
            .run(source)
            .unwrap_or_else(|e| panic!("failed to run:\n{source}\nerror: {e}"));
        interp
    }

    fn value_of(source: &str) -> Value {
        let mut interp = Interpreter::new();
        interp
            .run(&format!("return {source}"))
            .unwrap_or_else(|e| panic!("failed on {source:?}: {e}"))
            .into_iter()
            .next()
            .unwrap_or(Value::Nil)
    }

    fn number_of(source: &str) -> f64 {
        match value_of(source) {
            Value::Number(n) => n,
            other => panic!("{source:?} produced {other:?}"),
        }
    }

    fn string_of(source: &str) -> String {
        match value_of(source) {
            Value::Str(s) => s.to_string(),
            other => panic!("{source:?} produced {other:?}"),
        }
    }

    // --- print and output ---

    /// `print` must reach the interpreter's buffer, not stdout — that is what
    /// makes it testable and what lets the in-engine console show script output.
    #[test]
    fn print_reaches_the_interpreter_output() {
        let interp = run("print(\"hello\")\nprint(1, 2)");
        assert_eq!(interp.output(), &["hello".to_string(), "1\t2".to_string()]);
    }

    /// The first thing anyone prints: `print(1 + 1)` must say `2`, not `2.0`.
    #[test]
    fn print_formats_integral_numbers_without_a_decimal_point() {
        let interp = run("print(1 + 1)\nprint(10 / 4)");
        assert_eq!(interp.output(), &["2".to_string(), "2.5".to_string()]);
    }

    #[test]
    fn output_is_bounded_so_a_looping_script_cannot_grow_it_forever() {
        let interp = run("for i = 1, 2000 do print(i) end");
        assert!(
            interp.output().len() <= 512,
            "output grew to {}",
            interp.output().len()
        );
        // And it keeps the *newest* lines, which are the ones that matter.
        assert_eq!(interp.output().last().map(|s| s.as_str()), Some("2000"));
    }

    // --- base functions ---

    #[test]
    fn type_reports_lua_type_names() {
        assert_eq!(string_of("type(nil)"), "nil");
        assert_eq!(string_of("type(1)"), "number");
        assert_eq!(string_of("type(\"s\")"), "string");
        assert_eq!(string_of("type(true)"), "boolean");
        assert_eq!(string_of("type({})"), "table");
        assert_eq!(string_of("type(print)"), "function");
    }

    #[test]
    fn tostring_and_tonumber_round_trip() {
        assert_eq!(string_of("tostring(42)"), "42");
        assert_eq!(string_of("tostring(nil)"), "nil");
        assert_eq!(number_of("tonumber(\"42\")"), 42.0);
        assert_eq!(number_of("tonumber(\"2.5\")"), 2.5);
    }

    /// `tonumber(x) or default` is the standard idiom, so a failed conversion must
    /// be nil rather than an error.
    #[test]
    fn a_failed_tonumber_is_nil_not_an_error() {
        assert_eq!(value_of("tonumber(\"abc\")"), Value::Nil);
        assert_eq!(value_of("tonumber({})"), Value::Nil);
        assert_eq!(number_of("tonumber(\"abc\") or 7"), 7.0);
    }

    #[test]
    fn tonumber_accepts_a_base() {
        assert_eq!(number_of("tonumber(\"ff\", 16)"), 255.0);
        assert_eq!(number_of("tonumber(\"101\", 2)"), 5.0);
        assert_eq!(value_of("tonumber(\"zz\", 16)"), Value::Nil);
        let mut interp = Interpreter::new();
        assert!(interp.run("return tonumber(\"1\", 99)").is_err());
    }

    #[test]
    fn assert_passes_through_or_errors() {
        assert_eq!(number_of("assert(5)"), 5.0);
        let mut interp = Interpreter::new();
        let e = interp.run("assert(false, \"my message\")").unwrap_err();
        assert!(e.to_string().contains("my message"), "{e}");
        assert!(interp.run("assert(nil)").is_err());
    }

    #[test]
    fn error_raises_its_message() {
        let mut interp = Interpreter::new();
        let e = interp.run("error(\"boom\")").unwrap_err();
        assert!(e.to_string().contains("boom"), "{e}");
    }

    // --- iteration ---

    #[test]
    fn ipairs_walks_the_array_part_in_order() {
        let interp = run(
            "local t = { \"a\", \"b\", \"c\" }\n\
             local joined = \"\"\n\
             for i, v in ipairs(t) do joined = joined .. i .. v end\n\
             result = joined",
        );
        assert_eq!(interp.get_global("result"), Value::str("1a2b3c"));
    }

    /// `ipairs` stops at the first nil, which is why it only ever walks the array
    /// part — a script relies on that to iterate a list with keys alongside it.
    #[test]
    fn ipairs_stops_at_the_first_hole() {
        let interp = run(
            "local t = {}\n\
             t[1] = \"a\"\n\
             t[3] = \"c\"\n\
             local n = 0\n\
             for _ in ipairs(t) do n = n + 1 end\n\
             result = n",
        );
        assert_eq!(interp.get_global("result"), Value::Number(1.0));
    }

    #[test]
    fn pairs_visits_every_key() {
        let interp = run(
            "local t = { 1, 2, x = 3, y = 4 }\n\
             local count, sum = 0, 0\n\
             for k, v in pairs(t) do\n\
               count = count + 1\n\
               sum = sum + v\n\
             end\n\
             n = count\n\
             total = sum",
        );
        assert_eq!(interp.get_global("n"), Value::Number(4.0));
        assert_eq!(interp.get_global("total"), Value::Number(10.0));
    }

    /// Lua explicitly permits assigning to an existing field during a `pairs`
    /// loop. Re-deriving the key list per step gets this wrong over a HashMap: the
    /// write can reorder the map, so the successor search lands in the wrong place
    /// and an entry is skipped. Measured at 42 instead of 60 before `pairs` was
    /// changed to snapshot its keys.
    #[test]
    fn pairs_survives_assignment_to_an_existing_field() {
        let interp = run(
            "local t = { a = 1, b = 2, c = 3 }\n\
             local visited = 0\n\
             for k, v in pairs(t) do\n\
               t[k] = v * 10\n\
               visited = visited + 1\n\
             end\n\
             result = t.a + t.b + t.c\n\
             count = visited",
        );
        assert_eq!(
            interp.get_global("count"),
            Value::Number(3.0),
            "every key must be visited exactly once"
        );
        assert_eq!(interp.get_global("result"), Value::Number(60.0));
    }

    /// The same hazard with more keys, where a reordering is far more likely to
    /// show up than with three.
    #[test]
    fn pairs_visits_every_key_exactly_once_while_writing() {
        let interp = run(
            "local t = {}\n\
             for i = 1, 50 do t[\"k\" .. i] = i end\n\
             local visited, sum = 0, 0\n\
             for k, v in pairs(t) do\n\
               t[k] = v + 100\n\
               visited = visited + 1\n\
             end\n\
             for k, v in pairs(t) do sum = sum + v end\n\
             count = visited\n\
             total = sum",
        );
        assert_eq!(interp.get_global("count"), Value::Number(50.0));
        // 1..50 sums to 1275, plus 100 each.
        assert_eq!(interp.get_global("total"), Value::Number(1275.0 + 5000.0));
    }

    /// A field set to nil during the loop must be skipped rather than reported
    /// with a nil value, which would break `for k, v in pairs` destructuring.
    #[test]
    fn pairs_skips_a_key_removed_during_iteration() {
        let interp = run(
            "local t = { a = 1, b = 2, c = 3 }\n\
             local seen = 0\n\
             for k, v in pairs(t) do\n\
               if v == nil then error(\"pairs reported a nil value\") end\n\
               seen = seen + 1\n\
               t.b = nil\n\
             end\n\
             result = seen",
        );
        // Either 2 or 3 depending on whether b came before the removal, but never
        // more than 3 and never with a nil value.
        let seen = match interp.get_global("result") {
            Value::Number(n) => n,
            other => panic!("got {other:?}"),
        };
        assert!(
            (2.0..=3.0).contains(&seen),
            "visited {seen} keys, which is neither 2 nor 3"
        );
    }

    #[test]
    fn pairs_over_an_empty_table_runs_zero_times() {
        let interp = run(
            "local n = 0\n\
             for _ in pairs({}) do n = n + 1 end\n\
             result = n",
        );
        assert_eq!(interp.get_global("result"), Value::Number(0.0));
    }

    #[test]
    fn pairs_on_a_non_table_is_an_error() {
        let mut interp = Interpreter::new();
        assert!(interp.run("for k in pairs(nil) do end").is_err());
        assert!(interp.run("for k in pairs(5) do end").is_err());
    }

    // --- metatables ---

    /// `__index` is what makes prototype-style objects work, which is the whole
    /// reason a game script wants metatables.
    #[test]
    fn setmetatable_provides_index_inheritance() {
        let interp = run(
            "local Base = { greet = function(self) return \"hi \" .. self.name end }\n\
             local obj = setmetatable({ name = \"world\" }, { __index = Base })\n\
             result = obj:greet()",
        );
        assert_eq!(interp.get_global("result"), Value::str("hi world"));
    }

    #[test]
    fn getmetatable_reports_the_index_table() {
        let interp = run(
            "local base = { x = 1 }\n\
             local obj = setmetatable({}, { __index = base })\n\
             local meta = getmetatable(obj)\n\
             result = meta.__index.x\n\
             none = getmetatable({})",
        );
        assert_eq!(interp.get_global("result"), Value::Number(1.0));
        assert_eq!(interp.get_global("none"), Value::Nil);
    }

    #[test]
    fn setmetatable_with_nil_clears_inheritance() {
        let interp = run(
            "local base = { x = 1 }\n\
             local obj = setmetatable({}, { __index = base })\n\
             before = obj.x\n\
             setmetatable(obj, nil)\n\
             after = obj.x",
        );
        assert_eq!(interp.get_global("before"), Value::Number(1.0));
        assert_eq!(interp.get_global("after"), Value::Nil);
    }

    // --- math ---

    #[test]
    fn math_basics_work() {
        assert_eq!(number_of("math.abs(-3)"), 3.0);
        assert_eq!(number_of("math.floor(2.7)"), 2.0);
        assert_eq!(number_of("math.ceil(2.1)"), 3.0);
        assert_eq!(number_of("math.sqrt(16)"), 4.0);
        assert_eq!(number_of("math.min(3, 1, 2)"), 1.0);
        assert_eq!(number_of("math.max(3, 1, 2)"), 3.0);
        assert!((number_of("math.pi") - std::f64::consts::PI).abs() < 1e-12);
        assert!(number_of("math.huge").is_infinite());
    }

    #[test]
    fn math_trigonometry_agrees_with_known_values() {
        assert!(number_of("math.sin(0)").abs() < 1e-12);
        assert!((number_of("math.cos(0)") - 1.0).abs() < 1e-12);
        assert!((number_of("math.sin(math.pi / 2)") - 1.0).abs() < 1e-12);
        // Two-argument atan is atan2; scripts are written both ways.
        assert!((number_of("math.atan(1, 1)") - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((number_of("math.atan(1)") - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    /// `math.fmod` truncates toward zero; the `%` operator floors. Lua exposes
    /// both and conflating them is a classic off-by-a-modulus.
    #[test]
    fn fmod_differs_from_the_modulo_operator_on_negatives() {
        assert_eq!(number_of("math.fmod(-1, 3)"), -1.0);
        assert_eq!(number_of("-1 % 3"), 2.0);
    }

    /// Deterministic on purpose: a script that misbehaves must misbehave the same
    /// way next run, or it cannot be debugged.
    #[test]
    fn math_random_is_deterministic_across_interpreters() {
        let first = run("local t = {} for i = 1, 5 do t[i] = math.random() end result = t[1] .. \",\" .. t[5]");
        let second = run("local t = {} for i = 1, 5 do t[i] = math.random() end result = t[1] .. \",\" .. t[5]");
        assert_eq!(
            first.get_global("result"),
            second.get_global("result"),
            "two fresh interpreters must produce the same sequence"
        );
    }

    #[test]
    fn math_random_respects_its_bounds() {
        let interp = run(
            "local lo, hi = 100, 100\n\
             local ok = true\n\
             for i = 1, 200 do\n\
               local r = math.random(1, 6)\n\
               if r < 1 or r > 6 or r ~= math.floor(r) then ok = false end\n\
             end\n\
             result = ok",
        );
        assert_eq!(interp.get_global("result"), Value::Bool(true));

        let unit = run(
            "local ok = true\n\
             for i = 1, 200 do\n\
               local r = math.random()\n\
               if r < 0 or r >= 1 then ok = false end\n\
             end\n\
             result = ok",
        );
        assert_eq!(unit.get_global("result"), Value::Bool(true));
    }

    #[test]
    fn math_functions_reject_non_numbers_with_a_useful_message() {
        let mut interp = Interpreter::new();
        let e = interp.run("return math.sqrt(\"abc\")").unwrap_err();
        assert!(
            e.to_string().contains("sqrt") && e.to_string().contains("number"),
            "{e}"
        );
    }

    // --- string ---

    #[test]
    fn string_basics_work() {
        assert_eq!(number_of("string.len(\"hello\")"), 5.0);
        assert_eq!(string_of("string.upper(\"abc\")"), "ABC");
        assert_eq!(string_of("string.lower(\"ABC\")"), "abc");
        assert_eq!(string_of("string.rep(\"ab\", 3)"), "ababab");
        assert_eq!(number_of("string.byte(\"A\")"), 65.0);
        assert_eq!(string_of("string.char(72, 105)"), "Hi");
    }

    /// Negative indices count from the end, which is what `s:sub(-3)` relies on
    /// and the part most often got wrong.
    #[test]
    fn string_sub_handles_negative_indices() {
        assert_eq!(string_of("string.sub(\"hello\", 2, 4)"), "ell");
        assert_eq!(string_of("string.sub(\"hello\", 2)"), "ello");
        assert_eq!(string_of("string.sub(\"hello\", -3)"), "llo");
        assert_eq!(string_of("string.sub(\"hello\", -3, -2)"), "ll");
        assert_eq!(string_of("string.sub(\"hello\", 1, -1)"), "hello");
    }

    #[test]
    fn string_sub_clamps_out_of_range_indices() {
        assert_eq!(string_of("string.sub(\"abc\", 10)"), "");
        assert_eq!(string_of("string.sub(\"abc\", 1, 100)"), "abc");
        assert_eq!(string_of("string.sub(\"abc\", 3, 1)"), "");
        assert_eq!(string_of("string.sub(\"\", 1, 5)"), "");
    }

    #[test]
    fn string_find_reports_one_based_bounds() {
        let interp = run(
            "local a, b = string.find(\"hello world\", \"world\")\n\
             start = a\n\
             finish = b\n\
             missing = string.find(\"abc\", \"xyz\")",
        );
        assert_eq!(interp.get_global("start"), Value::Number(7.0));
        assert_eq!(interp.get_global("finish"), Value::Number(11.0));
        assert_eq!(interp.get_global("missing"), Value::Nil);
    }

    #[test]
    fn string_format_covers_the_common_specifiers() {
        assert_eq!(string_of("string.format(\"%d\", 42)"), "42");
        assert_eq!(string_of("string.format(\"%.2f\", 3.14159)"), "3.14");
        assert_eq!(string_of("string.format(\"%s!\", \"hi\")"), "hi!");
        assert_eq!(string_of("string.format(\"%x\", 255)"), "ff");
        assert_eq!(string_of("string.format(\"%X\", 255)"), "FF");
        assert_eq!(string_of("string.format(\"100%%\")"), "100%");
        assert_eq!(
            string_of("string.format(\"%s=%d\", \"n\", 7)"),
            "n=7",
            "multiple specifiers consume arguments in order"
        );
    }

    /// An unsupported specifier must be an error, not copied through: silently
    /// emitting `%z` into a log is worse than refusing it.
    #[test]
    fn an_unsupported_format_specifier_is_rejected() {
        let mut interp = Interpreter::new();
        let e = interp.run("return string.format(\"%z\", 1)").unwrap_err();
        assert!(e.to_string().contains("%z"), "{e}");
        assert!(interp.run("return string.format(\"%\")").is_err());
    }

    /// One line of script must not be able to allocate a gigabyte.
    #[test]
    fn string_rep_is_bounded() {
        let mut interp = Interpreter::new();
        let e = interp.run("return string.rep(\"x\", 1e9)").unwrap_err();
        assert!(e.to_string().contains("limit"), "{e}");
        // A sane count still works.
        assert_eq!(string_of("string.rep(\"x\", 5)"), "xxxxx");
    }

    #[test]
    fn string_functions_count_characters_not_bytes() {
        assert_eq!(
            number_of("string.len(\"привет\")"),
            6.0,
            "six characters, twelve bytes"
        );
        assert_eq!(string_of("string.sub(\"привет\", 1, 3)"), "при");
    }

    // --- table ---

    #[test]
    fn table_insert_and_remove_work_at_the_end() {
        let interp = run(
            "local t = {}\n\
             table.insert(t, \"a\")\n\
             table.insert(t, \"b\")\n\
             len = #t\n\
             last = table.remove(t)\n\
             after = #t",
        );
        assert_eq!(interp.get_global("len"), Value::Number(2.0));
        assert_eq!(interp.get_global("last"), Value::str("b"));
        assert_eq!(interp.get_global("after"), Value::Number(1.0));
    }

    #[test]
    fn table_insert_and_remove_work_at_a_position() {
        let interp = run(
            "local t = { \"a\", \"c\" }\n\
             table.insert(t, 2, \"b\")\n\
             joined = table.concat(t)\n\
             local removed = table.remove(t, 1)\n\
             first = removed\n\
             rest = table.concat(t)",
        );
        assert_eq!(interp.get_global("joined"), Value::str("abc"));
        assert_eq!(interp.get_global("first"), Value::str("a"));
        assert_eq!(interp.get_global("rest"), Value::str("bc"));
    }

    #[test]
    fn table_insert_rejects_an_out_of_bounds_position() {
        let mut interp = Interpreter::new();
        assert!(interp.run("local t = {} table.insert(t, 5, \"x\")").is_err());
        assert!(interp.run("local t = {} table.insert(t, 0, \"x\")").is_err());
    }

    #[test]
    fn removing_from_an_empty_table_gives_nil() {
        assert_eq!(value_of("table.remove({})"), Value::Nil);
    }

    #[test]
    fn table_concat_joins_with_a_separator() {
        assert_eq!(string_of("table.concat({1, 2, 3}, \"-\")"), "1-2-3");
        assert_eq!(string_of("table.concat({\"a\", \"b\"})"), "ab");
        assert_eq!(string_of("table.concat({})"), "");
    }

    #[test]
    fn table_concat_rejects_a_non_string_element() {
        let mut interp = Interpreter::new();
        assert!(interp.run("return table.concat({1, {}, 3})").is_err());
    }

    #[test]
    fn table_sort_orders_numbers_and_strings() {
        let interp = run(
            "local n = { 3, 1, 2 }\n\
             table.sort(n)\n\
             numbers = table.concat(n, \",\")\n\
             local s = { \"c\", \"a\", \"b\" }\n\
             table.sort(s)\n\
             strings = table.concat(s)",
        );
        assert_eq!(interp.get_global("numbers"), Value::str("1,2,3"));
        assert_eq!(interp.get_global("strings"), Value::str("abc"));
    }

    /// Sorting mixed types cannot produce a consistent order, so it must be an
    /// error rather than an arbitrary arrangement.
    #[test]
    fn sorting_incomparable_values_is_an_error() {
        let mut interp = Interpreter::new();
        assert!(interp.run("local t = {1, \"a\", 2} table.sort(t)").is_err());
    }

    /// A comparator would need to call back into the interpreter from inside
    /// `sort_by`, which a native closure cannot do. Refusing it beats sorting by
    /// the wrong order silently.
    #[test]
    fn table_sort_with_a_comparator_says_it_is_unsupported() {
        let mut interp = Interpreter::new();
        let e = interp
            .run("local t = {1, 2} table.sort(t, function(a, b) return a > b end)")
            .unwrap_err();
        assert!(e.to_string().contains("comparator"), "{e}");
    }

    #[test]
    fn table_functions_reject_non_tables() {
        let mut interp = Interpreter::new();
        assert!(interp.run("table.insert(nil, 1)").is_err());
        assert!(interp.run("return table.concat(5)").is_err());
    }
}
