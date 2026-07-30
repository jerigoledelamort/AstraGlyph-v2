// Lua scripting: lexer, parser, interpreter, and the engine bindings.
//
// Self-implemented per the "no external crates" rule — no mlua, no rlua. That
// makes the scope a deliberate choice rather than whatever a crate happened to
// offer, and the choice is: enough Lua to write entity behaviour and triggers in,
// not a conforming implementation.
//
// Present: local and global variables, tables (array and hash parts), functions
// with closures and varargs, all the operators with Lua's precedence, if/while/
// repeat/numeric and generic for, break, return, method call sugar (`a:b()`), and
// a standard library subset (print, type, tostring, tonumber, pairs, ipairs,
// math, string, table).
//
// Absent, and why: coroutines (they need a resumable interpreter, which means
// either a bytecode VM or a state machine over the AST — a different design, not
// an addition), metatables beyond `__index` on tables, `goto`, weak tables, and
// the `io`/`os` libraries. A script that uses one gets an error naming it rather
// than a wrong answer.

pub mod bindings;
pub mod interp;
pub mod lexer;
pub mod parser;
pub mod stdlib;
pub mod value;

pub use bindings::{EngineState, ScriptCommand, ScriptHost};
