// Lua interpreter: walks the AST from `parser` and evaluates it.
//
// A tree-walking interpreter rather than a bytecode VM. The scripts here run a
// handful of statements per entity per frame, so the constant factor does not
// matter, and a tree walker is a few hundred lines against a compiler plus a VM
// plus a disassembler for when the compiler is wrong.
//
// Two things are bounded rather than trusted, because a script is input the user
// edits live and reloads:
//
// - Call depth, so infinite recursion is an error rather than a process abort. A
//   stack overflow cannot be caught in safe Rust.
// - Instruction count, so `while true do end` returns an error rather than hanging
//   the render loop. A frozen window with no message is indistinguishable from a
//   crash, and the whole point of hot-reload is that a bad edit is recoverable.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::engine::core::{EngineError, Result};
use crate::scripting::parser::{
    parse, BinOp, Expr, FunctionBody, Stat, TableField, UnOp,
};
use crate::scripting::value::{format_number, Key, LuaFunction, NativeFunction, Table, Value};

/// Deepest Lua call nesting allowed.
const MAX_CALL_DEPTH: u32 = 100;

/// Statements and expressions a single `run` or `call` may evaluate.
///
/// Not a wall-clock timeout: a step budget is deterministic, so a script that
/// fails on one machine fails identically on another and a test can assert on it.
/// A time limit would make the same script pass or fail depending on load.
pub const DEFAULT_STEP_LIMIT: u64 = 2_000_000;

/// How a block finished. Not an error type: `break` and `return` are ordinary
/// control flow, and modelling them as errors would let a `pcall`-style construct
/// swallow them.
enum Flow {
    Normal,
    Break,
    Return(Vec<Value>),
}

/// A scope: one lexical block's local variables.
type Scope = Rc<RefCell<HashMap<String, Value>>>;

/// The interpreter and its global state.
pub struct Interpreter {
    /// Globals, as a Lua table so a script can reach them through `_G`.
    globals: Rc<RefCell<Table>>,
    /// Active local scopes, innermost last.
    scopes: Vec<Scope>,
    /// Varargs of the innermost function, for `...`.
    varargs: Vec<Vec<Value>>,
    call_depth: u32,
    steps: u64,
    step_limit: u64,
    /// Lines printed by scripts, so `print` is observable from Rust without
    /// capturing stdout — which is what makes the standard library testable.
    output: Vec<String>,
    /// The buffer `print` writes into, shared with the native closure.
    output_sink: Option<Rc<RefCell<Vec<String>>>>,
}

fn err(msg: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidState(msg.to_string())
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    /// A fresh interpreter with the standard library installed.
    pub fn new() -> Self {
        let mut interp = Self {
            globals: Rc::new(RefCell::new(Table::new())),
            scopes: Vec::new(),
            varargs: Vec::new(),
            call_depth: 0,
            steps: 0,
            step_limit: DEFAULT_STEP_LIMIT,
            output: Vec::new(),
            output_sink: None,
        };
        crate::scripting::stdlib::install(&mut interp);
        interp
    }

    /// Change the step budget. Zero means unlimited, which is only safe for code
    /// the caller wrote.
    pub fn set_step_limit(&mut self, limit: u64) {
        self.step_limit = limit;
    }

    /// Steps consumed by the most recent run.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Lines a script printed, oldest first.
    ///
    /// Drains the shared sink first, so lines `print` wrote are visible here
    /// without the native closure needing a reference to the interpreter.
    pub fn output(&self) -> &[String] {
        &self.output
    }

    /// Move anything `print` wrote into the interpreter's own buffer.
    ///
    /// Called before every read of `output`, and after every run, so a caller
    /// never sees a stale view. Bounded the same way `print_line` is.
    pub fn drain_output(&mut self) {
        let Some(sink) = self.output_sink.clone() else {
            return;
        };
        let lines: Vec<String> = sink.borrow_mut().drain(..).collect();
        for line in lines {
            self.print_line(line);
        }
    }

    /// Clear the captured output, including anything still in the sink.
    pub fn clear_output(&mut self) {
        self.output.clear();
        if let Some(sink) = &self.output_sink {
            sink.borrow_mut().clear();
        }
    }

    /// Adopt the sink `stdlib` gave to `print`.
    ///
    /// A native closure is `Fn(&[Value])` with no path back to the interpreter, so
    /// `print` cannot call `print_line` directly. The two share a buffer instead,
    /// and this is where the interpreter picks up its end of it.
    pub fn attach_output_sink(&mut self, sink: Rc<RefCell<Vec<String>>>) {
        self.output_sink = Some(sink);
    }

    /// Record a line of script output.
    pub fn print_line(&mut self, line: String) {
        // Bounded, because a script in a loop can print without limit and this
        // buffer is held for the process's lifetime.
        const MAX_LINES: usize = 512;
        if self.output.len() >= MAX_LINES {
            self.output.remove(0);
        }
        self.output.push(line);
    }

    /// The globals table.
    pub fn globals(&self) -> &Rc<RefCell<Table>> {
        &self.globals
    }

    /// Read a global.
    pub fn get_global(&self, name: &str) -> Value {
        self.globals.borrow().get(&Key::Str(Rc::from(name)))
    }

    /// Write a global.
    pub fn set_global(&mut self, name: &str, value: Value) {
        self.globals.borrow_mut().set(Key::Str(Rc::from(name)), value);
    }

    /// Expose a Rust function to scripts.
    pub fn set_native(
        &mut self,
        name: &str,
        function: impl Fn(&[Value]) -> std::result::Result<Vec<Value>, String> + 'static,
    ) {
        let value = Value::Native(Rc::new(NativeFunction {
            name: name.to_string(),
            function: Box::new(function),
        }));
        self.set_global(name, value);
    }

    /// Parse and run a chunk, returning whatever it returned.
    pub fn run(&mut self, source: &str) -> Result<Vec<Value>> {
        let block = parse(source)?;
        self.steps = 0;
        self.call_depth = 0;
        // A chunk gets a scope of its own, so its locals do not leak into the next
        // chunk — which matters for hot-reload, where the same names are redefined
        // repeatedly.
        self.push_scope();
        let flow = self.exec_block(&block);
        self.pop_scope();
        self.drain_output();
        match flow? {
            Flow::Return(values) => Ok(values),
            _ => Ok(Vec::new()),
        }
    }

    /// Call a global function by name.
    ///
    /// This is the entry point the engine uses for entity behaviour: a script
    /// defines `function update(dt)` and Rust calls it every frame.
    pub fn call_global(&mut self, name: &str, args: &[Value]) -> Result<Vec<Value>> {
        let function = self.get_global(name);
        if !function.is_callable() {
            return Err(err(format!(
                "global {name:?} is {}, not a function",
                function.type_name()
            )));
        }
        self.steps = 0;
        self.call_depth = 0;
        let result = self.call_value(&function, args);
        self.drain_output();
        result
    }

    /// Whether a global holds something callable.
    pub fn has_function(&self, name: &str) -> bool {
        self.get_global(name).is_callable()
    }

    // --- scopes ---

    fn push_scope(&mut self) {
        self.scopes.push(Rc::new(RefCell::new(HashMap::new())));
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Declare a local in the innermost scope.
    fn declare_local(&mut self, name: &str, value: Value) {
        if let Some(scope) = self.scopes.last() {
            scope.borrow_mut().insert(name.to_string(), value);
        } else {
            // No scope: a top-level declaration outside any chunk. Treat it as a
            // global rather than dropping it.
            self.set_global(name, value);
        }
    }

    /// Look a name up: innermost scope outward, then globals.
    fn lookup(&self, name: &str) -> Value {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.borrow().get(name) {
                return value.clone();
            }
        }
        self.get_global(name)
    }

    /// Assign to an existing local if there is one, otherwise to a global.
    ///
    /// This is Lua's rule and it is easy to get backwards: an assignment with no
    /// `local` writes a *global* unless a local of that name is in scope.
    fn assign_name(&mut self, name: &str, value: Value) {
        for scope in self.scopes.iter().rev() {
            let mut borrowed = scope.borrow_mut();
            if borrowed.contains_key(name) {
                borrowed.insert(name.to_string(), value);
                return;
            }
        }
        self.set_global(name, value);
    }

    /// Count one step, failing if the budget is exhausted.
    fn step(&mut self) -> Result<()> {
        self.steps += 1;
        if self.step_limit > 0 && self.steps > self.step_limit {
            return Err(err(format!(
                "script exceeded {} steps (infinite loop?)",
                self.step_limit
            )));
        }
        Ok(())
    }

    // --- statements ---

    fn exec_block(&mut self, block: &[Stat]) -> Result<Flow> {
        for stat in block {
            match self.exec_stat(stat)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    /// Execute a block in its own scope.
    fn exec_scoped_block(&mut self, block: &[Stat]) -> Result<Flow> {
        self.push_scope();
        let flow = self.exec_block(block);
        self.pop_scope();
        flow
    }

    fn exec_stat(&mut self, stat: &Stat) -> Result<Flow> {
        self.step()?;
        match stat {
            Stat::Local(names, exprs) => {
                let values = self.eval_multi(exprs, names.len())?;
                for (i, name) in names.iter().enumerate() {
                    let value = values.get(i).cloned().unwrap_or(Value::Nil);
                    self.declare_local(name, value);
                }
                Ok(Flow::Normal)
            }
            Stat::Assign(targets, exprs) => {
                let values = self.eval_multi(exprs, targets.len())?;
                for (i, target) in targets.iter().enumerate() {
                    let value = values.get(i).cloned().unwrap_or(Value::Nil);
                    self.assign_to(target, value)?;
                }
                Ok(Flow::Normal)
            }
            Stat::ExprStat(expr) => {
                self.eval(expr)?;
                Ok(Flow::Normal)
            }
            Stat::If(branches, else_block) => {
                for (condition, body) in branches {
                    if self.eval(condition)?.truthy() {
                        return self.exec_scoped_block(body);
                    }
                }
                if let Some(body) = else_block {
                    return self.exec_scoped_block(body);
                }
                Ok(Flow::Normal)
            }
            Stat::While(condition, body) => {
                while self.eval(condition)?.truthy() {
                    self.step()?;
                    match self.exec_scoped_block(body)? {
                        Flow::Break => break,
                        Flow::Return(values) => return Ok(Flow::Return(values)),
                        Flow::Normal => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stat::Repeat(body, condition) => {
                loop {
                    self.step()?;
                    // The condition sees the body's locals, so both live in one
                    // scope. This is the one place Lua's scoping is not
                    // block-shaped, and splitting them would break
                    // `repeat local x = f() until x`.
                    self.push_scope();
                    let flow = self.exec_block(body);
                    let flow = match flow {
                        Ok(f) => f,
                        Err(e) => {
                            self.pop_scope();
                            return Err(e);
                        }
                    };
                    match flow {
                        Flow::Break => {
                            self.pop_scope();
                            break;
                        }
                        Flow::Return(values) => {
                            self.pop_scope();
                            return Ok(Flow::Return(values));
                        }
                        Flow::Normal => {}
                    }
                    let done = self.eval(condition);
                    self.pop_scope();
                    if done?.truthy() {
                        break;
                    }
                }
                Ok(Flow::Normal)
            }
            Stat::NumericFor {
                var,
                start,
                limit,
                step,
                body,
            } => {
                let start = self.eval_number(start, "'for' initial value")?;
                let limit = self.eval_number(limit, "'for' limit")?;
                let step_value = match step {
                    Some(expr) => self.eval_number(expr, "'for' step")?,
                    None => 1.0,
                };
                if step_value == 0.0 {
                    return Err(err("'for' step is zero"));
                }
                let mut i = start;
                loop {
                    // The direction of the comparison follows the sign of the
                    // step, which is what makes `for i = 10, 1, -1` terminate.
                    let done = if step_value > 0.0 { i > limit } else { i < limit };
                    if done {
                        break;
                    }
                    self.step()?;
                    self.push_scope();
                    self.declare_local(var, Value::Number(i));
                    let flow = self.exec_block(body);
                    self.pop_scope();
                    match flow? {
                        Flow::Break => break,
                        Flow::Return(values) => return Ok(Flow::Return(values)),
                        Flow::Normal => {}
                    }
                    i += step_value;
                }
                Ok(Flow::Normal)
            }
            Stat::GenericFor { vars, exprs, body } => {
                // `for ... in f, s, var` — the iterator protocol: call f(s, var)
                // until it returns nil.
                let control = self.eval_multi(exprs, 3)?;
                let iterator = control.first().cloned().unwrap_or(Value::Nil);
                let state = control.get(1).cloned().unwrap_or(Value::Nil);
                let mut var = control.get(2).cloned().unwrap_or(Value::Nil);
                if !iterator.is_callable() {
                    return Err(err(format!(
                        "'for' iterator is {}, not a function",
                        iterator.type_name()
                    )));
                }
                loop {
                    self.step()?;
                    let results = self.call_value(&iterator, &[state.clone(), var.clone()])?;
                    let first = results.first().cloned().unwrap_or(Value::Nil);
                    if matches!(first, Value::Nil) {
                        break;
                    }
                    var = first.clone();
                    self.push_scope();
                    for (i, name) in vars.iter().enumerate() {
                        self.declare_local(name, results.get(i).cloned().unwrap_or(Value::Nil));
                    }
                    let flow = self.exec_block(body);
                    self.pop_scope();
                    match flow? {
                        Flow::Break => break,
                        Flow::Return(values) => return Ok(Flow::Return(values)),
                        Flow::Normal => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stat::Do(body) => self.exec_scoped_block(body),
            Stat::Return(exprs) => {
                let values = self.eval_multi_all(exprs)?;
                Ok(Flow::Return(values))
            }
            Stat::Break => Ok(Flow::Break),
            Stat::FunctionDecl { target, body, .. } => {
                let function = self.make_closure(body);
                self.assign_to(target, function)?;
                Ok(Flow::Normal)
            }
            Stat::LocalFunction(name, body) => {
                // Declared *before* the body closes over the scope, so the
                // function can call itself. Doing it after would make every
                // recursive local function a call to nil.
                self.declare_local(name, Value::Nil);
                let function = self.make_closure(body);
                self.declare_local(name, function);
                Ok(Flow::Normal)
            }
        }
    }

    fn assign_to(&mut self, target: &Expr, value: Value) -> Result<()> {
        match target {
            Expr::Name(name) => {
                self.assign_name(name, value);
                Ok(())
            }
            Expr::Index(table_expr, key_expr) => {
                let table = self.eval(table_expr)?;
                let key_value = self.eval(key_expr)?;
                let Value::Table(table) = table else {
                    return Err(err(format!(
                        "cannot index a {} value",
                        table.type_name()
                    )));
                };
                let key = Key::from_value(&key_value).ok_or_else(|| {
                    err(format!("invalid table key: {}", key_value.type_name()))
                })?;
                table.borrow_mut().set(key, value);
                Ok(())
            }
            other => Err(err(format!("cannot assign to {other:?}"))),
        }
    }

    /// Build a closure over the current scopes.
    fn make_closure(&self, body: &FunctionBody) -> Value {
        Value::Function(Rc::new(LuaFunction {
            body: Rc::new(body.clone()),
            // The scopes are captured by handle, not by copy, so a closure that
            // mutates an outer local is visible to everything else holding it.
            captured: self.scopes.clone(),
        }))
    }

    // --- expressions ---

    /// Evaluate an expression list, expanding the last element's multiple returns
    /// so `local a, b = f()` binds both.
    fn eval_multi(&mut self, exprs: &[Expr], want: usize) -> Result<Vec<Value>> {
        let mut values = self.eval_multi_all(exprs)?;
        values.resize(values.len().max(want), Value::Nil);
        Ok(values)
    }

    /// Evaluate an expression list, expanding only the last element.
    ///
    /// Lua truncates every call but the last to one value: in `f(), g()` only
    /// `g()`'s extra returns survive. Expanding all of them would make argument
    /// counts unpredictable.
    fn eval_multi_all(&mut self, exprs: &[Expr]) -> Result<Vec<Value>> {
        let mut values = Vec::with_capacity(exprs.len());
        for (i, expr) in exprs.iter().enumerate() {
            let is_last = i + 1 == exprs.len();
            if is_last {
                values.extend(self.eval_expanded(expr)?);
            } else {
                values.push(self.eval(expr)?);
            }
        }
        Ok(values)
    }

    /// Evaluate an expression, keeping all its results if it is a call or `...`.
    fn eval_expanded(&mut self, expr: &Expr) -> Result<Vec<Value>> {
        match expr {
            Expr::Call(_, _) | Expr::MethodCall(_, _, _) => self.eval_call(expr),
            Expr::Vararg => Ok(self.varargs.last().cloned().unwrap_or_default()),
            other => Ok(vec![self.eval(other)?]),
        }
    }

    /// Evaluate to exactly one value.
    fn eval(&mut self, expr: &Expr) -> Result<Value> {
        self.step()?;
        match expr {
            Expr::Nil => Ok(Value::Nil),
            Expr::True => Ok(Value::Bool(true)),
            Expr::False => Ok(Value::Bool(false)),
            Expr::Number(n) => Ok(Value::Number(*n)),
            Expr::Str(s) => Ok(Value::str(s)),
            // A bare `...` in a single-value position is its first element, which
            // is what `local a = ...` means.
            Expr::Vararg => Ok(self
                .varargs
                .last()
                .and_then(|v| v.first().cloned())
                .unwrap_or(Value::Nil)),
            Expr::Name(name) => Ok(self.lookup(name)),
            Expr::Index(table_expr, key_expr) => {
                let table = self.eval(table_expr)?;
                let key_value = self.eval(key_expr)?;
                self.index(&table, &key_value)
            }
            Expr::Call(_, _) | Expr::MethodCall(_, _, _) => {
                // In a single-value position a call yields its first result.
                Ok(self.eval_call(expr)?.into_iter().next().unwrap_or(Value::Nil))
            }
            Expr::Function(body) => Ok(self.make_closure(body)),
            Expr::Table(fields) => self.eval_table(fields),
            Expr::Binary(op, left, right) => self.eval_binary(*op, left, right),
            Expr::Unary(op, operand) => {
                let value = self.eval(operand)?;
                self.eval_unary(*op, value)
            }
        }
    }

    fn index(&mut self, table: &Value, key_value: &Value) -> Result<Value> {
        match table {
            Value::Table(table) => {
                let key = Key::from_value(key_value).ok_or_else(|| {
                    err(format!("invalid table key: {}", key_value.type_name()))
                })?;
                Ok(table.borrow().get(&key))
            }
            // Strings support `#` and the string library through method syntax in
            // real Lua via a metatable; here indexing a string is an error with a
            // message rather than a silent nil, which is the failure mode that
            // actually helps.
            other => Err(err(format!(
                "attempt to index a {} value",
                other.type_name()
            ))),
        }
    }

    fn eval_table(&mut self, fields: &[TableField]) -> Result<Value> {
        let table = Rc::new(RefCell::new(Table::new()));
        for (i, field) in fields.iter().enumerate() {
            match field {
                TableField::Positional(expr) => {
                    let is_last = i + 1 == fields.len();
                    if is_last {
                        // The last positional field expands: `{f()}` collects all
                        // of f's results, which is how `{...}` packs varargs.
                        for value in self.eval_expanded(expr)? {
                            table.borrow_mut().push(value);
                        }
                    } else {
                        let value = self.eval(expr)?;
                        table.borrow_mut().push(value);
                    }
                }
                TableField::Keyed(key_expr, value_expr) => {
                    let key_value = self.eval(key_expr)?;
                    let value = self.eval(value_expr)?;
                    let key = Key::from_value(&key_value).ok_or_else(|| {
                        err(format!("invalid table key: {}", key_value.type_name()))
                    })?;
                    table.borrow_mut().set(key, value);
                }
            }
        }
        Ok(Value::Table(table))
    }

    fn eval_number(&mut self, expr: &Expr, what: &str) -> Result<f64> {
        let value = self.eval(expr)?;
        value
            .as_number()
            .ok_or_else(|| err(format!("{what} must be a number, got {}", value.type_name())))
    }

    fn eval_binary(&mut self, op: BinOp, left: &Expr, right: &Expr) -> Result<Value> {
        // `and`/`or` short-circuit, so the right side must not be evaluated
        // eagerly. This is also what makes `x and x.field` a safe idiom.
        match op {
            BinOp::And => {
                let l = self.eval(left)?;
                return if l.truthy() { self.eval(right) } else { Ok(l) };
            }
            BinOp::Or => {
                let l = self.eval(left)?;
                return if l.truthy() { Ok(l) } else { self.eval(right) };
            }
            _ => {}
        }

        let l = self.eval(left)?;
        let r = self.eval(right)?;
        binary_op(op, &l, &r).map_err(err)
    }

    fn eval_unary(&mut self, op: UnOp, value: Value) -> Result<Value> {
        match op {
            UnOp::Neg => value
                .as_number()
                .map(|n| Value::Number(-n))
                .ok_or_else(|| err(format!("cannot negate a {} value", value.type_name()))),
            UnOp::Not => Ok(Value::Bool(!value.truthy())),
            UnOp::Len => match &value {
                Value::Str(s) => Ok(Value::Number(s.chars().count() as f64)),
                Value::Table(t) => Ok(Value::Number(t.borrow().length() as f64)),
                other => Err(err(format!(
                    "cannot take the length of a {} value",
                    other.type_name()
                ))),
            },
        }
    }

    // --- calls ---

    fn eval_call(&mut self, expr: &Expr) -> Result<Vec<Value>> {
        match expr {
            Expr::Call(callee_expr, arg_exprs) => {
                let callee = self.eval(callee_expr)?;
                let args = self.eval_multi_all(arg_exprs)?;
                self.call_value(&callee, &args).map_err(|e| {
                    // Name the callee in the error: "attempt to call a nil value"
                    // is much less useful than knowing which name was nil.
                    if let Expr::Name(name) = &**callee_expr {
                        err(format!("in call to {name:?}: {e}"))
                    } else {
                        e
                    }
                })
            }
            Expr::MethodCall(receiver_expr, method, arg_exprs) => {
                // The receiver is evaluated once and passed as the first argument.
                // Desugaring at parse time would evaluate it twice, so a receiver
                // with side effects (`get_thing():act()`) would run twice.
                let receiver = self.eval(receiver_expr)?;
                let function = self.index(&receiver, &Value::str(method.as_str()))?;
                let mut args = vec![receiver];
                args.extend(self.eval_multi_all(arg_exprs)?);
                self.call_value(&function, &args)
                    .map_err(|e| err(format!("in method {method:?}: {e}")))
            }
            other => Ok(vec![self.eval(other)?]),
        }
    }

    /// Call a callable value with the given arguments.
    pub fn call_value(&mut self, callee: &Value, args: &[Value]) -> Result<Vec<Value>> {
        self.step()?;
        match callee {
            Value::Native(native) => (native.function)(args).map_err(err),
            Value::Function(function) => {
                self.call_depth += 1;
                if self.call_depth > MAX_CALL_DEPTH {
                    self.call_depth -= 1;
                    return Err(err(format!(
                        "call nesting deeper than {MAX_CALL_DEPTH} (infinite recursion?)"
                    )));
                }
                let result = self.call_lua(function, args);
                self.call_depth -= 1;
                result
            }
            other => Err(err(format!(
                "attempt to call a {} value",
                other.type_name()
            ))),
        }
    }

    fn call_lua(&mut self, function: &Rc<LuaFunction>, args: &[Value]) -> Result<Vec<Value>> {
        let body = function.body.clone();

        // The callee's scope chain is the one it *closed over*, not the caller's.
        // Using the caller's would give dynamic scoping, where a function sees
        // whatever locals happen to be live at the call site — which is a
        // different language and a source of bugs that look like haunting.
        let saved_scopes = std::mem::replace(&mut self.scopes, function.captured.clone());
        self.push_scope();

        for (i, param) in body.params.iter().enumerate() {
            self.declare_local(param, args.get(i).cloned().unwrap_or(Value::Nil));
        }
        // Extra arguments become `...`, or are discarded for a fixed-arity
        // function — Lua accepts both rather than erroring on arity.
        let extra = if body.is_vararg && args.len() > body.params.len() {
            args[body.params.len()..].to_vec()
        } else {
            Vec::new()
        };
        self.varargs.push(extra);

        let flow = self.exec_block(&body.body);

        self.varargs.pop();
        self.pop_scope();
        self.scopes = saved_scopes;

        match flow? {
            Flow::Return(values) => Ok(values),
            _ => Ok(Vec::new()),
        }
    }
}

/// Apply a binary operator to two evaluated values.
///
/// Free-standing so the operator semantics can be tested without an interpreter,
/// and so `stdlib` can reuse the comparison for `table.sort`.
pub fn binary_op(op: BinOp, l: &Value, r: &Value) -> std::result::Result<Value, String> {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
            let (a, b) = (l.as_number(), r.as_number());
            let (Some(a), Some(b)) = (a, b) else {
                return Err(format!(
                    "cannot perform arithmetic on {} and {}",
                    l.type_name(),
                    r.type_name()
                ));
            };
            Ok(Value::Number(match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                // Lua does not special-case division by zero: it produces inf, and
                // erroring here would diverge from the language.
                BinOp::Div => a / b,
                // Lua's `%` follows the sign of the *divisor*, unlike Rust's `%`
                // which follows the dividend: `-1 % 3` is 2 in Lua and -1 in Rust.
                BinOp::Mod => a - (a / b).floor() * b,
                BinOp::Pow => a.powf(b),
                _ => unreachable!(),
            }))
        }
        BinOp::Concat => {
            let (Some(a), Some(b)) = (l.as_str_coerced(), r.as_str_coerced()) else {
                return Err(format!(
                    "cannot concatenate {} and {}",
                    l.type_name(),
                    r.type_name()
                ));
            };
            Ok(Value::str(format!("{a}{b}")))
        }
        BinOp::Eq => Ok(Value::Bool(l == r)),
        BinOp::NotEq => Ok(Value::Bool(l != r)),
        BinOp::Less | BinOp::LessEq | BinOp::Greater | BinOp::GreaterEq => {
            let ordering = compare(l, r)?;
            Ok(Value::Bool(match op {
                BinOp::Less => ordering == std::cmp::Ordering::Less,
                BinOp::LessEq => ordering != std::cmp::Ordering::Greater,
                BinOp::Greater => ordering == std::cmp::Ordering::Greater,
                BinOp::GreaterEq => ordering != std::cmp::Ordering::Less,
                _ => unreachable!(),
            }))
        }
        BinOp::And | BinOp::Or => {
            // Handled by the caller, which must short-circuit.
            Ok(Value::Bool(l.truthy() && r.truthy()))
        }
    }
}

/// Order two values as Lua's relational operators do.
///
/// Numbers compare with numbers and strings with strings, and mixing them is an
/// error rather than a coercion — `"10" < 5` is an error in Lua, not `false`. This
/// asymmetry with arithmetic (where `"10" + 5` is 15) is deliberate in the
/// language: a silent comparison of a string against a number is almost always a
/// bug, while the arithmetic case is usually intentional.
pub fn compare(l: &Value, r: &Value) -> std::result::Result<std::cmp::Ordering, String> {
    match (l, r) {
        (Value::Number(a), Value::Number(b)) => a
            .partial_cmp(b)
            .ok_or_else(|| "cannot compare NaN".to_string()),
        (Value::Str(a), Value::Str(b)) => Ok(a.as_ref().cmp(b.as_ref())),
        _ => Err(format!(
            "cannot compare {} with {}",
            l.type_name(),
            r.type_name()
        )),
    }
}

/// `tostring`, shared with the standard library.
pub fn tostring(value: &Value) -> String {
    match value {
        Value::Number(n) => format_number(*n),
        other => other.to_display_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a chunk and return its first result.
    fn eval_expr(source: &str) -> Value {
        let mut interp = Interpreter::new();
        interp
            .run(&format!("return {source}"))
            .unwrap_or_else(|e| panic!("failed on {source:?}: {e}"))
            .into_iter()
            .next()
            .unwrap_or(Value::Nil)
    }

    fn run_ok(source: &str) -> Interpreter {
        let mut interp = Interpreter::new();
        interp
            .run(source)
            .unwrap_or_else(|e| panic!("failed to run:\n{source}\nerror: {e}"));
        interp
    }

    fn number(source: &str) -> f64 {
        match eval_expr(source) {
            Value::Number(n) => n,
            other => panic!("{source:?} produced {other:?}, not a number"),
        }
    }

    fn string(source: &str) -> String {
        match eval_expr(source) {
            Value::Str(s) => s.to_string(),
            other => panic!("{source:?} produced {other:?}, not a string"),
        }
    }

    // --- arithmetic ---

    #[test]
    fn arithmetic_works() {
        assert_eq!(number("1 + 2"), 3.0);
        assert_eq!(number("7 - 3"), 4.0);
        assert_eq!(number("3 * 4"), 12.0);
        assert_eq!(number("10 / 4"), 2.5);
        assert_eq!(number("2 ^ 10"), 1024.0);
        assert_eq!(number("-5"), -5.0);
    }

    /// Precedence must survive evaluation, not just parsing.
    #[test]
    fn precedence_holds_through_evaluation() {
        assert_eq!(number("1 + 2 * 3"), 7.0);
        assert_eq!(number("(1 + 2) * 3"), 9.0);
        assert_eq!(number("2 ^ 3 ^ 2"), 512.0, "^ is right-associative");
        assert_eq!(number("-2 ^ 2"), -4.0, "unary minus binds looser than ^");
    }

    /// Lua's `%` follows the sign of the divisor; Rust's follows the dividend.
    /// `-1 % 3` is 2 in Lua and -1 in Rust, and using Rust's would silently break
    /// every wrap-around computation a script does.
    #[test]
    fn modulo_follows_lua_not_rust() {
        assert_eq!(number("-1 % 3"), 2.0);
        assert_eq!(number("1 % 3"), 1.0);
        assert_eq!(number("-7 % 3"), 2.0);
        assert_eq!(number("7 % -3"), -2.0);
    }

    /// Division by zero gives infinity in Lua rather than an error. Erroring would
    /// diverge from the language on an expression that is occasionally intentional.
    #[test]
    fn division_by_zero_is_infinity_not_an_error() {
        assert!(number("1 / 0").is_infinite());
        assert!(number("-1 / 0").is_infinite());
        assert!(number("0 / 0").is_nan());
    }

    /// `"10" + 5` is 15 in Lua: arithmetic coerces numeric strings.
    #[test]
    fn arithmetic_coerces_numeric_strings() {
        assert_eq!(number(r#""10" + 5"#), 15.0);
        assert_eq!(number(r#""2" * "3""#), 6.0);
    }

    #[test]
    fn arithmetic_on_a_non_number_is_an_error() {
        let mut interp = Interpreter::new();
        assert!(interp.run("return {} + 1").is_err());
        assert!(interp.run("return nil + 1").is_err());
        assert!(interp.run(r#"return "abc" + 1"#).is_err());
    }

    // --- comparison ---

    #[test]
    fn comparison_works_within_a_type() {
        assert_eq!(eval_expr("1 < 2"), Value::Bool(true));
        assert_eq!(eval_expr("2 <= 2"), Value::Bool(true));
        assert_eq!(eval_expr("3 > 4"), Value::Bool(false));
        assert_eq!(eval_expr(r#""a" < "b""#), Value::Bool(true));
        assert_eq!(eval_expr(r#""abc" < "abd""#), Value::Bool(true));
    }

    /// Unlike arithmetic, comparison does *not* coerce: `"10" < 5` is an error in
    /// Lua, because a silent comparison of a string against a number is almost
    /// always a bug.
    #[test]
    fn comparison_does_not_coerce_across_types() {
        let mut interp = Interpreter::new();
        assert!(interp.run(r#"return "10" < 5"#).is_err());
        assert!(interp.run("return nil < 1").is_err());
        assert!(interp.run("return {} < {}").is_err());
    }

    #[test]
    fn equality_works_across_types_without_erroring() {
        assert_eq!(eval_expr("1 == 1"), Value::Bool(true));
        assert_eq!(
            eval_expr(r#"1 == "1""#),
            Value::Bool(false),
            "no coercion in =="
        );
        assert_eq!(eval_expr("nil == false"), Value::Bool(false));
        assert_eq!(eval_expr("1 ~= 2"), Value::Bool(true));
    }

    // --- logical operators ---

    /// `and` and `or` return one of their operands, not a boolean — which is what
    /// makes `x or default` the standard idiom.
    #[test]
    fn logical_operators_return_a_value_not_a_boolean() {
        assert_eq!(number("nil or 5"), 5.0);
        assert_eq!(number("3 or 5"), 3.0);
        assert_eq!(number("3 and 5"), 5.0);
        assert_eq!(eval_expr("nil and 5"), Value::Nil);
        assert_eq!(eval_expr("false or nil"), Value::Nil);
    }

    /// Short-circuiting is not an optimisation, it is semantics: `t and t.x` must
    /// not index nil.
    #[test]
    fn logical_operators_short_circuit() {
        // If `and` evaluated its right side eagerly this would error.
        assert_eq!(eval_expr("nil and (nil).x"), Value::Nil);
        assert_eq!(number("1 or (nil).x"), 1.0);
        // And the side effect really does not happen.
        let interp = run_ok(
            "local calls = 0\n\
             local function bump() calls = calls + 1 return true end\n\
             local _ = false and bump()\n\
             result = calls",
        );
        assert_eq!(interp.get_global("result"), Value::Number(0.0));
    }

    #[test]
    fn not_and_length_work() {
        assert_eq!(eval_expr("not nil"), Value::Bool(true));
        assert_eq!(eval_expr("not 0"), Value::Bool(false), "0 is truthy in Lua");
        assert_eq!(number(r#"#"hello""#), 5.0);
        assert_eq!(number("#{1, 2, 3}"), 3.0);
    }

    // --- strings ---

    #[test]
    fn concatenation_coerces_numbers() {
        assert_eq!(string(r#""a" .. "b""#), "ab");
        assert_eq!(string(r#""n=" .. 42"#), "n=42");
        assert_eq!(
            string(r#""x=" .. 1.5"#),
            "x=1.5",
            "a fractional number keeps its digits"
        );
    }

    /// An integral number concatenates without a decimal point, matching Lua.
    #[test]
    fn concatenating_an_integral_number_omits_the_decimal_point() {
        assert_eq!(string(r#""" .. 2"#), "2");
        assert_eq!(string(r#""" .. (1 + 1)"#), "2");
    }

    #[test]
    fn concatenating_a_boolean_or_nil_is_an_error() {
        let mut interp = Interpreter::new();
        assert!(interp.run(r#"return "x" .. true"#).is_err());
        assert!(interp.run(r#"return "x" .. nil"#).is_err());
    }

    // --- variables and scope ---

    #[test]
    fn locals_and_globals_are_distinct() {
        let interp = run_ok("g = 1\nlocal l = 2\nresult = l");
        assert_eq!(interp.get_global("g"), Value::Number(1.0));
        assert_eq!(interp.get_global("result"), Value::Number(2.0));
        assert_eq!(
            interp.get_global("l"),
            Value::Nil,
            "a local must not leak into the globals"
        );
    }

    /// An assignment with no `local` writes a global *unless* a local of that name
    /// is in scope. It is easy to get backwards, and getting it backwards makes
    /// every function silently clobber its caller's variables.
    #[test]
    fn assignment_prefers_an_in_scope_local_over_a_global() {
        let interp = run_ok(
            "x = \"global\"\n\
             local function f()\n\
               local x = \"local\"\n\
               x = \"changed\"\n\
               return x\n\
             end\n\
             inner = f()\n\
             outer = x",
        );
        assert_eq!(interp.get_global("inner"), Value::str("changed"));
        assert_eq!(
            interp.get_global("outer"),
            Value::str("global"),
            "the function must not have touched the global"
        );
    }

    #[test]
    fn a_block_scopes_its_locals() {
        let interp = run_ok(
            "local x = 1\n\
             do local x = 2 end\n\
             result = x",
        );
        assert_eq!(interp.get_global("result"), Value::Number(1.0));
    }

    // --- control flow ---

    #[test]
    fn if_elseif_else_picks_one_branch() {
        let interp = run_ok(
            "local function classify(n)\n\
               if n < 0 then return \"neg\"\n\
               elseif n == 0 then return \"zero\"\n\
               else return \"pos\" end\n\
             end\n\
             a = classify(-1)\n\
             b = classify(0)\n\
             c = classify(1)",
        );
        assert_eq!(interp.get_global("a"), Value::str("neg"));
        assert_eq!(interp.get_global("b"), Value::str("zero"));
        assert_eq!(interp.get_global("c"), Value::str("pos"));
    }

    #[test]
    fn while_loops_and_break_works() {
        let interp = run_ok(
            "local i, sum = 1, 0\n\
             while true do\n\
               if i > 5 then break end\n\
               sum = sum + i\n\
               i = i + 1\n\
             end\n\
             result = sum",
        );
        assert_eq!(interp.get_global("result"), Value::Number(15.0));
    }

    /// `repeat`'s condition sees the body's locals — the one place Lua's scoping is
    /// not block-shaped. Splitting them would break `repeat local x = f() until x`.
    #[test]
    fn repeat_until_sees_the_bodys_locals() {
        let interp = run_ok(
            "local n = 0\n\
             repeat\n\
               n = n + 1\n\
               local done = n >= 3\n\
             until done\n\
             result = n",
        );
        assert_eq!(interp.get_global("result"), Value::Number(3.0));
    }

    #[test]
    fn repeat_runs_its_body_at_least_once() {
        let interp = run_ok("local n = 0\nrepeat n = n + 1 until true\nresult = n");
        assert_eq!(interp.get_global("result"), Value::Number(1.0));
    }

    #[test]
    fn numeric_for_counts_up_and_down() {
        let interp = run_ok(
            "local up = 0\n\
             for i = 1, 5 do up = up + i end\n\
             local down = \"\"\n\
             for i = 3, 1, -1 do down = down .. i end\n\
             sum = up\n\
             order = down",
        );
        assert_eq!(interp.get_global("sum"), Value::Number(15.0));
        assert_eq!(
            interp.get_global("order"),
            Value::str("321"),
            "a negative step must count down and terminate"
        );
    }

    #[test]
    fn a_numeric_for_with_a_zero_step_is_an_error_not_a_hang() {
        let mut interp = Interpreter::new();
        let e = interp.run("for i = 1, 10, 0 do end").unwrap_err();
        assert!(e.to_string().contains("step"), "{e}");
    }

    #[test]
    fn a_for_loop_variable_does_not_escape() {
        let interp = run_ok("for i = 1, 3 do end\nresult = i");
        assert_eq!(interp.get_global("result"), Value::Nil);
    }

    // --- functions ---

    #[test]
    fn functions_take_arguments_and_return_values() {
        let interp = run_ok(
            "local function add(a, b) return a + b end\n\
             result = add(2, 3)",
        );
        assert_eq!(interp.get_global("result"), Value::Number(5.0));
    }

    #[test]
    fn functions_return_multiple_values() {
        let interp = run_ok(
            "local function two() return 1, 2 end\n\
             local a, b = two()\n\
             x, y = a, b",
        );
        assert_eq!(interp.get_global("x"), Value::Number(1.0));
        assert_eq!(interp.get_global("y"), Value::Number(2.0));
    }

    /// Lua truncates every call but the last in an expression list. Expanding all
    /// of them would make argument counts unpredictable.
    #[test]
    fn only_the_last_call_in_a_list_expands() {
        let interp = run_ok(
            "local function two() return 1, 2 end\n\
             local t = { two(), two() }\n\
             result = #t",
        );
        assert_eq!(
            interp.get_global("result"),
            Value::Number(3.0),
            "the first call contributes 1 value, the last contributes 2"
        );
    }

    /// A local function must be able to call itself, which means its name has to
    /// be in scope before the body closes over it.
    #[test]
    fn a_local_function_can_recurse() {
        let interp = run_ok(
            "local function fact(n)\n\
               if n <= 1 then return 1 end\n\
               return n * fact(n - 1)\n\
             end\n\
             result = fact(5)",
        );
        assert_eq!(interp.get_global("result"), Value::Number(120.0));
    }

    /// A closure must capture the scope by handle, so mutating an outer local is
    /// visible to everything else holding it — the basis of counters and state.
    #[test]
    fn closures_share_the_captured_scope() {
        let interp = run_ok(
            "local function counter()\n\
               local n = 0\n\
               return function() n = n + 1 return n end\n\
             end\n\
             local c = counter()\n\
             c()\n\
             c()\n\
             result = c()",
        );
        assert_eq!(interp.get_global("result"), Value::Number(3.0));
    }

    /// Two closures from separate calls must have separate state.
    #[test]
    fn separate_closures_have_separate_state() {
        let interp = run_ok(
            "local function counter()\n\
               local n = 0\n\
               return function() n = n + 1 return n end\n\
             end\n\
             local a, b = counter(), counter()\n\
             a() a() a()\n\
             first = a()\n\
             second = b()",
        );
        assert_eq!(interp.get_global("first"), Value::Number(4.0));
        assert_eq!(
            interp.get_global("second"),
            Value::Number(1.0),
            "the second counter must not share the first's state"
        );
    }

    /// A function sees the scope it was *defined* in, not the one it is called
    /// from. Dynamic scoping produces bugs that look like haunting.
    #[test]
    fn functions_use_lexical_not_dynamic_scope() {
        let interp = run_ok(
            "local x = \"outer\"\n\
             local function f() return x end\n\
             local function g()\n\
               local x = \"inner\"\n\
               return f()\n\
             end\n\
             result = g()",
        );
        assert_eq!(
            interp.get_global("result"),
            Value::str("outer"),
            "f must see its own definition scope, not g's"
        );
    }

    #[test]
    fn missing_arguments_are_nil_and_extra_ones_are_dropped() {
        let interp = run_ok(
            "local function f(a, b) return a, b end\n\
             local x, y = f(1)\n\
             local p = f(1, 2, 3)\n\
             first = x\n\
             second = y\n\
             third = p",
        );
        assert_eq!(interp.get_global("first"), Value::Number(1.0));
        assert_eq!(interp.get_global("second"), Value::Nil);
        assert_eq!(interp.get_global("third"), Value::Number(1.0));
    }

    #[test]
    fn varargs_collect_the_extra_arguments() {
        let interp = run_ok(
            "local function count(...)\n\
               local t = {...}\n\
               return #t\n\
             end\n\
             result = count(1, 2, 3, 4)",
        );
        assert_eq!(interp.get_global("result"), Value::Number(4.0));
    }

    #[test]
    fn varargs_come_after_the_named_parameters() {
        let interp = run_ok(
            "local function f(first, ...)\n\
               local rest = {...}\n\
               return first, #rest\n\
             end\n\
             local a, b = f(\"x\", 1, 2)\n\
             head = a\n\
             tail = b",
        );
        assert_eq!(interp.get_global("head"), Value::str("x"));
        assert_eq!(interp.get_global("tail"), Value::Number(2.0));
    }

    #[test]
    fn calling_a_non_function_is_an_error_naming_it() {
        let mut interp = Interpreter::new();
        let e = interp.run("undefined_function()").unwrap_err();
        assert!(
            e.to_string().contains("undefined_function"),
            "the error should name the callee: {e}"
        );
    }

    // --- tables ---

    #[test]
    fn tables_index_by_number_and_string() {
        let interp = run_ok(
            "local t = { 10, 20, name = \"thing\" }\n\
             a = t[1]\n\
             b = t[2]\n\
             c = t.name\n\
             d = t[\"name\"]\n\
             e = t.missing",
        );
        assert_eq!(interp.get_global("a"), Value::Number(10.0));
        assert_eq!(interp.get_global("b"), Value::Number(20.0));
        assert_eq!(interp.get_global("c"), Value::str("thing"));
        assert_eq!(
            interp.get_global("d"),
            Value::str("thing"),
            "t.name and t[\"name\"] must be the same slot"
        );
        assert_eq!(interp.get_global("e"), Value::Nil);
    }

    #[test]
    fn tables_can_be_written_and_nested() {
        let interp = run_ok(
            "local t = {}\n\
             t.x = 1\n\
             t[2] = \"two\"\n\
             t.nested = { deep = true }\n\
             a = t.x\n\
             b = t[2]\n\
             c = t.nested.deep",
        );
        assert_eq!(interp.get_global("a"), Value::Number(1.0));
        assert_eq!(interp.get_global("b"), Value::str("two"));
        assert_eq!(interp.get_global("c"), Value::Bool(true));
    }

    /// `t[1]` written by a literal and read by a loop counter must be one slot.
    #[test]
    fn integer_and_float_keys_address_the_same_slot() {
        let interp = run_ok(
            "local t = {}\n\
             t[1] = \"one\"\n\
             local found\n\
             for i = 1, 1 do found = t[i] end\n\
             result = found",
        );
        assert_eq!(interp.get_global("result"), Value::str("one"));
    }

    #[test]
    fn indexing_a_non_table_is_an_error() {
        let mut interp = Interpreter::new();
        assert!(interp.run("return (nil).x").is_err());
        assert!(interp.run("local n = 5 return n.field").is_err());
    }

    // --- methods ---

    #[test]
    fn method_calls_pass_the_receiver_as_self() {
        let interp = run_ok(
            "local obj = { value = 7 }\n\
             function obj:get() return self.value end\n\
             result = obj:get()",
        );
        assert_eq!(interp.get_global("result"), Value::Number(7.0));
    }

    /// The receiver must be evaluated once. Desugaring at parse time would
    /// evaluate it twice, so a receiver with side effects would run twice.
    #[test]
    fn a_method_receiver_is_evaluated_exactly_once() {
        let interp = run_ok(
            "calls = 0\n\
             local obj = { greet = function(self) return \"hi\" end }\n\
             local function get()\n\
               calls = calls + 1\n\
               return obj\n\
             end\n\
             get():greet()",
        );
        assert_eq!(interp.get_global("calls"), Value::Number(1.0));
    }

    // --- limits ---

    /// `while true do end` must return an error rather than hang the render loop.
    /// A frozen window with no message is indistinguishable from a crash.
    #[test]
    fn an_infinite_loop_hits_the_step_limit_instead_of_hanging() {
        let mut interp = Interpreter::new();
        interp.set_step_limit(10_000);
        let e = interp.run("while true do end").unwrap_err();
        assert!(e.to_string().contains("steps"), "{e}");
    }

    #[test]
    fn a_long_but_finite_loop_completes() {
        let mut interp = Interpreter::new();
        let result = interp.run(
            "local sum = 0\n\
             for i = 1, 1000 do sum = sum + i end\n\
             return sum",
        );
        assert_eq!(result.unwrap()[0], Value::Number(500500.0));
    }

    /// Infinite recursion must be an error, not a process abort: a stack overflow
    /// cannot be caught in safe Rust.
    #[test]
    fn infinite_recursion_errors_rather_than_overflowing_the_stack() {
        let mut interp = Interpreter::new();
        let e = interp
            .run("local function f() return f() end\nreturn f()")
            .unwrap_err();
        assert!(
            e.to_string().contains("nesting") || e.to_string().contains("recursion"),
            "{e}"
        );
    }

    #[test]
    fn ordinary_recursion_depth_is_allowed() {
        let mut interp = Interpreter::new();
        let result = interp.run(
            "local function count(n)\n\
               if n <= 0 then return 0 end\n\
               return 1 + count(n - 1)\n\
             end\n\
             return count(50)",
        );
        assert_eq!(result.unwrap()[0], Value::Number(50.0));
    }

    // --- host interface ---

    #[test]
    fn a_native_function_is_callable_from_lua() {
        let mut interp = Interpreter::new();
        interp.set_native("double", |args| {
            let n = args.first().and_then(|v| v.as_number()).unwrap_or(0.0);
            Ok(vec![Value::Number(n * 2.0)])
        });
        let result = interp.run("return double(21)").unwrap();
        assert_eq!(result[0], Value::Number(42.0));
    }

    #[test]
    fn a_native_function_can_report_an_error() {
        let mut interp = Interpreter::new();
        interp.set_native("boom", |_| Err("deliberate failure".to_string()));
        let e = interp.run("boom()").unwrap_err();
        assert!(e.to_string().contains("deliberate failure"), "{e}");
    }

    /// The engine's use of scripting: a script declares a handler and Rust calls it
    /// every frame.
    #[test]
    fn the_host_can_call_a_script_defined_function() {
        let mut interp = Interpreter::new();
        interp
            .run("function update(dt) return dt * 2 end")
            .unwrap();
        assert!(interp.has_function("update"));
        let result = interp
            .call_global("update", &[Value::Number(0.5)])
            .unwrap();
        assert_eq!(result[0], Value::Number(1.0));
    }

    #[test]
    fn calling_a_missing_global_is_an_error_not_a_panic() {
        let mut interp = Interpreter::new();
        assert!(!interp.has_function("nope"));
        assert!(interp.call_global("nope", &[]).is_err());
    }

    /// Hot-reload runs chunks repeatedly. Locals from one must not leak into the
    /// next, and a redefinition must take effect.
    #[test]
    fn chunks_do_not_leak_locals_and_can_be_redefined() {
        let mut interp = Interpreter::new();
        interp.run("local hidden = 1\nfunction f() return 1 end").unwrap();
        assert_eq!(interp.get_global("hidden"), Value::Nil);
        assert_eq!(
            interp.call_global("f", &[]).unwrap()[0],
            Value::Number(1.0)
        );
        // Reload with a new definition.
        interp.run("function f() return 2 end").unwrap();
        assert_eq!(
            interp.call_global("f", &[]).unwrap()[0],
            Value::Number(2.0),
            "the redefinition must replace the old function"
        );
    }

    /// State must survive a reload, so a script can keep accumulated data across
    /// edits — which is most of the value of hot-reload.
    #[test]
    fn globals_survive_a_reload() {
        let mut interp = Interpreter::new();
        interp.run("counter = 5").unwrap();
        interp.run("function bump() counter = counter + 1 end").unwrap();
        interp.call_global("bump", &[]).unwrap();
        assert_eq!(interp.get_global("counter"), Value::Number(6.0));
    }

    // --- errors ---

    /// Every error must name a line, or a hot-reload failure gives the author
    /// nothing to go on.
    #[test]
    fn syntax_errors_carry_a_line_number() {
        let mut interp = Interpreter::new();
        let e = interp.run("local x = \nlocal y = 1").unwrap_err();
        assert!(e.to_string().contains("line"), "{e}");
    }

    #[test]
    fn a_failing_script_leaves_the_interpreter_usable() {
        let mut interp = Interpreter::new();
        interp.run("good = 1").unwrap();
        assert!(interp.run("return nil + 1").is_err());
        // Still works afterwards: a failed chunk must not corrupt the state.
        interp.run("also_good = 2").unwrap();
        assert_eq!(interp.get_global("good"), Value::Number(1.0));
        assert_eq!(interp.get_global("also_good"), Value::Number(2.0));
    }

    /// A script that errors mid-call must not leave the scope stack unbalanced, or
    /// every later lookup would resolve in the wrong frame.
    #[test]
    fn an_error_inside_a_call_restores_the_scope_stack() {
        let mut interp = Interpreter::new();
        interp
            .run(
                "outer = \"visible\"\n\
                 function boom()\n\
                   local hidden = 1\n\
                   return nil + 1\n\
                 end",
            )
            .unwrap();
        assert!(interp.call_global("boom", &[]).is_err());
        // The scope stack must be back to empty, so a fresh chunk sees globals.
        let result = interp.run("return outer").unwrap();
        assert_eq!(result[0], Value::str("visible"));
    }

    #[test]
    fn an_empty_chunk_is_valid_and_returns_nothing() {
        let mut interp = Interpreter::new();
        assert!(interp.run("").unwrap().is_empty());
        assert!(interp.run("-- just a comment").unwrap().is_empty());
    }

    #[test]
    fn steps_are_counted_and_reported() {
        let mut interp = Interpreter::new();
        interp.run("local x = 1").unwrap();
        let small = interp.steps();
        interp.run("local s = 0\nfor i = 1, 100 do s = s + i end").unwrap();
        assert!(
            interp.steps() > small,
            "a loop should cost more steps than an assignment: {} vs {small}",
            interp.steps()
        );
    }
}
