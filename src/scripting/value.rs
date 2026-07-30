// Lua values and tables.
//
// Numbers are `f64` only, as in Lua 5.1: there is no integer subtype, so `1` and
// `1.0` are the same value and the same table key. That is what makes `t[1]` from
// a script and `t[1.0]` from a loop counter reach the same slot.
//
// Tables carry both an array part and a hash part, because Lua's `#` operator,
// `ipairs` and the `table` library are all defined over a contiguous run of
// integer keys from 1, and recovering that run from a hash map every time it is
// needed would make every length query O(n) with a bad constant.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::scripting::parser::FunctionBody;

/// A Lua value.
///
/// `Rc` for tables and functions because Lua's semantics are reference semantics:
/// `local a = {} ; local b = a ; b.x = 1` must be visible through `a`. Cloning the
/// table would make that assignment invisible, which is the single most confusing
/// way a scripting language can be wrong.
#[derive(Clone)]
pub enum Value {
    Nil,
    Bool(bool),
    Number(f64),
    /// `Rc<str>` rather than `String`: strings are copied constantly (every table
    /// key lookup, every concatenation result passed along) and they are immutable
    /// in Lua, so sharing them is free and correct.
    Str(Rc<str>),
    Table(Rc<RefCell<Table>>),
    /// A function defined in Lua, with the environment it closed over.
    Function(Rc<LuaFunction>),
    /// A function implemented in Rust and exposed to scripts.
    Native(Rc<NativeFunction>),
}

/// A Lua-defined function and its captured scope.
pub struct LuaFunction {
    pub body: Rc<FunctionBody>,
    /// Captured scopes, outermost first. Shared rather than copied, so a closure
    /// that mutates an outer local is visible to everything else holding that
    /// scope — which is what `local n = 0; return function() n = n + 1 end` means.
    pub captured: Vec<Rc<RefCell<HashMap<String, Value>>>>,
}

/// A Rust function callable from Lua.
pub struct NativeFunction {
    pub name: String,
    /// Takes the argument list, returns the result list. Multiple returns are the
    /// norm in Lua, so the signature is plural in both directions rather than
    /// wrapping a single value.
    pub function: Box<dyn Fn(&[Value]) -> Result<Vec<Value>, String>>,
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nil => write!(f, "nil"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Number(n) => write!(f, "{}", format_number(*n)),
            Self::Str(s) => write!(f, "{s:?}"),
            Self::Table(t) => write!(f, "table<{} entries>", t.borrow().len()),
            Self::Function(_) => write!(f, "function"),
            Self::Native(n) => write!(f, "native<{}>", n.name),
        }
    }
}

impl Value {
    /// A string value from anything string-like.
    pub fn str(text: impl AsRef<str>) -> Self {
        Self::Str(Rc::from(text.as_ref()))
    }

    /// An empty table.
    pub fn table() -> Self {
        Self::Table(Rc::new(RefCell::new(Table::new())))
    }

    /// Lua's truthiness: only `nil` and `false` are false.
    ///
    /// Notably 0 and "" are *true*, unlike most languages. Getting this wrong
    /// makes `if count then` behave differently from Lua for the one value the
    /// author most likely cared about.
    pub fn truthy(&self) -> bool {
        !matches!(self, Self::Nil | Self::Bool(false))
    }

    /// The name `type()` reports.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Nil => "nil",
            Self::Bool(_) => "boolean",
            Self::Number(_) => "number",
            Self::Str(_) => "string",
            Self::Table(_) => "table",
            // Both callable kinds report "function": a script has no business
            // knowing which side of the boundary a function came from.
            Self::Function(_) | Self::Native(_) => "function",
        }
    }

    /// Numeric value, coercing a numeric string as Lua does.
    ///
    /// The coercion is why `"10" + 5` is 15 in Lua. It applies to arithmetic, not
    /// to comparison — `"10" < 5` is an error, not `false`.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            Self::Str(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        }
    }

    /// String value, coercing a number as Lua does for concatenation.
    pub fn as_str_coerced(&self) -> Option<String> {
        match self {
            Self::Str(s) => Some(s.to_string()),
            Self::Number(n) => Some(format_number(*n)),
            _ => None,
        }
    }

    /// Whether this value can be called.
    pub fn is_callable(&self) -> bool {
        matches!(self, Self::Function(_) | Self::Native(_))
    }

    /// The text `tostring()` produces.
    pub fn to_display_string(&self) -> String {
        match self {
            Self::Nil => "nil".to_string(),
            Self::Bool(b) => b.to_string(),
            Self::Number(n) => format_number(*n),
            Self::Str(s) => s.to_string(),
            // Identity in the address, matching Lua's `table: 0x...`. Useful
            // because it lets a script tell two tables apart in a log.
            Self::Table(t) => format!("table: {:p}", Rc::as_ptr(t)),
            Self::Function(func) => format!("function: {:p}", Rc::as_ptr(func)),
            Self::Native(n) => format!("function: builtin {}", n.name),
        }
    }
}

/// Format a number the way Lua's `tostring` does.
///
/// An integral float prints without a decimal point: `print(1 + 1)` must say `2`,
/// not `2.0`. This is the single most visible difference between a Lua-like
/// language and Lua, because it shows up in the first line of output anyone writes.
pub fn format_number(n: f64) -> String {
    if n.is_nan() {
        return "nan".to_string();
    }
    if n.is_infinite() {
        return if n > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if n == n.trunc() && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        // Lua uses "%.14g". The nearest equivalent here is 14 significant digits
        // with trailing zeros trimmed.
        let mut text = format!("{n:.14}");
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
}

impl PartialEq for Value {
    /// Lua's `==`: values of different types are never equal (no coercion),
    /// tables and functions compare by identity.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Nil, Self::Nil) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Table(a), Self::Table(b)) => Rc::ptr_eq(a, b),
            (Self::Function(a), Self::Function(b)) => Rc::ptr_eq(a, b),
            (Self::Native(a), Self::Native(b)) => Rc::ptr_eq(a, b),
            // Deliberately no cross-type arm: `1 == "1"` is false in Lua, and
            // making it true would be a silent departure from the language.
            _ => false,
        }
    }
}

/// A table key.
///
/// Only the hashable value types can be keys: `nil` is an error in Lua and a
/// float key is normalized to an integer when it is integral, so `t[1]` and
/// `t[1.0]` are one slot. Without that normalization a loop counter and a literal
/// would address different entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Bool(bool),
    /// Integral numbers, as an integer so the hash is exact.
    Int(i64),
    /// Non-integral numbers, by their bit pattern. Exact rather than approximate:
    /// two keys are the same slot only if they are the same number.
    NumberBits(u64),
    Str(Rc<str>),
    /// A table or function used as a key, by address.
    Pointer(usize),
}

impl Key {
    /// The key for a value, or `None` for `nil` (and NaN, which Lua also rejects
    /// because `NaN ~= NaN` makes such a key unretrievable).
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Nil => None,
            Value::Bool(b) => Some(Self::Bool(*b)),
            Value::Number(n) => {
                if n.is_nan() {
                    None
                } else if *n == n.trunc() && n.abs() < 9.0e15 {
                    Some(Self::Int(*n as i64))
                } else {
                    Some(Self::NumberBits(n.to_bits()))
                }
            }
            Value::Str(s) => Some(Self::Str(s.clone())),
            Value::Table(t) => Some(Self::Pointer(Rc::as_ptr(t) as *const u8 as usize)),
            Value::Function(f) => Some(Self::Pointer(Rc::as_ptr(f) as *const u8 as usize)),
            Value::Native(n) => Some(Self::Pointer(Rc::as_ptr(n) as *const u8 as usize)),
        }
    }

    /// The value this key came from, for iteration.
    pub fn to_value(&self) -> Value {
        match self {
            Self::Bool(b) => Value::Bool(*b),
            Self::Int(i) => Value::Number(*i as f64),
            Self::NumberBits(bits) => Value::Number(f64::from_bits(*bits)),
            Self::Str(s) => Value::Str(s.clone()),
            // A pointer key cannot be turned back into its value; iteration over
            // a table keyed by tables reports nil for those keys rather than
            // fabricating a value. Rare enough in game scripts to be acceptable,
            // and documented rather than silent.
            Self::Pointer(_) => Value::Nil,
        }
    }
}

/// A Lua table: a contiguous array part plus a hash part.
#[derive(Default)]
pub struct Table {
    /// Values for keys 1..=array.len(), in order. Holds the run `#` measures.
    array: Vec<Value>,
    /// Everything else.
    hash: HashMap<Key, Value>,
    /// Optional `__index` fallback, the one metatable feature implemented.
    ///
    /// Only `__index`, because that is what makes prototype-style objects and
    /// method tables work — which is the whole reason a game script wants
    /// metatables. The rest (`__add`, `__eq`, `__call`) would each need a hook in
    /// the interpreter's operator paths, and none of them earns that here.
    index_fallback: Option<Value>,
}

impl Table {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a key. Falls through to `__index` when the key is absent.
    pub fn get(&self, key: &Key) -> Value {
        if let Key::Int(i) = key {
            if *i >= 1 {
                if let Some(value) = self.array.get((*i - 1) as usize) {
                    return value.clone();
                }
            }
        }
        if let Some(value) = self.hash.get(key) {
            return value.clone();
        }
        // `__index` is consulted only for a *missing* key, never to shadow a
        // present one — including one present and nil, which in Lua means absent.
        if let Some(fallback) = &self.index_fallback {
            if let Value::Table(parent) = fallback {
                return parent.borrow().get(key);
            }
        }
        Value::Nil
    }

    /// Read without the `__index` fallback: the table's own contents only.
    pub fn get_own(&self, key: &Key) -> Value {
        if let Key::Int(i) = key {
            if *i >= 1 {
                if let Some(value) = self.array.get((*i - 1) as usize) {
                    return value.clone();
                }
            }
        }
        self.hash.get(key).cloned().unwrap_or(Value::Nil)
    }

    /// Write a key. Assigning `nil` removes the entry, as in Lua.
    pub fn set(&mut self, key: Key, value: Value) {
        if let Key::Int(i) = key {
            if i >= 1 {
                let index = (i - 1) as usize;
                if index < self.array.len() {
                    if matches!(value, Value::Nil) && index == self.array.len() - 1 {
                        // Removing the last element shrinks the array part, so `#`
                        // follows. Removing from the middle cannot: it would leave
                        // a hole, and Lua's `#` on a table with holes is
                        // explicitly unspecified.
                        self.array.pop();
                        // Any trailing nils exposed by the pop go too.
                        while matches!(self.array.last(), Some(Value::Nil)) {
                            self.array.pop();
                        }
                    } else {
                        self.array[index] = value;
                    }
                    return;
                }
                if index == self.array.len() && !matches!(value, Value::Nil) {
                    // Extends the run. Anything in the hash part that now
                    // continues it migrates across, so `t[1]=a; t[3]=c; t[2]=b`
                    // ends with a length of 3 rather than 1.
                    self.array.push(value);
                    self.absorb_from_hash();
                    return;
                }
            }
        }
        if matches!(value, Value::Nil) {
            self.hash.remove(&key);
        } else {
            self.hash.insert(key, value);
        }
    }

    /// Move keys that now continue the array run out of the hash part.
    fn absorb_from_hash(&mut self) {
        loop {
            let next = self.array.len() as i64 + 1;
            match self.hash.remove(&Key::Int(next)) {
                Some(value) => self.array.push(value),
                None => break,
            }
        }
    }

    /// `#t`: the length of the array part.
    pub fn length(&self) -> usize {
        self.array.len()
    }

    /// Total entries, array and hash.
    pub fn len(&self) -> usize {
        self.array.len() + self.hash.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append to the array part, as `table.insert` does.
    pub fn push(&mut self, value: Value) {
        if matches!(value, Value::Nil) {
            // Appending nil would create a hole immediately.
            return;
        }
        self.array.push(value);
        self.absorb_from_hash();
    }

    /// Remove and return the last array element.
    pub fn pop(&mut self) -> Value {
        self.array.pop().unwrap_or(Value::Nil)
    }

    /// The array part, for `ipairs` and the `table` library.
    pub fn array_part(&self) -> &[Value] {
        &self.array
    }

    /// Mutable access to the array part, for `table.sort`.
    pub fn array_part_mut(&mut self) -> &mut Vec<Value> {
        &mut self.array
    }

    /// Every key, array part first then hash part.
    ///
    /// Collected rather than returned as an iterator because `pairs` has to be
    /// able to run script code in its loop body, which may mutate the table — and
    /// a live iterator over a `RefCell` borrow would panic the moment it did.
    pub fn keys(&self) -> Vec<Key> {
        let mut keys: Vec<Key> = (1..=self.array.len() as i64).map(Key::Int).collect();
        keys.extend(self.hash.keys().cloned());
        keys
    }

    /// Set the `__index` fallback.
    pub fn set_index_fallback(&mut self, value: Value) {
        self.index_fallback = if matches!(value, Value::Nil) {
            None
        } else {
            Some(value)
        };
    }

    pub fn index_fallback(&self) -> Option<&Value> {
        self.index_fallback.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- truthiness ---

    /// Lua's truthiness is unusual: 0 and "" are TRUE. A script author writing
    /// `if count then` relies on it, and getting it wrong would silently diverge
    /// on the one value they most likely cared about.
    #[test]
    fn only_nil_and_false_are_falsy() {
        assert!(!Value::Nil.truthy());
        assert!(!Value::Bool(false).truthy());
        assert!(Value::Bool(true).truthy());
        assert!(Value::Number(0.0).truthy(), "0 is true in Lua");
        assert!(Value::str("").truthy(), "the empty string is true in Lua");
        assert!(Value::table().truthy());
    }

    // --- number formatting ---

    /// `print(1 + 1)` must say `2`, not `2.0`. It is the first line of output
    /// anyone writes, and the most visible way a Lua-like language betrays itself.
    #[test]
    fn integral_numbers_print_without_a_decimal_point() {
        assert_eq!(format_number(2.0), "2");
        assert_eq!(format_number(-7.0), "-7");
        assert_eq!(format_number(0.0), "0");
        assert_eq!(format_number(1e6), "1000000");
    }

    #[test]
    fn fractional_numbers_keep_their_digits_without_trailing_zeros() {
        assert_eq!(format_number(1.5), "1.5");
        assert_eq!(format_number(-0.25), "-0.25");
        assert_eq!(format_number(0.1), "0.1");
    }

    #[test]
    fn non_finite_numbers_have_readable_names() {
        assert_eq!(format_number(f64::NAN), "nan");
        assert_eq!(format_number(f64::INFINITY), "inf");
        assert_eq!(format_number(f64::NEG_INFINITY), "-inf");
    }

    // --- equality ---

    /// `1 == "1"` is false in Lua. Coercing here would be a silent departure from
    /// the language in the direction people expect from other languages.
    #[test]
    fn equality_never_coerces_across_types() {
        assert_ne!(Value::Number(1.0), Value::str("1"));
        assert_ne!(Value::Bool(true), Value::Number(1.0));
        assert_ne!(Value::Nil, Value::Bool(false));
    }

    #[test]
    fn numbers_and_strings_compare_by_value() {
        assert_eq!(Value::Number(1.0), Value::Number(1.0));
        assert_eq!(Value::str("abc"), Value::str("abc"));
        assert_ne!(Value::str("abc"), Value::str("abd"));
    }

    /// Tables compare by identity, not contents: two empty tables are different
    /// tables. This is what makes them usable as unique tokens.
    #[test]
    fn tables_compare_by_identity() {
        let a = Value::table();
        let b = Value::table();
        assert_ne!(a, b, "two distinct empty tables must not be equal");
        assert_eq!(a, a.clone(), "a clone shares the same table");
    }

    // --- reference semantics ---

    /// `local a = {} ; local b = a ; b.x = 1` must be visible through `a`. Copying
    /// the table on assignment is the single most confusing way to be wrong.
    #[test]
    fn tables_have_reference_semantics() {
        let a = Value::table();
        let b = a.clone();
        if let Value::Table(t) = &b {
            t.borrow_mut().set(Key::Str(Rc::from("x")), Value::Number(1.0));
        }
        if let Value::Table(t) = &a {
            assert_eq!(
                t.borrow().get(&Key::Str(Rc::from("x"))),
                Value::Number(1.0),
                "the write through b must be visible through a"
            );
        }
    }

    // --- keys ---

    /// `t[1]` from a script and `t[1.0]` from a loop counter must be one slot.
    /// Without normalization they would be different entries and a loop would
    /// never find what a literal wrote.
    #[test]
    fn integral_float_keys_normalize_to_integers() {
        assert_eq!(
            Key::from_value(&Value::Number(1.0)),
            Key::from_value(&Value::Number(1.0000000000))
        );
        assert_eq!(Key::from_value(&Value::Number(3.0)), Some(Key::Int(3)));
        assert_ne!(
            Key::from_value(&Value::Number(1.5)),
            Key::from_value(&Value::Number(1.0))
        );
    }

    /// A NaN key is unretrievable — `NaN ~= NaN` — so Lua rejects it, and so must
    /// this, or a script could write a value it can never read back.
    #[test]
    fn nil_and_nan_are_not_valid_keys() {
        assert_eq!(Key::from_value(&Value::Nil), None);
        assert_eq!(Key::from_value(&Value::Number(f64::NAN)), None);
    }

    #[test]
    fn keys_round_trip_back_to_values() {
        for value in [
            Value::Number(42.0),
            Value::Number(1.5),
            Value::Bool(true),
            Value::str("key"),
        ] {
            let key = Key::from_value(&value).expect("should be a valid key");
            assert_eq!(key.to_value(), value, "round trip failed for {value:?}");
        }
    }

    // --- table array/hash behaviour ---

    #[test]
    fn sequential_integer_keys_form_the_array_part() {
        let mut t = Table::new();
        for i in 1..=5 {
            t.set(Key::Int(i), Value::Number(i as f64 * 10.0));
        }
        assert_eq!(t.length(), 5);
        assert_eq!(t.array_part().len(), 5, "all five should be in the array part");
        assert_eq!(t.get(&Key::Int(3)), Value::Number(30.0));
    }

    /// A key that arrives out of order must migrate into the array part once the
    /// run reaches it, or `#t` would report 1 after `t[1]=a; t[3]=c; t[2]=b`.
    #[test]
    fn out_of_order_keys_migrate_into_the_array_run() {
        let mut t = Table::new();
        t.set(Key::Int(1), Value::str("a"));
        t.set(Key::Int(3), Value::str("c"));
        assert_eq!(t.length(), 1, "3 cannot join the run while 2 is missing");
        t.set(Key::Int(2), Value::str("b"));
        assert_eq!(
            t.length(),
            3,
            "filling the gap must pull 3 in from the hash part"
        );
        assert_eq!(t.get(&Key::Int(3)), Value::str("c"));
    }

    /// Assigning nil removes an entry — it does not store a nil. In Lua a nil
    /// value and an absent key are the same thing.
    #[test]
    fn assigning_nil_removes_the_entry() {
        let mut t = Table::new();
        t.set(Key::Str(Rc::from("x")), Value::Number(1.0));
        assert_eq!(t.len(), 1);
        t.set(Key::Str(Rc::from("x")), Value::Nil);
        assert_eq!(t.len(), 0);
        assert_eq!(t.get(&Key::Str(Rc::from("x"))), Value::Nil);
    }

    #[test]
    fn removing_the_last_array_element_shrinks_the_length() {
        let mut t = Table::new();
        for i in 1..=3 {
            t.set(Key::Int(i), Value::Number(i as f64));
        }
        t.set(Key::Int(3), Value::Nil);
        assert_eq!(t.length(), 2);
        t.set(Key::Int(2), Value::Nil);
        assert_eq!(t.length(), 1);
    }

    #[test]
    fn push_and_pop_work_on_the_array_part() {
        let mut t = Table::new();
        t.push(Value::str("a"));
        t.push(Value::str("b"));
        assert_eq!(t.length(), 2);
        assert_eq!(t.pop(), Value::str("b"));
        assert_eq!(t.length(), 1);
        assert_eq!(t.pop(), Value::str("a"));
        assert_eq!(t.pop(), Value::Nil, "popping empty gives nil, not a panic");
    }

    /// Appending nil would create a hole in the array part immediately, making `#`
    /// meaningless.
    #[test]
    fn pushing_nil_is_ignored() {
        let mut t = Table::new();
        t.push(Value::Nil);
        assert_eq!(t.length(), 0);
    }

    #[test]
    fn mixed_keys_are_all_reachable_and_counted() {
        let mut t = Table::new();
        t.set(Key::Int(1), Value::str("first"));
        t.set(Key::Str(Rc::from("name")), Value::str("thing"));
        t.set(Key::Bool(true), Value::Number(1.0));
        assert_eq!(t.length(), 1, "# counts only the array run");
        assert_eq!(t.len(), 3, "but all three entries exist");
        assert_eq!(t.get(&Key::Str(Rc::from("name"))), Value::str("thing"));
        assert_eq!(t.get(&Key::Bool(true)), Value::Number(1.0));
    }

    /// Keys have to be enumerable for `pairs`, and both parts must appear.
    #[test]
    fn keys_covers_both_the_array_and_hash_parts() {
        let mut t = Table::new();
        t.set(Key::Int(1), Value::str("a"));
        t.set(Key::Int(2), Value::str("b"));
        t.set(Key::Str(Rc::from("x")), Value::str("c"));
        let keys = t.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&Key::Int(1)));
        assert!(keys.contains(&Key::Int(2)));
        assert!(keys.contains(&Key::Str(Rc::from("x"))));
    }

    // --- __index ---

    /// `__index` is what makes prototype-style objects and method tables work,
    /// which is the whole reason a game script wants metatables.
    #[test]
    fn index_fallback_supplies_missing_keys() {
        let parent = Rc::new(RefCell::new(Table::new()));
        parent
            .borrow_mut()
            .set(Key::Str(Rc::from("inherited")), Value::Number(7.0));

        let mut child = Table::new();
        child.set_index_fallback(Value::Table(parent));
        assert_eq!(
            child.get(&Key::Str(Rc::from("inherited"))),
            Value::Number(7.0)
        );
    }

    /// The fallback must not shadow the table's own value.
    #[test]
    fn a_present_key_wins_over_the_index_fallback() {
        let parent = Rc::new(RefCell::new(Table::new()));
        parent
            .borrow_mut()
            .set(Key::Str(Rc::from("x")), Value::str("parent"));
        let mut child = Table::new();
        child.set_index_fallback(Value::Table(parent));
        child.set(Key::Str(Rc::from("x")), Value::str("child"));
        assert_eq!(child.get(&Key::Str(Rc::from("x"))), Value::str("child"));
    }

    #[test]
    fn get_own_ignores_the_fallback() {
        let parent = Rc::new(RefCell::new(Table::new()));
        parent
            .borrow_mut()
            .set(Key::Str(Rc::from("x")), Value::Number(1.0));
        let mut child = Table::new();
        child.set_index_fallback(Value::Table(parent));
        assert_eq!(child.get(&Key::Str(Rc::from("x"))), Value::Number(1.0));
        assert_eq!(
            child.get_own(&Key::Str(Rc::from("x"))),
            Value::Nil,
            "get_own must see only the table's own contents"
        );
    }

    #[test]
    fn setting_a_nil_fallback_clears_it() {
        let mut t = Table::new();
        t.set_index_fallback(Value::table());
        assert!(t.index_fallback().is_some());
        t.set_index_fallback(Value::Nil);
        assert!(t.index_fallback().is_none());
    }

    // --- coercion ---

    /// `"10" + 5` is 15 in Lua. The coercion applies to arithmetic only.
    #[test]
    fn numeric_strings_coerce_to_numbers() {
        assert_eq!(Value::str("10").as_number(), Some(10.0));
        assert_eq!(Value::str(" 2.5 ").as_number(), Some(2.5));
        assert_eq!(Value::str("abc").as_number(), None);
        assert_eq!(Value::Bool(true).as_number(), None);
        assert_eq!(Value::Nil.as_number(), None);
    }

    #[test]
    fn numbers_coerce_to_strings_for_concatenation() {
        assert_eq!(Value::Number(3.0).as_str_coerced().as_deref(), Some("3"));
        assert_eq!(Value::Number(1.5).as_str_coerced().as_deref(), Some("1.5"));
        assert_eq!(Value::str("x").as_str_coerced().as_deref(), Some("x"));
        assert_eq!(
            Value::Bool(true).as_str_coerced(),
            None,
            "a boolean does not concatenate in Lua"
        );
        assert_eq!(Value::Nil.as_str_coerced(), None);
    }

    // --- type names ---

    /// A script has no business knowing which side of the FFI boundary a function
    /// came from, so both report "function".
    #[test]
    fn both_kinds_of_function_report_the_same_type() {
        let native = Value::Native(Rc::new(NativeFunction {
            name: "test".into(),
            function: Box::new(|_| Ok(vec![])),
        }));
        assert_eq!(native.type_name(), "function");
        assert!(native.is_callable());
    }

    #[test]
    fn type_names_match_lua() {
        assert_eq!(Value::Nil.type_name(), "nil");
        assert_eq!(Value::Bool(true).type_name(), "boolean");
        assert_eq!(Value::Number(1.0).type_name(), "number");
        assert_eq!(Value::str("s").type_name(), "string");
        assert_eq!(Value::table().type_name(), "table");
    }

    #[test]
    fn tostring_gives_distinguishable_text_for_every_type() {
        assert_eq!(Value::Nil.to_display_string(), "nil");
        assert_eq!(Value::Bool(false).to_display_string(), "false");
        assert_eq!(Value::Number(2.0).to_display_string(), "2");
        assert_eq!(Value::str("hi").to_display_string(), "hi");
        // Two tables that exist *at the same time* must produce different text, so
        // a script can tell them apart in a log.
        //
        // Both must be held alive across the comparison. Formatting two temporaries
        // one after the other can legitimately give the same string: the first is
        // dropped before the second is allocated, and the allocator is free to hand
        // back the same address. That made this test fail about one run in twelve
        // before the values were bound.
        let first = Value::table();
        let second = Value::table();
        let a = first.to_display_string();
        let b = second.to_display_string();
        assert!(a.starts_with("table: "), "{a}");
        assert_ne!(a, b, "two live tables must be distinguishable");
        // And the same table always formats the same way, which is what makes the
        // text usable as an identity in a log.
        assert_eq!(a, first.to_display_string());
        assert_eq!(a, first.clone().to_display_string());
    }
}
