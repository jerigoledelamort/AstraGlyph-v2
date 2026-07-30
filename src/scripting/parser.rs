// Lua parser: tokens to an abstract syntax tree.
//
// Recursive descent, with a precedence-climbing expression parser. The AST is
// interpreted directly by `interp` rather than compiled to bytecode: a bytecode
// VM would be faster, but the scripts here run a handful of statements per entity
// per frame, and a tree-walking interpreter is a few hundred lines against a
// compiler plus a VM plus a disassembler for when it goes wrong.
//
// Recursion depth is bounded. A parser that recurses on nested expressions will
// blow the native stack on `((((((...))))))`, and a script is untrusted input —
// hot-reloaded from a file the user is editing, so a malformed one is expected
// rather than exceptional.

use crate::engine::core::{EngineError, Result};
use crate::scripting::lexer::{tokenize, Spanned, Token};

/// Deepest nesting the parser will accept.
///
/// Set from a measurement, not from taste. A stack overflow is not a catchable
/// error in Rust — the process aborts — so the limit has to be low enough that the
/// *largest* frame on the recursive path cannot exhaust the stack before the
/// counter trips.
///
/// The frames are very unequal: `parse_expr` is small, but `parse_statement` holds
/// a match over every statement form with locals for each, and is far larger.
/// Measured on a debug build (the worst case, since nothing is inlined), 300
/// nested parentheses errored cleanly at a limit of 128 while 128 nested
/// `do ... end` blocks — which recurse through `parse_statement` — overflowed
/// before reaching it. 48 clears both with margin, and is still far past anything
/// hand-written: an expression nested past ten levels is already unreadable.
const MAX_DEPTH: u32 = 48;

/// A binary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Concat,
    Eq,
    NotEq,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    And,
    Or,
}

/// A unary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    Len,
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Nil,
    True,
    False,
    Number(f64),
    Str(String),
    /// `...`
    Vararg,
    /// A variable reference.
    Name(String),
    /// `table[key]`. Field access `a.b` parses to this with a string key, so the
    /// interpreter has one indexing path rather than two.
    Index(Box<Expr>, Box<Expr>),
    /// `f(args)`
    Call(Box<Expr>, Vec<Expr>),
    /// `obj:method(args)` — kept distinct from `Call` because the receiver has to
    /// be evaluated once and passed as the first argument, which desugaring at
    /// parse time cannot express without duplicating the receiver expression.
    MethodCall(Box<Expr>, String, Vec<Expr>),
    /// `function(params) body end`
    Function(FunctionBody),
    /// `{ ... }`
    Table(Vec<TableField>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
}

/// One entry in a table constructor.
#[derive(Clone, Debug, PartialEq)]
pub enum TableField {
    /// `value` — appended to the array part.
    Positional(Expr),
    /// `key = value` or `[key] = value`.
    Keyed(Expr, Expr),
}

/// A function's parameters and body.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionBody {
    pub params: Vec<String>,
    /// Whether the last parameter is `...`.
    pub is_vararg: bool,
    pub body: Vec<Stat>,
    /// Line the function was declared on, for error messages.
    pub line: u32,
}

/// A statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Stat {
    /// `local a, b = x, y`
    Local(Vec<String>, Vec<Expr>),
    /// `a, b = x, y` — the targets are `Name` or `Index` expressions.
    Assign(Vec<Expr>, Vec<Expr>),
    /// An expression evaluated for its effect; only calls are legal here in Lua.
    ExprStat(Expr),
    /// `if c then ... elseif c then ... else ... end`
    If(Vec<(Expr, Vec<Stat>)>, Option<Vec<Stat>>),
    While(Expr, Vec<Stat>),
    /// `repeat ... until c` — the condition is evaluated *inside* the body's scope,
    /// which is why it is stored with the body rather than beside it.
    Repeat(Vec<Stat>, Expr),
    /// `for i = start, limit, step do ... end`
    NumericFor {
        var: String,
        start: Expr,
        limit: Expr,
        step: Option<Expr>,
        body: Vec<Stat>,
    },
    /// `for a, b in explist do ... end`
    GenericFor {
        vars: Vec<String>,
        exprs: Vec<Expr>,
        body: Vec<Stat>,
    },
    Do(Vec<Stat>),
    Return(Vec<Expr>),
    Break,
    /// `function a.b.c:d() ... end`, already resolved to a target and a body.
    FunctionDecl {
        target: Expr,
        body: FunctionBody,
        /// True for `:` declarations, which take an implicit `self`.
        is_method: bool,
    },
    /// `local function f() ... end` — distinct from `Local` because the name must
    /// be in scope inside the body, so a recursive local function can call itself.
    LocalFunction(String, FunctionBody),
}

struct Parser {
    tokens: Vec<Spanned>,
    position: usize,
    depth: u32,
}

fn err(line: u32, msg: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidState(format!("line {line}: {msg}"))
}

/// Parse Lua source into a list of statements.
pub fn parse(source: &str) -> Result<Vec<Stat>> {
    let tokens = tokenize(source)?;
    let mut parser = Parser {
        tokens,
        position: 0,
        depth: 0,
    };
    let block = parser.parse_block()?;
    parser.expect(Token::Eof)?;
    Ok(block)
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.position.min(self.tokens.len() - 1)].token
    }

    fn line(&self) -> u32 {
        self.tokens[self.position.min(self.tokens.len() - 1)].line
    }

    fn advance(&mut self) -> Token {
        let token = self.tokens[self.position.min(self.tokens.len() - 1)]
            .token
            .clone();
        if self.position < self.tokens.len() - 1 {
            self.position += 1;
        }
        token
    }

    fn check(&self, token: &Token) -> bool {
        self.peek() == token
    }

    /// Consume `token` if present.
    fn accept(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token: Token) -> Result<()> {
        if self.check(&token) {
            self.advance();
            Ok(())
        } else {
            Err(err(
                self.line(),
                format!(
                    "expected {}, found {}",
                    token.describe(),
                    self.peek().describe()
                ),
            ))
        }
    }

    fn expect_name(&mut self) -> Result<String> {
        match self.peek().clone() {
            Token::Name(name) => {
                self.advance();
                Ok(name)
            }
            other => Err(err(
                self.line(),
                format!("expected a name, found {}", other.describe()),
            )),
        }
    }

    /// Guard against unbounded recursion, which would overflow the native stack.
    fn enter(&mut self) -> Result<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(err(
                self.line(),
                format!("nesting deeper than {MAX_DEPTH} levels"),
            ));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// A sequence of statements, up to a block terminator.
    fn parse_block(&mut self) -> Result<Vec<Stat>> {
        self.enter()?;
        let mut stats = Vec::new();
        loop {
            match self.peek() {
                Token::Eof
                | Token::End
                | Token::Else
                | Token::Elseif
                | Token::Until => break,
                Token::Semicolon => {
                    self.advance();
                }
                // `return` and `break` end a block: nothing may follow them, and
                // parsing on would accept code Lua rejects.
                Token::Return => {
                    self.advance();
                    let exprs = if self.block_ends() {
                        Vec::new()
                    } else {
                        self.parse_exprlist()?
                    };
                    self.accept(&Token::Semicolon);
                    stats.push(Stat::Return(exprs));
                    break;
                }
                Token::Break => {
                    self.advance();
                    self.accept(&Token::Semicolon);
                    stats.push(Stat::Break);
                    break;
                }
                _ => stats.push(self.parse_statement()?),
            }
        }
        self.leave();
        Ok(stats)
    }

    fn block_ends(&self) -> bool {
        matches!(
            self.peek(),
            Token::Eof | Token::End | Token::Else | Token::Elseif | Token::Until | Token::Semicolon
        )
    }

    fn parse_statement(&mut self) -> Result<Stat> {
        // Counted as well as `parse_block`, because a nested `do ... end` recurses
        // through here and this is the larger frame of the two.
        self.enter()?;
        let stat = self.parse_statement_inner();
        self.leave();
        stat
    }

    fn parse_statement_inner(&mut self) -> Result<Stat> {
        match self.peek().clone() {
            Token::Local => {
                self.advance();
                if self.accept(&Token::Function) {
                    let name = self.expect_name()?;
                    let body = self.parse_function_body(false)?;
                    return Ok(Stat::LocalFunction(name, body));
                }
                let mut names = vec![self.expect_name()?];
                while self.accept(&Token::Comma) {
                    names.push(self.expect_name()?);
                }
                let exprs = if self.accept(&Token::Assign) {
                    self.parse_exprlist()?
                } else {
                    Vec::new()
                };
                Ok(Stat::Local(names, exprs))
            }
            Token::If => {
                self.advance();
                let mut branches = Vec::new();
                let condition = self.parse_expr()?;
                self.expect(Token::Then)?;
                branches.push((condition, self.parse_block()?));
                let mut else_block = None;
                loop {
                    if self.accept(&Token::Elseif) {
                        let condition = self.parse_expr()?;
                        self.expect(Token::Then)?;
                        branches.push((condition, self.parse_block()?));
                    } else if self.accept(&Token::Else) {
                        else_block = Some(self.parse_block()?);
                        break;
                    } else {
                        break;
                    }
                }
                self.expect(Token::End)?;
                Ok(Stat::If(branches, else_block))
            }
            Token::While => {
                self.advance();
                let condition = self.parse_expr()?;
                self.expect(Token::Do)?;
                let body = self.parse_block()?;
                self.expect(Token::End)?;
                Ok(Stat::While(condition, body))
            }
            Token::Repeat => {
                self.advance();
                let body = self.parse_block()?;
                self.expect(Token::Until)?;
                let condition = self.parse_expr()?;
                Ok(Stat::Repeat(body, condition))
            }
            Token::For => {
                self.advance();
                let first = self.expect_name()?;
                if self.accept(&Token::Assign) {
                    let start = self.parse_expr()?;
                    self.expect(Token::Comma)?;
                    let limit = self.parse_expr()?;
                    let step = if self.accept(&Token::Comma) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    self.expect(Token::Do)?;
                    let body = self.parse_block()?;
                    self.expect(Token::End)?;
                    Ok(Stat::NumericFor {
                        var: first,
                        start,
                        limit,
                        step,
                        body,
                    })
                } else {
                    let mut vars = vec![first];
                    while self.accept(&Token::Comma) {
                        vars.push(self.expect_name()?);
                    }
                    self.expect(Token::In)?;
                    let exprs = self.parse_exprlist()?;
                    self.expect(Token::Do)?;
                    let body = self.parse_block()?;
                    self.expect(Token::End)?;
                    Ok(Stat::GenericFor { vars, exprs, body })
                }
            }
            Token::Do => {
                self.advance();
                let body = self.parse_block()?;
                self.expect(Token::End)?;
                Ok(Stat::Do(body))
            }
            Token::Function => {
                self.advance();
                // `function a.b.c:d()` — the target is a chain of field accesses.
                let mut target = Expr::Name(self.expect_name()?);
                let mut is_method = false;
                loop {
                    if self.accept(&Token::Dot) {
                        let field = self.expect_name()?;
                        target = Expr::Index(Box::new(target), Box::new(Expr::Str(field)));
                    } else if self.accept(&Token::Colon) {
                        let field = self.expect_name()?;
                        target = Expr::Index(Box::new(target), Box::new(Expr::Str(field)));
                        is_method = true;
                        break;
                    } else {
                        break;
                    }
                }
                let body = self.parse_function_body(is_method)?;
                Ok(Stat::FunctionDecl {
                    target,
                    body,
                    is_method,
                })
            }
            _ => {
                // Either an assignment or a call. Both start with a prefix
                // expression, so parse that and then decide.
                let first = self.parse_suffixed()?;
                if self.check(&Token::Assign) || self.check(&Token::Comma) {
                    let mut targets = vec![first];
                    while self.accept(&Token::Comma) {
                        targets.push(self.parse_suffixed()?);
                    }
                    self.expect(Token::Assign)?;
                    let values = self.parse_exprlist()?;
                    for target in &targets {
                        if !matches!(target, Expr::Name(_) | Expr::Index(_, _)) {
                            return Err(err(
                                self.line(),
                                "cannot assign to this expression",
                            ));
                        }
                    }
                    Ok(Stat::Assign(targets, values))
                } else {
                    // Only a call is a legal statement on its own; anything else
                    // is a typo (`x + 1` alone does nothing) and Lua rejects it.
                    if !matches!(first, Expr::Call(_, _) | Expr::MethodCall(_, _, _)) {
                        return Err(err(
                            self.line(),
                            "this expression is not a statement (only calls are)",
                        ));
                    }
                    Ok(Stat::ExprStat(first))
                }
            }
        }
    }

    fn parse_function_body(&mut self, is_method: bool) -> Result<FunctionBody> {
        let line = self.line();
        self.expect(Token::LParen)?;
        // A method's implicit `self` is materialised here as a real first
        // parameter, so the interpreter's calling convention has no special case.
        let mut params: Vec<String> = if is_method {
            vec!["self".to_string()]
        } else {
            Vec::new()
        };
        let mut is_vararg = false;
        if !self.check(&Token::RParen) {
            loop {
                if self.accept(&Token::Ellipsis) {
                    is_vararg = true;
                    break;
                }
                params.push(self.expect_name()?);
                if !self.accept(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect(Token::RParen)?;
        let body = self.parse_block()?;
        self.expect(Token::End)?;
        Ok(FunctionBody {
            params,
            is_vararg,
            body,
            line,
        })
    }

    fn parse_exprlist(&mut self) -> Result<Vec<Expr>> {
        let mut exprs = vec![self.parse_expr()?];
        while self.accept(&Token::Comma) {
            exprs.push(self.parse_expr()?);
        }
        Ok(exprs)
    }

    fn parse_expr(&mut self) -> Result<Expr> {
        self.enter()?;
        let expr = self.parse_binary(0)?;
        self.leave();
        Ok(expr)
    }

    /// Precedence-climbing binary expression parser.
    fn parse_binary(&mut self, min_precedence: u8) -> Result<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            let Some((op, left_precedence, right_precedence)) = binary_op(self.peek()) else {
                break;
            };
            if left_precedence < min_precedence {
                break;
            }
            self.advance();
            // Right-associative operators (`^`, `..`) recurse at the same
            // precedence rather than one above it, which is what makes
            // `2^3^2` parse as `2^(3^2)`.
            let right = self.parse_binary(right_precedence)?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        let op = match self.peek() {
            Token::Minus => Some(UnOp::Neg),
            Token::Not => Some(UnOp::Not),
            Token::Hash => Some(UnOp::Len),
            _ => None,
        };
        if let Some(op) = op {
            self.advance();
            self.enter()?;
            // Unary operators bind tighter than every binary one except `^`,
            // so `-x^2` is `-(x^2)`.
            let operand = self.parse_binary(UNARY_PRECEDENCE)?;
            self.leave();
            return Ok(Expr::Unary(op, Box::new(operand)));
        }
        self.parse_suffixed()
    }

    /// A primary expression followed by any number of suffixes: indexing, field
    /// access, calls, method calls.
    fn parse_suffixed(&mut self) -> Result<Expr> {
        self.enter()?;
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek().clone() {
                Token::Dot => {
                    self.advance();
                    let field = self.expect_name()?;
                    expr = Expr::Index(Box::new(expr), Box::new(Expr::Str(field)));
                }
                Token::LBracket => {
                    self.advance();
                    let key = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    expr = Expr::Index(Box::new(expr), Box::new(key));
                }
                Token::LParen => {
                    self.advance();
                    let args = if self.check(&Token::RParen) {
                        Vec::new()
                    } else {
                        self.parse_exprlist()?
                    };
                    self.expect(Token::RParen)?;
                    expr = Expr::Call(Box::new(expr), args);
                }
                // `f"string"` and `f{table}` — Lua's sugar for a single-argument
                // call. Common enough in real scripts to be worth supporting.
                Token::Str(text) => {
                    self.advance();
                    expr = Expr::Call(Box::new(expr), vec![Expr::Str(text)]);
                }
                Token::LBrace => {
                    let table = self.parse_table()?;
                    expr = Expr::Call(Box::new(expr), vec![table]);
                }
                Token::Colon => {
                    self.advance();
                    let method = self.expect_name()?;
                    let args = match self.peek().clone() {
                        Token::LParen => {
                            self.advance();
                            let args = if self.check(&Token::RParen) {
                                Vec::new()
                            } else {
                                self.parse_exprlist()?
                            };
                            self.expect(Token::RParen)?;
                            args
                        }
                        Token::Str(text) => {
                            self.advance();
                            vec![Expr::Str(text)]
                        }
                        Token::LBrace => vec![self.parse_table()?],
                        other => {
                            return Err(err(
                                self.line(),
                                format!("expected method arguments, found {}", other.describe()),
                            ))
                        }
                    };
                    expr = Expr::MethodCall(Box::new(expr), method, args);
                }
                _ => break,
            }
        }
        self.leave();
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Token::Nil => {
                self.advance();
                Ok(Expr::Nil)
            }
            Token::True => {
                self.advance();
                Ok(Expr::True)
            }
            Token::False => {
                self.advance();
                Ok(Expr::False)
            }
            Token::Number(n) => {
                self.advance();
                Ok(Expr::Number(n))
            }
            Token::Str(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Token::Ellipsis => {
                self.advance();
                Ok(Expr::Vararg)
            }
            Token::Name(name) => {
                self.advance();
                Ok(Expr::Name(name))
            }
            Token::Function => {
                self.advance();
                Ok(Expr::Function(self.parse_function_body(false)?))
            }
            Token::LBrace => self.parse_table(),
            Token::LParen => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(inner)
            }
            other => Err(err(
                self.line(),
                format!("unexpected {} in an expression", other.describe()),
            )),
        }
    }

    fn parse_table(&mut self) -> Result<Expr> {
        self.enter()?;
        self.expect(Token::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(&Token::RBrace) {
            match self.peek().clone() {
                Token::LBracket => {
                    self.advance();
                    let key = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    self.expect(Token::Assign)?;
                    fields.push(TableField::Keyed(key, self.parse_expr()?));
                }
                // `name = value`, but only when the `=` really follows: `{ x }` is
                // a positional field holding the variable `x`, and looking one
                // token further is the only way to tell them apart.
                Token::Name(name)
                    if matches!(
                        self.tokens.get(self.position + 1).map(|s| &s.token),
                        Some(Token::Assign)
                    ) =>
                {
                    self.advance();
                    self.advance();
                    fields.push(TableField::Keyed(Expr::Str(name), self.parse_expr()?));
                }
                _ => fields.push(TableField::Positional(self.parse_expr()?)),
            }
            // Both `,` and `;` separate fields, and a trailing one is allowed.
            if !self.accept(&Token::Comma) && !self.accept(&Token::Semicolon) {
                break;
            }
        }
        self.expect(Token::RBrace)?;
        self.leave();
        Ok(Expr::Table(fields))
    }
}

/// Precedence at which a unary operator binds its operand.
///
/// Above every binary operator except `^`, so `-x^2` is `-(x^2)` and `-x*y` is
/// `(-x)*y`.
const UNARY_PRECEDENCE: u8 = 8;

/// A binary operator's left and right precedences.
///
/// Two numbers rather than one plus an associativity flag: a right-associative
/// operator is exactly one whose right precedence equals its left, and encoding
/// it that way removes the branch from the parse loop.
fn binary_op(token: &Token) -> Option<(BinOp, u8, u8)> {
    Some(match token {
        Token::Or => (BinOp::Or, 1, 2),
        Token::And => (BinOp::And, 2, 3),
        Token::Less => (BinOp::Less, 3, 4),
        Token::Greater => (BinOp::Greater, 3, 4),
        Token::LessEqual => (BinOp::LessEq, 3, 4),
        Token::GreaterEqual => (BinOp::GreaterEq, 3, 4),
        Token::NotEqual => (BinOp::NotEq, 3, 4),
        Token::Equal => (BinOp::Eq, 3, 4),
        // Right-associative: `a .. b .. c` is `a .. (b .. c)`, which for strings
        // is the same answer but for a metamethod would not be.
        Token::Concat => (BinOp::Concat, 5, 5),
        Token::Plus => (BinOp::Add, 6, 7),
        Token::Minus => (BinOp::Sub, 6, 7),
        Token::Star => (BinOp::Mul, 7, 8),
        Token::Slash => (BinOp::Div, 7, 8),
        Token::Percent => (BinOp::Mod, 7, 8),
        // Right-associative and above unary: `2^3^2` is `2^(3^2)` = 512.
        Token::Caret => (BinOp::Pow, 10, 9),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(source: &str) -> Vec<Stat> {
        parse(source).unwrap_or_else(|e| panic!("failed to parse {source:?}: {e}"))
    }

    #[test]
    fn parses_a_local_assignment() {
        assert_eq!(
            parse_ok("local x = 1"),
            vec![Stat::Local(vec!["x".into()], vec![Expr::Number(1.0)])]
        );
    }

    #[test]
    fn parses_multiple_assignment() {
        assert_eq!(
            parse_ok("local a, b = 1, 2"),
            vec![Stat::Local(
                vec!["a".into(), "b".into()],
                vec![Expr::Number(1.0), Expr::Number(2.0)]
            )]
        );
    }

    #[test]
    fn a_local_without_a_value_is_allowed() {
        assert_eq!(
            parse_ok("local x"),
            vec![Stat::Local(vec!["x".into()], vec![])]
        );
    }

    /// Precedence: `1 + 2 * 3` must be `1 + (2 * 3)`. A flat left-to-right parse
    /// would compute 9 instead of 7, silently.
    #[test]
    fn multiplication_binds_tighter_than_addition() {
        let stats = parse_ok("local x = 1 + 2 * 3");
        let Stat::Local(_, exprs) = &stats[0] else {
            panic!("expected a local");
        };
        assert_eq!(
            exprs[0],
            Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Number(1.0)),
                Box::new(Expr::Binary(
                    BinOp::Mul,
                    Box::new(Expr::Number(2.0)),
                    Box::new(Expr::Number(3.0))
                ))
            )
        );
    }

    /// `^` is right-associative, so `2^3^2` is 2^9 = 512, not 8^2 = 64.
    #[test]
    fn exponentiation_is_right_associative() {
        let stats = parse_ok("local x = 2 ^ 3 ^ 2");
        let Stat::Local(_, exprs) = &stats[0] else {
            panic!("expected a local");
        };
        assert_eq!(
            exprs[0],
            Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Number(2.0)),
                Box::new(Expr::Binary(
                    BinOp::Pow,
                    Box::new(Expr::Number(3.0)),
                    Box::new(Expr::Number(2.0))
                ))
            )
        );
    }

    /// `..` is right-associative too.
    #[test]
    fn concatenation_is_right_associative() {
        let stats = parse_ok(r#"local x = "a" .. "b" .. "c""#);
        let Stat::Local(_, exprs) = &stats[0] else {
            panic!("expected a local");
        };
        match &exprs[0] {
            Expr::Binary(BinOp::Concat, left, right) => {
                assert_eq!(**left, Expr::Str("a".into()));
                assert!(
                    matches!(**right, Expr::Binary(BinOp::Concat, _, _)),
                    "the right side should be the nested concat"
                );
            }
            other => panic!("expected a concat, got {other:?}"),
        }
    }

    /// Unary minus binds tighter than `*` but looser than `^`: `-x^2` is `-(x^2)`.
    #[test]
    fn unary_minus_binds_looser_than_exponentiation() {
        let stats = parse_ok("local x = -y ^ 2");
        let Stat::Local(_, exprs) = &stats[0] else {
            panic!("expected a local");
        };
        assert!(
            matches!(&exprs[0], Expr::Unary(UnOp::Neg, inner)
                if matches!(**inner, Expr::Binary(BinOp::Pow, _, _))),
            "got {:?}",
            exprs[0]
        );
    }

    #[test]
    fn field_access_and_indexing_produce_the_same_node() {
        let dot = parse_ok("local x = a.b");
        let bracket = parse_ok(r#"local x = a["b"]"#);
        let Stat::Local(_, dot_exprs) = &dot[0] else { panic!() };
        let Stat::Local(_, bracket_exprs) = &bracket[0] else { panic!() };
        assert_eq!(
            dot_exprs[0], bracket_exprs[0],
            "a.b and a[\"b\"] must parse identically"
        );
    }

    #[test]
    fn parses_calls_including_the_sugar_forms() {
        assert!(matches!(
            &parse_ok("f()")[0],
            Stat::ExprStat(Expr::Call(_, args)) if args.is_empty()
        ));
        assert!(matches!(
            &parse_ok("f(1, 2)")[0],
            Stat::ExprStat(Expr::Call(_, args)) if args.len() == 2
        ));
        assert!(
            matches!(
                &parse_ok(r#"print"hello""#)[0],
                Stat::ExprStat(Expr::Call(_, args)) if args.len() == 1
            ),
            "f\"str\" is Lua's single-argument sugar"
        );
        assert!(
            matches!(
                &parse_ok("f{1, 2}")[0],
                Stat::ExprStat(Expr::Call(_, args)) if args.len() == 1
            ),
            "f{{table}} is the same sugar"
        );
    }

    #[test]
    fn parses_a_method_call() {
        assert!(matches!(
            &parse_ok("obj:method(1)")[0],
            Stat::ExprStat(Expr::MethodCall(_, name, args))
                if name == "method" && args.len() == 1
        ));
    }

    /// A method declaration must materialise `self` as a real parameter, or the
    /// interpreter would need a special calling convention for methods.
    #[test]
    fn a_method_declaration_gains_an_implicit_self() {
        let stats = parse_ok("function obj:greet(name) return name end");
        match &stats[0] {
            Stat::FunctionDecl { body, is_method, .. } => {
                assert!(is_method);
                assert_eq!(body.params, vec!["self".to_string(), "name".to_string()]);
            }
            other => panic!("expected a function declaration, got {other:?}"),
        }
    }

    #[test]
    fn parses_control_flow() {
        assert!(matches!(&parse_ok("if a then b() end")[0], Stat::If(_, None)));
        assert!(matches!(
            &parse_ok("if a then b() else c() end")[0],
            Stat::If(_, Some(_))
        ));
        let stats = parse_ok("if a then x() elseif b then y() else z() end");
        match &stats[0] {
            Stat::If(branches, else_block) => {
                assert_eq!(branches.len(), 2, "if and elseif are both branches");
                assert!(else_block.is_some());
            }
            other => panic!("got {other:?}"),
        }
        assert!(matches!(&parse_ok("while a do b() end")[0], Stat::While(_, _)));
        assert!(matches!(
            &parse_ok("repeat a() until b")[0],
            Stat::Repeat(_, _)
        ));
        assert!(matches!(&parse_ok("do a() end")[0], Stat::Do(_)));
    }

    #[test]
    fn parses_both_kinds_of_for_loop() {
        match &parse_ok("for i = 1, 10 do f(i) end")[0] {
            Stat::NumericFor { var, step, .. } => {
                assert_eq!(var, "i");
                assert!(step.is_none(), "an absent step must stay absent, not default to 1 here");
            }
            other => panic!("got {other:?}"),
        }
        match &parse_ok("for i = 10, 1, -1 do f(i) end")[0] {
            Stat::NumericFor { step, .. } => assert!(step.is_some()),
            other => panic!("got {other:?}"),
        }
        match &parse_ok("for k, v in pairs(t) do f(k, v) end")[0] {
            Stat::GenericFor { vars, exprs, .. } => {
                assert_eq!(vars.len(), 2);
                assert_eq!(exprs.len(), 1);
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn parses_table_constructors() {
        let stats = parse_ok(r#"local t = { 1, 2, x = 3, ["y"] = 4, }"#);
        let Stat::Local(_, exprs) = &stats[0] else { panic!() };
        match &exprs[0] {
            Expr::Table(fields) => {
                assert_eq!(fields.len(), 4, "a trailing comma must not add a field");
                assert!(matches!(fields[0], TableField::Positional(_)));
                assert!(matches!(fields[2], TableField::Keyed(_, _)));
            }
            other => panic!("got {other:?}"),
        }
    }

    /// `{ x }` holds the *value of* x, while `{ x = 1 }` holds a key. Telling them
    /// apart needs a look at the token after the name.
    #[test]
    fn a_bare_name_in_a_table_is_positional_not_a_key() {
        let stats = parse_ok("local t = { x }");
        let Stat::Local(_, exprs) = &stats[0] else { panic!() };
        match &exprs[0] {
            Expr::Table(fields) => {
                assert!(
                    matches!(&fields[0], TableField::Positional(Expr::Name(n)) if n == "x"),
                    "got {:?}",
                    fields[0]
                );
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn parses_function_declarations_and_expressions() {
        assert!(matches!(
            &parse_ok("function f() end")[0],
            Stat::FunctionDecl { is_method: false, .. }
        ));
        assert!(matches!(
            &parse_ok("local function f() end")[0],
            Stat::LocalFunction(_, _)
        ));
        let stats = parse_ok("local f = function(a, b) return a end");
        let Stat::Local(_, exprs) = &stats[0] else { panic!() };
        assert!(matches!(&exprs[0], Expr::Function(_)));
    }

    #[test]
    fn a_nested_function_name_becomes_an_index_chain() {
        match &parse_ok("function a.b.c() end")[0] {
            Stat::FunctionDecl { target, .. } => {
                // a.b.c is Index(Index(Name(a), "b"), "c").
                assert!(
                    matches!(target, Expr::Index(inner, _)
                        if matches!(**inner, Expr::Index(_, _))),
                    "got {target:?}"
                );
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn parses_varargs() {
        let stats = parse_ok("local function f(a, ...) return ... end");
        match &stats[0] {
            Stat::LocalFunction(_, body) => {
                assert!(body.is_vararg);
                assert_eq!(body.params, vec!["a".to_string()]);
                assert_eq!(body.body, vec![Stat::Return(vec![Expr::Vararg])]);
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn return_and_break_end_a_block() {
        assert_eq!(parse_ok("return"), vec![Stat::Return(vec![])]);
        assert_eq!(
            parse_ok("return 1, 2"),
            vec![Stat::Return(vec![Expr::Number(1.0), Expr::Number(2.0)])]
        );
        assert_eq!(
            parse_ok("while true do break end"),
            vec![Stat::While(Expr::True, vec![Stat::Break])]
        );
    }

    /// Only a call may stand alone as a statement. `x + 1` on its own line does
    /// nothing and is almost always a typo, so Lua rejects it and so does this.
    #[test]
    fn a_bare_expression_is_not_a_statement() {
        assert!(parse("x + 1").is_err());
        assert!(parse("42").is_err());
        // But a call is fine.
        assert!(parse("f()").is_ok());
    }

    #[test]
    fn assignment_targets_must_be_assignable() {
        assert!(parse("a = 1").is_ok());
        assert!(parse("a.b = 1").is_ok());
        assert!(parse("a[1] = 1").is_ok());
        assert!(parse("f() = 1").is_err(), "cannot assign to a call");
    }

    /// A malformed script is expected input — the user is editing it live — so it
    /// must produce an error naming a line, not a panic and not a stack overflow.
    #[test]
    fn syntax_errors_report_a_line_and_never_panic() {
        for bad in [
            "local",
            "local x =",
            "if then end",
            "if x then",
            "while do end",
            "for i do end",
            "function",
            "f(",
            "local t = {",
            "end",
            "a = = 1",
            ")",
        ] {
            let e = parse(bad).unwrap_err();
            assert!(
                e.to_string().contains("line"),
                "error for {bad:?} should name a line: {e}"
            );
        }
    }

    /// Unbounded recursion on nested expressions overflows the native stack, which
    /// is an abort rather than a catchable error. A script is untrusted input.
    #[test]
    fn deeply_nested_input_errors_instead_of_overflowing_the_stack() {
        // Expressions: the small-frame path.
        let deep = format!("local x = {}1{}", "(".repeat(5000), ")".repeat(5000));
        let e = parse(&deep).unwrap_err();
        assert!(e.to_string().contains("nesting"), "{e}");

        // Statements: the large-frame path, and the one that actually overflowed
        // at a limit of 128. `do ... end` recurses through parse_statement, whose
        // frame holds a match over every statement form.
        let blocks = format!("{}f(){}", "do ".repeat(5000), " end".repeat(5000));
        let e = parse(&blocks).unwrap_err();
        assert!(e.to_string().contains("nesting"), "{e}");

        // Nested tables and calls reach the guard through yet another path.
        let tables = format!("local t = {}1{}", "{".repeat(5000), "}".repeat(5000));
        assert!(parse(&tables).is_err());
        let calls = format!("local x = f{}1{}", "(".repeat(2000), ")".repeat(2000));
        assert!(parse(&calls).is_err());
    }

    /// Moderate nesting must still work, or the guard is set too tight to be
    /// usable.
    /// The limit has to leave room for real code. A guard so tight that ordinary
    /// scripts trip it is worse than no guard, because it fails on valid input.
    #[test]
    fn the_depth_limit_is_far_above_anything_hand_written() {
        assert!(MAX_DEPTH >= 32, "a limit of {MAX_DEPTH} would reject real code");
        // Deliberately close to the limit but under it, in the large-frame path.
        let n = (MAX_DEPTH as usize) / 3;
        let blocks = format!("{}f(){}", "do ".repeat(n), " end".repeat(n));
        assert!(
            parse(&blocks).is_ok(),
            "{n} nested blocks should be accepted at a limit of {MAX_DEPTH}"
        );
    }

    #[test]
    fn ordinary_nesting_is_accepted() {
        let nested = format!("local x = {}1{}", "(".repeat(20), ")".repeat(20));
        assert!(parse(&nested).is_ok());
        let source = "\
            local function outer()\n\
              if a then\n\
                for i = 1, 10 do\n\
                  while b do\n\
                    local t = { x = { y = { z = 1 } } }\n\
                  end\n\
                end\n\
              end\n\
            end";
        assert!(parse(source).is_ok());
    }

    #[test]
    fn semicolons_are_optional_separators() {
        assert_eq!(parse_ok("local a = 1; local b = 2").len(), 2);
        assert_eq!(parse_ok(";;; local a = 1 ;;;").len(), 1);
    }

    #[test]
    fn an_empty_program_parses_to_nothing() {
        assert!(parse_ok("").is_empty());
        assert!(parse_ok("-- just a comment").is_empty());
    }
}
