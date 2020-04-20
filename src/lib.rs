use anyhow::anyhow;
use std::dbg;
use std::fmt;
use std::iter::Iterator;

/// Interpret line and return the top item of the stack after execution.
pub fn interpret(line: &str) -> anyhow::Result<Option<isize>> {
    let (program, args) = parse_input_string(line)?;
    let mut stack = program.exec(args)?;

    match stack.pop() {
        Some(token) => match token {
            Token::Num(val) => Ok(Some(val)),
            t => Err(Error::ReturnWrongToken(t.to_string()).into()),
        },
        None => Ok(None),
    }
}

/// Parses an input string into a syntactically correct `Program` and a list of input arguments.
fn parse_input_string(s: &str) -> anyhow::Result<(Program, Vec<isize>)> {
    let (program, args) = split(s);

    dbg!(program);
    dbg!(args);

    let program = Program::new(program)?;
    let args = parse_args(args)?;

    Ok((program, args))
}

// FIXME: The following two functions are only public so that we can include tests in the docs.

/// Splits an input string into the two strings, program and args.
///
/// # Examples
///
/// ```
/// let s = "(postfix 1 2 add)[3]";
/// let (prog, args) = postfix::split(s);
///
/// assert_eq!(prog, "(postfix 1 2 add)");
/// assert_eq!(args, "[3]");
/// ```
pub fn split(s: &str) -> (&str, &str) {
    if let Some(index) = s.find('[') {
        return (&s[..index], &s[index..]);
    }
    (s, "")
}

/// Parse an args list into a list of integers
///
/// # Examples
/// ```
/// let args = "[1, 2, 3]";
/// let want = vec![3, 2, 1];
/// let got = postfix::parse_args(args).unwrap();
/// assert_eq!(got, want);
/// ```
pub fn parse_args(s: &str) -> anyhow::Result<Vec<isize>> {
    let mut v = vec![];

    let s = s.trim();

    if !s.starts_with('[') {
        return Err(Error::ArgsParseMissingOpeningBracket.into());
    }
    let s = &s[1..];

    if !s.ends_with(']') {
        return Err(Error::ArgsParseMissingClosingBracket.into());
    }
    let s = &s[..s.len() - 1];

    for arg in s.split_ascii_whitespace() {
        let arg = if arg.ends_with(',') {
            &arg[..arg.len() - 1]
        } else {
            &arg
        };

        let val = arg.parse::<isize>()?;
        v.push(val);
    }
    // We push the args onto the stack in reverse order.
    v.reverse();

    Ok(v)
}

/// Program holds the parsed Postfix program, guaranteed to be syntactically correct.
struct Program {
    /// The number of arguments the program expects.
    n_args: usize,

    /// The tokens that make up this program, these can be integers or commands.
    tokens: Vec<Token>,
}

impl Program {
    /// Parses `s` and creates a `Program` object.
    /// Returns an `Err` if the line is not syntactically correct.
    fn new(s: &str) -> anyhow::Result<Program> {
        validate_program_string(s)?;
        let s = s.trim();

        if !s.starts_with('(') {
            return Err(Error::ProgramParseMissingOpeningParens.into());
        }
        let s = &s[1..];

        if !s.ends_with(')') {
            return Err(Error::ProgramParseMissingClosingParens.into());
        }
        let s = &s[..s.len() - 1];
        let s = s.trim();

        let mut iter = PostfixIterator::new(s);

        match iter.next() {
            Some(token) if token == "postfix" => {}
            _ => return Err(Error::ProgramParseMissingPostfix.into()),
        }

        let n_args = iter
            .next()
            .ok_or_else(|| Error::ProgramParseMissingNumArgs)?
            .parse::<usize>()?;

        let mut tokens = vec![];

        for t in iter {
            let token = Token::new(&t)?;
            tokens.push(token);
        }

        Ok(Program { n_args, tokens })
    }

    /// Execute the program and return the top item from the stack.
    fn exec(&self, args: Vec<isize>) -> anyhow::Result<Vec<Token>> {
        if self.n_args != args.len() {
            return Err(anyhow!(
                "wrong number of arguments, expected {}: {:?}",
                self.n_args,
                args,
            ));
        }

        let mut stack = Vec::new();

        for arg in args {
            stack.push(Token::Num(arg));
        }

        for token in self.tokens.iter() {
            handle_token(&mut stack, *token)?;
        }

        Ok(stack)
    }
}

fn validate_program_string(s: &str) -> anyhow::Result<()> {
    validate_parenthesis(s)?;
    // TODO: Do more validation?
    Ok(())
}

fn validate_parenthesis(s: &str) -> anyhow::Result<()> {
    let mut open_parens = 0;
    let mut close_parens = 0;

    for c in s.chars() {
        if c == '(' {
            open_parens += 1;
            continue;
        }
        if c == ')' {
            close_parens += 1;
            continue;
        }
    }

    if open_parens < close_parens {
        return Err(Error::ProgramParseMissingOpeningParens)?;
    }

    if close_parens < open_parens {
        return Err(Error::ProgramParseMissingClosingParens)?;
    }

    Ok(())
}

/// An iterator that iterates over an input string representing a postfix program.
struct PostfixIterator {
    s: String,
}

impl PostfixIterator {
    fn new(s: &str) -> Self {
        PostfixIterator { s: s.to_string() }
    }
}

impl Iterator for PostfixIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.s.len() == 0 {
            return None;
        }

        if self.s.starts_with("(") {
            // TODO: Handle executable sequence.
        }

        if is_last_token(&self.s) {
            let last = self.s.clone();
            self.s = String::from("");
            return Some(last);
        }

        let index = self.s.find(" ").unwrap(); // If its not the last token there must be white space.
        let tmp = self.s.split_off(index);
        let token = self.s.clone();
        self.s = tmp;

        self.s = self.s.trim_start().to_string();

        Some(token)
    }
}

fn is_last_token(s: &str) -> bool {
    return !s.contains(" ");
}

/// Handle a single token during execution of a program.
fn handle_token(stack: &mut Vec<Token>, token: Token) -> anyhow::Result<()> {
    match token {
        // By definition, integers are just pushed onto the stack.
        Token::Num(val) => {
            stack.push(Token::Num(val));
            Ok(())
        }
        // Commands are executed with the current stack.
        Token::Cmd(cmd) => match cmd {
            Cmd::Pop => pop(stack),
            Cmd::Add => add(stack),
            Cmd::Sub => sub(stack),
            Cmd::Mul => mul(stack),
            Cmd::Div => div(stack),
            Cmd::Rem => rem(stack),
            Cmd::Lt => lt(stack),
            Cmd::Gt => gt(stack),
            Cmd::Eq => eq(stack),
            Cmd::Swap => swap(stack),
            Cmd::Sel => sel(stack),
            Cmd::Nget => nget(stack),
        },
    }
}

/// Execute the 'pop' command with a given stack.  Pops the top item off the
/// stack and discards it.
fn pop(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    if stack.is_empty() {
        return Err(Error::StackMissingToken.into());
    }
    stack.pop();
    Ok(())
}

/// Execute the 'add' command with a given stack.  Pops the two top integers off
/// the stack and adds them together.  Pushes the result back onto the stack.
fn add(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_operation(stack, "add", |x, y| x + y)
}

/// Execute the 'sub' command with a given stack.  Pops the two top integers off
/// the stack and does 'x - y' where 'x' was the second to top stack integer and
/// 'y' was the top stack integer.  Pushes the result back onto the stack.
fn sub(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_operation(stack, "sub", |x, y| x - y)
}

/// Execute the 'mul' command with a given stack.  Pops the two top integers off
/// the stack and multiplies them together.  Pushes the result back onto the stack.
fn mul(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_operation(stack, "mul", |x, y| x * y)
}

/// Execute the 'div' command with a given stack.  Pops the two top integers off
/// the stack and does x / y where 'x' is the second to top stack integer and
/// 'y' is the top stack integer.  Discards the remainder and pushes the result
/// back onto the stack.
fn div(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_operation(stack, "div", |x, y| x / y)
}

/// Executed the 'rem' command with a given stack.  Pops the two top integers off
/// the stack and does integer division pushing the remainder back onto the stack.
fn rem(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_operation(stack, "rem", |x, y| x - ((x / y) * y))
}

fn do_binary_operation(
    stack: &mut Vec<Token>,
    name: &str,
    op: fn(isize, isize) -> isize,
) -> anyhow::Result<()> {
    if stack.len() < 2 {
        return Err(Error::StackMissingToken.into());
    }

    let ty = stack.pop().unwrap();
    let tx = stack.pop().unwrap();
    match (tx, ty) {
        (Token::Num(x), Token::Num(y)) => {
            let res = op(x, y);
            stack.push(Token::Num(res));
        }
        (x, y) => {
            return Err(Error::StackBinaryCmd {
                x: x.to_string(),
                y: y.to_string(),
                cmd: name.to_string(),
            }
            .into());
        }
    }

    Ok(())
}

/// Execute the 'lt' command with a given stack.  Pops the two top integers off
/// the stack and does 'x < y' where 'x' was the second to top stack integer and
/// 'y' was the top stack integer.  If true, pushes '1' onto the stack.  If
/// false, pushes '0' onto the stack.
fn lt(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_truth_operation(stack, "lt", |x, y| x < y)
}

/// Execute the 'gt' command with a given stack.  Pops the two top integers off
/// the stack and does 'x > y' where 'x' was the second to top stack integer and
/// 'y' was the top stack integer.  If true, pushes '1' onto the stack.  If
/// false, pushes '0' onto the stack.
fn gt(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_truth_operation(stack, "gt", |x, y| x > y)
}

/// Execute the 'eq' command with a given stack.  Pops the two top integers off
/// the stack and does 'x > y' where 'x' was the second to top stack integer and
/// 'y' was the top stack integer.  If true, pushes '1' onto the stack.  If
/// false, pushes '0' onto the stack.
fn eq(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_truth_operation(stack, "eq", |x, y| x == y)
}

fn do_binary_truth_operation(
    stack: &mut Vec<Token>,
    name: &str,
    op: fn(isize, isize) -> bool,
) -> anyhow::Result<()> {
    if stack.len() < 2 {
        return Err(Error::StackMissingToken.into());
    }

    let ty = stack.pop().unwrap();
    let tx = stack.pop().unwrap();
    match (tx, ty) {
        (Token::Num(x), Token::Num(y)) => {
            let val = if op(x, y) { 1 } else { 0 };
            stack.push(Token::Num(val));
        }
        (x, y) => {
            return Err(Error::StackBinaryCmd {
                x: x.to_string(),
                y: y.to_string(),
                cmd: name.to_string(),
            }
            .into());
        }
    }

    Ok(())
}

/// Execute the 'swap' command with a given stack.  Pops the two top integers off
/// the stack and pushes them back on in the opposite order.
fn swap(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    if stack.len() < 2 {
        return Err(Error::StackMissingToken.into());
    }

    let ty = stack.pop().unwrap();
    let tx = stack.pop().unwrap();
    match (tx, ty) {
        (Token::Num(x), Token::Num(y)) => {
            stack.push(Token::Num(y));
            stack.push(Token::Num(x));
        }
        (x, y) => {
            return Err(Error::StackBinaryCmd {
                x: x.to_string(),
                y: y.to_string(),
                cmd: "swap".to_string(),
            }
            .into());
        }
    }

    Ok(())
}

/// Execute the 'sel' command with a given stack.  Pops the two three integers
/// off the stack, let us call them v1, v2, v3 (from top down).  If v3 == 0
/// pushes v1, if v3 is non-zero pushes v2.
fn sel(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    if stack.len() < 2 {
        return Err(Error::StackMissingToken.into());
    }

    let ty = stack.pop().unwrap();
    let tx = stack.pop().unwrap();
    let op = stack.pop().unwrap();
    match (op, tx, ty) {
        (Token::Num(op), Token::Num(x), Token::Num(y)) => {
            if op == 0 {
                stack.push(Token::Num(y));
            } else {
                stack.push(Token::Num(x));
            }
        }
        (op, x, y) => {
            return Err(Error::StackTernaryCmd {
                z: op.to_string(),
                x: x.to_string(),
                y: y.to_string(),
                cmd: "sel".to_string(),
            }
            .into());
        }
    }

    Ok(())
}

/// Execute the 'nget' command with a given stack.  Call the top value of stack
/// `vindex`, then subsequent values `v1`, `v2` ...  Pop `vindex`, the push `vi`
/// onto the stack.  Error if `i` is out of range or stack is empty.
fn nget(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    if stack.is_empty() {
        return Err(Error::StackMissingToken.into());
    }

    if let Token::Num(index) = stack.pop().unwrap() {
        let mut save = vec![];
        for _ in 0..index {
            if stack.is_empty() {
                return Err(Error::StackMissingToken.into());
            }
            save.push(stack.pop().unwrap());
        }
        save.reverse();
        let new = save[0].clone();

        for t in save.into_iter() {
            stack.push(t);
        }
        stack.push(new);
    } else {
        return Err(Error::StackWrongToken.into());
    }

    Ok(())
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut v = vec![];

        v.push("(postfix ".to_string());
        v.push(self.n_args.to_string());

        for t in self.tokens.iter() {
            v.push(" ".to_string());
            v.push(t.to_string());
        }

        for s in v.iter() {
            write!(f, "{}", s)?;
        }

        write!(f, ")")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("program parse error: missing opening parenthesis")]
    ProgramParseMissingOpeningParens,

    #[error("program parse error: missing closing parenthesis")]
    ProgramParseMissingClosingParens,

    #[error("program parse error: missing 'postfix'")]
    ProgramParseMissingPostfix,

    #[error("program parse error: missing numb args")]
    ProgramParseMissingNumArgs,

    #[error("args parse error: missing opening bracket")]
    ArgsParseMissingOpeningBracket,

    #[error("args parse error: missing closing bracket")]
    ArgsParseMissingClosingBracket,

    #[error("not enough tokens on stack")]
    StackMissingToken,

    #[error("incorrect token type on stack")]
    StackWrongToken,

    #[error("wrong tokens on stack: (`{x:?}`, `{y:?}`, `{cmd:?}`)")]
    StackBinaryCmd { x: String, y: String, cmd: String },

    #[error("wrong tokens on stack: (`{x:?}`, `{y:?}`, `{z:?}`, `{cmd:?}`)")]
    StackTernaryCmd {
        x: String,
        y: String,
        z: String,
        cmd: String,
    },

    #[error("cannot return token on stack: (`{0}`")]
    ReturnWrongToken(String),
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Token {
    Num(isize),
    Cmd(Cmd),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Token::Num(val) => val.to_string(),
            Token::Cmd(cmd) => cmd.to_string(),
        };
        write!(f, "{}", s)
    }
}

impl Token {
    fn new(s: &str) -> anyhow::Result<Token> {
        let token = if s.ends_with(')') {
            &s[..s.len() - 1]
        } else {
            &s
        };

        if let Ok(val) = token.parse::<isize>() {
            return Ok(Token::Num(val));
        }

        match token {
            "pop" => Ok(Token::Cmd(Cmd::Pop)),
            "add" => Ok(Token::Cmd(Cmd::Add)),
            "sub" => Ok(Token::Cmd(Cmd::Sub)),
            "mul" => Ok(Token::Cmd(Cmd::Mul)),
            "div" => Ok(Token::Cmd(Cmd::Div)),
            "rem" => Ok(Token::Cmd(Cmd::Rem)),
            "lt" => Ok(Token::Cmd(Cmd::Lt)),
            "gt" => Ok(Token::Cmd(Cmd::Gt)),
            "eq" => Ok(Token::Cmd(Cmd::Eq)),
            "swap" => Ok(Token::Cmd(Cmd::Swap)),
            "sel" => Ok(Token::Cmd(Cmd::Sel)),
            "nget" => Ok(Token::Cmd(Cmd::Nget)),
            _ => Err(anyhow!(format!(
                "Parsing token failed: unknown command: {}",
                s
            ))),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Cmd {
    Pop,
    Add,
    Sub,
    Mul,
    Div, // Integer division.
    Rem, // Integer remainder.
    Lt,
    Gt,
    Eq,
    Swap,
    Sel, // Select, ternary operator.
    Nget,
}

impl fmt::Display for Cmd {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Cmd::Pop => "pop",
            Cmd::Add => "add",
            Cmd::Sub => "sub",
            Cmd::Mul => "mul",
            Cmd::Div => "div",
            Cmd::Rem => "rem",
            Cmd::Lt => "lt",
            Cmd::Gt => "gt",
            Cmd::Eq => "eq",
            Cmd::Swap => "swap",
            Cmd::Sel => "sel",
            Cmd::Nget => "nget",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_split_valid_input_line_with_empty_args() {
        let s = "(postfix 0 1 2 add)[]";
        let (prog, args) = split(s);

        assert_eq!(prog, "(postfix 0 1 2 add)");
        assert_eq!(args, "[]");
    }

    #[test]
    fn can_split_valid_input_line_without_args() {
        let s = "(postfix 0 1 2 add)";
        let (prog, args) = split(s);

        assert_eq!(prog, "(postfix 0 1 2 add)");
        assert_eq!(args, "");
    }

    #[test]
    fn can_split_valid_input_line_with_args() {
        let s = "(postfix 3 1 2 add)[1, 2, 3]";
        let (prog, args) = split(s);

        assert_eq!(prog, "(postfix 3 1 2 add)");
        assert_eq!(args, "[1, 2, 3]");
    }

    #[test]
    fn can_parse_empty_args() {
        let input = "[]";
        let want = vec![];
        let got = parse_args(input).expect("can parse args string");

        assert_eq!(got, want);
    }

    #[test]
    fn can_parse_args() {
        let input = "[1, 2, 3]";
        let want = vec![3, 2, 1]; // We push the args onto the stack starting at the back.
        let got = parse_args(input).expect("can parse args string");

        assert_eq!(got, want);
    }

    #[test]
    fn program_parses_add() {
        let s = "(postfix 0 1 2 add)";
        let _ = Program::new(s).expect("trivial add program");
    }

    #[test]
    fn program_execs_add() {
        let s = "(postfix 0 1 2 add)";
        let prog = Program::new(s).expect("parse trivial add program");

        let mut stack = prog.exec(vec![]).expect("exec");
        let got = stack.pop().expect("valid stack");
        let want = Token::Num(3);

        assert_eq!(got, want);
    }

    #[test]
    fn program_execs_add_with_error() {
        let s = "(postfix 0 1 add)"; // Remember first 0 is number of args.
        let prog = Program::new(s).expect("parse trivial add program");

        if let Ok(_) = prog.exec(vec![]) {
            panic!("we should have error'ed, not enough args on stack for add")
        }
    }

    fn run_interpreter_with(s: &str) -> isize {
        interpret(s)
            .expect("valid input string")
            .expect("valid return stack")
    }

    #[test]
    fn interpret_can_pop() {
        let res = run_interpreter_with("(postfix 0 1 2 3 pop)[]");
        assert_eq!(res, 2);
    }

    #[test]
    fn interpret_can_add_with_empty_args() {
        let res = run_interpreter_with("(postfix 0 1 2 add)[]");
        assert_eq!(res, 3);
    }

    #[test]
    fn interpret_can_add_with_one_arg() {
        let res = run_interpreter_with("(postfix 1 2 add)[3]");
        assert_eq!(res, 5);
    }

    #[test]
    fn interpret_can_add_with_two_arg() {
        let res = run_interpreter_with("(postfix 2 add)[2 3]");
        assert_eq!(res, 5);
    }

    #[test]
    fn interpret_can_sub_with_empty_args() {
        let res = run_interpreter_with("(postfix 0 2 1 sub)[]");
        assert_eq!(res, 1);
    }

    #[test]
    fn interpret_can_sub_negative_result() {
        let res = run_interpreter_with("(postfix 0 1 2 sub)[]");
        assert_eq!(res, -1);
    }

    #[test]
    fn interpret_can_sub_with_one_arg() {
        let res = run_interpreter_with("(postfix 1 2 sub)[3]");
        assert_eq!(res, 1);
    }

    #[test]
    fn interpret_can_sub_with_two_arg() {
        let res = run_interpreter_with("(postfix 2 sub)[2 3]");
        assert_eq!(res, 1);
    }

    #[test]
    fn only_top_of_stack_returned() {
        let res = run_interpreter_with("(postfix 0 1 2 3)[]");
        assert_eq!(res, 3);
    }

    #[test]
    fn interpret_can_mul_with_one_arg() {
        let res = run_interpreter_with("(postfix 1 2 mul)[3]");
        assert_eq!(res, 6);
    }

    #[test]
    fn interpret_can_div_with_one_arg() {
        let res = run_interpreter_with("(postfix 1 2 div)[7]");
        assert_eq!(res, 3);
    }

    #[test]
    fn interpret_can_get_remainder_with_one_arg() {
        let res = run_interpreter_with("(postfix 1 2 rem)[7]");
        assert_eq!(res, 1);
    }

    #[test]
    fn interpret_can_lt_true() {
        let res = run_interpreter_with("(postfix 1 7 lt)[2]");
        assert_eq!(res, 1);
    }

    #[test]
    fn interpret_can_lt_false() {
        let res = run_interpreter_with("(postfix 1 2 lt)[7]");
        assert_eq!(res, 0);
    }

    #[test]
    fn interpret_can_gt_true() {
        let res = run_interpreter_with("(postfix 1 2 gt)[7]");
        assert_eq!(res, 1);
    }

    #[test]
    fn interpret_can_gt_false() {
        let res = run_interpreter_with("(postfix 1 7 gt)[2]");
        assert_eq!(res, 0);
    }

    #[test]
    fn interpret_can_eq_true() {
        let res = run_interpreter_with("(postfix 1 7 eq)[7]");
        assert_eq!(res, 1);
    }

    #[test]
    fn interpret_can_eq_false() {
        let res = run_interpreter_with("(postfix 1 7 eq)[2]");
        assert_eq!(res, 0);
    }

    #[test]
    fn interpret_can_swap() {
        let res = run_interpreter_with("(postfix 0 1 2 3 swap)[]");
        assert_eq!(res, 2); // stack == (1, 3, 2)
    }

    #[test]
    fn interpret_can_select_0() {
        let res = run_interpreter_with("(postfix 3 sel)[8, 9, 0]");
        assert_eq!(res, 8); // stack == (8)
    }

    #[test]
    fn interpret_can_select_1() {
        let res = run_interpreter_with("(postfix 3 sel)[8, 9, 1]");
        assert_eq!(res, 9); // stack == (9)
    }

    #[test]
    fn interpret_can_nget_index_1() {
        let res = run_interpreter_with("(postfix 0 1 2 3 1 nget)[]");
        assert_eq!(res, 3); // stack == (1, 2, 3, 3)
    }

    #[test]
    fn interpret_can_nget_middle_index() {
        let res = run_interpreter_with("(postfix 5 4 nget)[1, 2, 3, 4, 5]");
        assert_eq!(res, 4); // stack == (5, 4, 3, 2, 1, 4)
    }

    #[test]
    fn interpret_can_nget_last_index() {
        let res = run_interpreter_with("(postfix 3 6 7 5 nget)[7, 8, 9]");
        assert_eq!(res, 9); // stack == (9, 8, 7, 6, 7, 9)
    }

    #[test]
    fn can_validate_balanced_parens() {
        let s = "(postfix 0 1 2 sub)";
        let result = validate_program_string(s);
        result.expect("valid result")
    }

    #[should_panic]
    #[test]
    fn can_invalidate_unbalanced_close_parens() {
        let s = "(postfix 0 1 2 sub))";
        let result = validate_program_string(s);
        result.expect("invalid result")
    }

    #[should_panic]
    #[test]
    fn can_invalidate_unbalanced_open_parens() {
        let s = "(postfix (0 1 2 sub)";
        let result = validate_program_string(s);
        result.expect("invalid result")
    }
}
