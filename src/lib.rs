use anyhow::anyhow;
use std::dbg;
use std::fmt;

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
/// let want = vec![1, 2, 3];
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
        let s = s.trim();

        if !s.starts_with('(') {
            return Err(Error::ProgramParseMissingOpeningParens.into());
        }
        let s = &s[1..];

        if !s.ends_with(')') {
            return Err(Error::ProgramParseMissingClosingParens.into());
        }
        let s = &s[..s.len() - 1];

        let mut iter = s.split_ascii_whitespace();

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
        },
    }
}

/// Executed the 'pop' command with a given stack.  Pops the top item off the stack (discarding it).
fn pop(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    if stack.is_empty() {
        return Err(Error::StackMissingToken.into());
    }
    stack.pop();
    Ok(())
}

/// Executed the 'add' command with a given stack.  Adds the top two integers
/// together and pushes the result back onto the stack.
fn add(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_operation(stack, "add", |x, y| x + y)
}

/// Executed the 'sub' command with a given stack.  Subtracts the top integer
/// on the stack from second to top integer and pushes the result back onto the stack.
fn sub(stack: &mut Vec<Token>) -> anyhow::Result<()> {
    do_binary_operation(stack, "add", |x, y| x - y)
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

    #[error("wrong tokens on stack: (`{x:?}`, `{y:?}`, `{cmd:?}`)")]
    StackBinaryCmd { x: String, y: String, cmd: String },

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
}

impl fmt::Display for Cmd {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Cmd::Pop => "pop",
            Cmd::Add => "add",
            Cmd::Sub => "sub",
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
        let want = vec![1, 2, 3];
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

    #[test]
    fn interpret_can_pop() {
        let s = "(postfix 0 1 2 3 pop)[]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = 2;

        assert_eq!(got, want);
    }

    #[test]
    fn interpret_can_add_with_empty_args() {
        let s = "(postfix 0 1 2 add)[]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = 3;

        assert_eq!(got, want);
    }

    #[test]
    fn interpret_can_add_with_one_arg() {
        let s = "(postfix 1 2 add)[3]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = 5;

        assert_eq!(got, want);
    }

    #[test]
    fn interpret_can_add_with_two_arg() {
        let s = "(postfix 2 add)[2 3]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = 5;

        assert_eq!(got, want);
    }

    #[test]
    fn interpret_can_sub_with_empty_args() {
        let s = "(postfix 0 2 1 sub)[]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = 1;

        assert_eq!(got, want);
    }

    #[test]
    fn interpret_can_sub_negative_result() {
        let s = "(postfix 0 1 2 sub)[]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = -1;

        assert_eq!(got, want);
    }

    #[test]
    fn interpret_can_sub_with_one_arg() {
        let s = "(postfix 1 2 sub)[3]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = 1;

        assert_eq!(got, want);
    }

    #[test]
    fn interpret_can_sub_with_two_arg() {
        let s = "(postfix 2 sub)[2 3]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = -1;

        assert_eq!(got, want);
    }

    #[test]
    fn only_top_of_stack_returned() {
        let s = "(postfix 0 1 2 3)[]";
        let res = interpret(s).expect("valid input string");

        let got = res.expect("valid stack");
        let want = 3;

        assert_eq!(got, want);
    }
}
