use anyhow::anyhow;
use std::dbg;
use std::fmt;

/// Interpret line and return the top item of the stack after execution.
pub fn interpret(line: &str) -> anyhow::Result<Option<isize>> {
    let (program, args) = parse_input_line(line)?;
    let mut stack = program.exec(args)?;

    match stack.pop() {
        Some(token) => match token {
            Token::Num(val) => Ok(Some(val)),
            _ => Err(anyhow!("invalid token on top of stack")),
        },
        None => Ok(None),
    }
}

fn parse_input_line(s: &str) -> anyhow::Result<(Program, Vec<isize>)> {
    let (program, args) = split_input_line(s);

    dbg!(program);
    dbg!(args);

    let program = Program::new(program)?;
    let args = parse_args(args)?;

    Ok((program, args))
}

fn split_input_line(s: &str) -> (&str, &str) {
    if let Some(index) = s.find("[") {
        return (&s[..index], &s[index..]);
    }
    (s, "")
}

fn parse_args(s: &str) -> anyhow::Result<Vec<isize>> {
    let mut v = vec![];

    let s = s.trim();

    if !s.starts_with("[") {
        return Err(arg_parse_error("missing leading '['"));
    }
    let s = &s[1..];

    if !s.ends_with("]") {
        return Err(arg_parse_error("missing trailing ']'"));
    }
    let s = &s[..s.len() - 1];

    for arg in s.split_ascii_whitespace() {
        let arg = if arg.ends_with(",") {
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

    /// The tokens that make up this program, these can be numbers or commands.
    tokens: Vec<Token>,
}

impl Program {
    /// Parses line and creates a `Program` object.
    /// Returns an `Err` if the line is not syntactically correct.
    fn new(line: &str) -> anyhow::Result<Program> {
        let line = line.trim();

        if !line.starts_with("(") {
            return Err(line_parse_error("missing leading '('"));
        }
        let line = &line[1..];

        if !line.ends_with(")") {
            return Err(line_parse_error("missing trailing ')'"));
        }
        let line = &line[..line.len() - 1];

        let mut iter = line.split_ascii_whitespace();

        match iter.next() {
            Some(token) if token == "postfix" => {}
            _ => return Err(line_parse_error("missing 'postfix'")),
        }

        let n_args = iter
            .next()
            .ok_or_else(|| line_parse_error("missing number of args"))?
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
                "wrong number of arguments: got {} want {}",
                args.len(),
                self.n_args
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

/// Handle a single token.
fn handle_token(stack: &mut Vec<Token>, token: Token) -> anyhow::Result<()> {
    match token {
        Token::Num(val) => stack.push(Token::Num(val)),
        Token::Cmd(cmd) => match cmd {
            Cmd::Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("not enough args to call 'add' command"));
                }
                let tx = stack.pop().unwrap();
                let ty = stack.pop().unwrap();
                match (tx, ty) {
                    (Token::Num(x), Token::Num(y)) => stack.push(Token::Num(x + y)),
                    (_, _) => return Err(anyhow!("not enough args to call 'add' command")),
                }
            }
        },
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

fn line_parse_error(msg: &str) -> anyhow::Error {
    anyhow!("Parsing line failed: {}", msg)
}

fn arg_parse_error(msg: &str) -> anyhow::Error {
    anyhow!("Parsing args string failed: {}", msg)
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
        let token = if s.ends_with(")") {
            &s[..s.len() - 1]
        } else {
            &s
        };

        if let Ok(val) = token.parse::<isize>() {
            return Ok(Token::Num(val));
        }

        match token {
            "add" => return Ok(Token::Cmd(Cmd::Add)),
            _ => {
                return Err(anyhow!(format!(
                    "Parsing token failed: unknown command: {}",
                    s
                )))
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Cmd {
    Add,
}

impl fmt::Display for Cmd {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Cmd::Add => "add",
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
        let (prog, args) = split_input_line(s);

        assert_eq!(prog, "(postfix 0 1 2 add)");
        assert_eq!(args, "[]");
    }

    #[test]
    fn can_split_valid_input_line_without_args() {
        let s = "(postfix 0 1 2 add)";
        let (prog, args) = split_input_line(s);

        assert_eq!(prog, "(postfix 0 1 2 add)");
        assert_eq!(args, "");
    }

    #[test]
    fn can_split_valid_input_line_with_args() {
        let s = "(postfix 0 1 2 add)[1, 2, 3]";
        let (prog, args) = split_input_line(s);

        assert_eq!(prog, "(postfix 0 1 2 add)");
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
}
