use anyhow::anyhow;
use std::fmt;

/// Interpret line and return the top item of the stack after execution.
pub fn interpret(line: &str) -> anyhow::Result<ExecResult> {
    println!("Interpreting Postfix program: {}", line);
    let program = Program::new(line)?;

    // TODO: We don't currently handle args.
    let empty = vec![];
    program.exec(empty)
}

/// Program holds the parsed Postfix program, guaranteed to be syntactically correct.
struct Program {
    /// The number of arguments the program expects.
    n_args: usize,
    /// The tokens that make up this program.
    tokens: Vec<Token>,
}

impl Program {
    /// Parses line and creates a `Program` object.
    /// Returns an `Err` if line does not parse correctly.
    fn new(line: &str) -> anyhow::Result<Program> {
        let line = line.trim();

        if !line.starts_with("(") {
            return Err(line_parse_error("missing leading '('"));
        }
        let line = &line[1..]; // FIXME: Does this work with UTF-8 characters?

        if !line.ends_with(")") {
            return Err(line_parse_error("missing trailing ')'"));
        }
        let line = &line[..line.len() - 1]; // FIXME: Does this work with UTF-8 characters?

        let mut iter = line.split_whitespace();

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
    fn exec(&self, args: Vec<usize>) -> anyhow::Result<ExecResult> {
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

        if stack.len() < 1 {
            return Err(anyhow!("stack empty"));
        }

        let token = stack.pop();
        match token {
            Some(Token::Num(val)) => Ok(ExecResult::Value(val)),
            _ => Err(anyhow!("invalid top of stack after program execution")),
        }
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

/// Results that program execution can return.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ExecResult {
    Value(usize),
}

impl fmt::Display for ExecResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ExecResult::Value(val) => write!(f, "{}", val),
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum Token {
    Num(usize),
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

        if let Ok(val) = token.parse::<usize>() {
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

#[derive(Debug, Copy, Clone)]
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
    fn program_parses_add() {
        let s = "(postfix 0 1 2 add)";
        let _ = Program::new(s).expect("trivial add program");
    }

    #[test]
    fn program_execs_add() {
        let s = "(postfix 0 1 2 add)";
        let prog = Program::new(s).expect("parse trivial add program");

        let got = prog.exec(vec![]).expect("exec");
        let want = ExecResult::Value(3);

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
}
