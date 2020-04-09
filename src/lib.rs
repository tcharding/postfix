use anyhow::anyhow;
use std::fmt;

/// Interpret line and return the top item of the stack after execution.
pub fn interpret(line: &str) -> anyhow::Result<ExecResult> {
    println!("Interpreting Postfix program: {}", line);
    let program = Program::new(line)?;

    program.exec()
}

struct Program {
    n_args: usize,
    tokens: Vec<Token>,
}

impl Program {
    /// Parses line and creates a `Program` object.
    /// Returns an `Err` if line does not parse correctly.
    fn new(line: &str) -> anyhow::Result<Self> {
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
    fn exec(&self) -> anyhow::Result<ExecResult> {
        // Quieten the compiler.
        println!("Program: {}", self);

        Ok(ExecResult::Value(0))
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
pub enum ExecResult {
    Value(usize),
}

enum Token {
    Num(usize),
    Cmd(Cmd),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

enum Cmd {
    Add,
}

impl fmt::Display for Cmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
}
