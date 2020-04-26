use anyhow::anyhow;
use std::fmt;

/// The tokens that make up a postfix program.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Num(isize),
    Cmd(Cmd),
    Seq(String), // An executable sequence e.g., "(sub 1)".
}

impl Token {
    pub fn new(s: &str) -> anyhow::Result<Token> {
        if s.starts_with("(") {
            return Ok(Token::Seq(s.to_string()));
        }

        if let Ok(val) = s.parse::<isize>() {
            return Ok(Token::Num(val));
        }

        if is_valid_command(s) {
            return Ok(Token::Cmd(Cmd::from_str(s).unwrap()));
        }

        Err(anyhow!(Error::UnknownToken(s.to_string())))
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Num(val) => write!(f, "{}", val),
            Token::Cmd(cmd) => write!(f, "{}", cmd),
            Token::Seq(seq) => write!(f, "{}", seq),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Cmd {
    Pop,
    Add,
    Sub,
    Mul,
    Div, // Integer division.
    Rem, // Integer remainder.
    Lt,
    Gt,
    Eq,
    Swap, // Swap top two elements on stack.
    Sel,  // Select, ternary operator.
    Nget, // Push element at index n onto stack.
    Exec, // Pop and execute an executable sequence.
}

fn is_valid_command(s: &str) -> bool {
    match s {
        "pop" => true,
        "add" => true,
        "sub" => true,
        "mul" => true,
        "div" => true,
        "rem" => true,
        "lt" => true,
        "gt" => true,
        "eq" => true,
        "swap" => true,
        "sel" => true,
        "nget" => true,
        "exec" => true,
        _ => false,
    }
}

impl Cmd {
    fn from_str(s: &str) -> anyhow::Result<Cmd> {
        match s {
            "pop" => Ok(Cmd::Pop),
            "add" => Ok(Cmd::Add),
            "sub" => Ok(Cmd::Sub),
            "mul" => Ok(Cmd::Mul),
            "div" => Ok(Cmd::Div),
            "rem" => Ok(Cmd::Rem),
            "lt" => Ok(Cmd::Lt),
            "gt" => Ok(Cmd::Gt),
            "eq" => Ok(Cmd::Eq),
            "swap" => Ok(Cmd::Swap),
            "sel" => Ok(Cmd::Sel),
            "nget" => Ok(Cmd::Nget),
            "exec" => Ok(Cmd::Exec),
            _ => Err(anyhow!(Error::UnknownCmd(s.to_string()))),
        }
    }
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
            Cmd::Exec => "exec",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("unknown token: {0}")]
    UnknownToken(String),

    #[error("unknown command: {0}")]
    UnknownCmd(String),
}
