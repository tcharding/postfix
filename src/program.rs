use crate::token::Token;
use std::collections::VecDeque;

/// Program holds the parsed Postfix program, only basic syntax validation is
/// done on construction.
pub struct Program {
    /// The number of arguments the program expects.
    n_args: usize,

    /// The tokens that make up this program.
    tokens: VecDeque<Token>,
}

impl Program {
    /// Parses `s` and creates a `Program` object.
    /// Returns an `Err` if the line is not syntactically correct.
    pub fn new(_s: &str) -> anyhow::Result<Program> {
        Ok(Program::default())
    }

    /// Run the program and return the top item from the stack.
    pub fn run(&mut self, _args: Vec<isize>) -> anyhow::Result<Option<Token>> {
        Ok(None)
    }
}

impl Default for Program {
    fn default() -> Self {
        let tokens: VecDeque<Token> = VecDeque::new();
        Program { n_args: 0, tokens }
    }
}
