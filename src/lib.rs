use anyhow::Result;

/// Interpret line and return the top item of the stack after execution.
pub fn interpret(line: &str) -> Result<Token> {
    println!("Interpreting Postfix program: {}", line);
    let program = Program::new(line)?;
}

struct Program {
    n_args: usize,
    tokens: Vec<Token>,
}

impl Program {
    /// Parses line and returns a program object, error if line does not parse correctly.
    fn new(line: &str) -> Self {}
}

enum Token {
    Num(usize),
    Cmd(Cmd),
}

enum Cmd {
    Add,
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
