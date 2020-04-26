use crate::token::{Cmd, Token};
use anyhow::anyhow;
use std::collections::VecDeque;

/// Program holds the parsed Postfix program, only basic syntax validation is
/// done on construction.
pub struct Program {
    /// The number of arguments the program expects.
    n_args: usize,

    /// The tokens that make up this program.
    tokens: VecDeque<Token>,

    /// Execution stack
    stack: Vec<Token>,
}

impl Program {
    /// Parses `s` and creates a `Program` object.
    /// Returns an `Err` if the line is not syntactically correct.
    pub fn new(s: &str) -> anyhow::Result<Program> {
        validate_program_string(s)?;

        let s = s.trim();

        if !s.starts_with('(') {
            return Err(anyhow!(Error::MissingOpeningParens));
        }
        let s = &s[1..];

        if !s.ends_with(')') {
            return Err(anyhow!(Error::MissingClosingParens));
        }
        let s = &s[..s.len() - 1];

        let s = s.trim();

        let mut iter = PostfixIterator::new(s);

        match iter.next() {
            Some(token) if token == "postfix" => {}
            _ => return Err(Error::MissingPostfix.into()),
        }

        let n_args = iter
            .next()
            .ok_or_else(|| Error::MissingNumArgs)?
            .parse::<usize>()?;

        let mut tokens = VecDeque::new();

        for t in iter {
            let token = Token::new(&t)?;
            tokens.push_back(token);
        }

        Ok(Program {
            n_args,
            tokens,
            stack: vec![],
        })
    }

    pub fn print_stack(&self) {
        eprint!("( ");
        for t in self.stack.iter() {
            eprint!(" {} ", t);
        }
        eprintln!(")");
    }

    /// Run the program and return the top item from the stack.
    pub fn run(&mut self, args: Vec<isize>) -> anyhow::Result<Option<Token>> {
        if self.n_args != args.len() {
            return Err(anyhow!(
                "wrong number of arguments, expected {}: {:?}",
                self.n_args,
                args,
            ));
        }

        // New stack for each program execution.
        self.stack = Vec::new();

        // In line with the spec, we push the args onto the stack in reverse order.
        for arg in args.iter().rev() {
            self.stack.push(Token::Num(*arg));
        }

        while self.tokens.len() > 0 {
            let token = self.tokens.pop_front().unwrap();
            //            self.print_stack();
            self.handle_token(&token)?;
        }

        Ok(self.stack.pop())
    }

    /// Handle a single token during execution of a program.
    ///
    /// As per the specification, integers and sequences are pushed onto the
    /// stack. Commands are executed with the current stack.
    fn handle_token(&mut self, token: &Token) -> anyhow::Result<()> {
        match token {
            Token::Num(val) => {
                self.stack.push(Token::Num(*val));
                Ok(())
            }
            Token::Seq(s) => {
                self.stack.push(Token::Seq(s.clone()));
                Ok(())
            }
            Token::Cmd(cmd) => match cmd {
                Cmd::Pop => self.pop(),
                Cmd::Add => self.add(),
                Cmd::Sub => self.sub(),
                Cmd::Mul => self.mul(),
                Cmd::Div => self.div(),
                Cmd::Rem => self.rem(),
                Cmd::Lt => self.lt(),
                Cmd::Gt => self.gt(),
                Cmd::Eq => self.eq(),
                Cmd::Swap => self.swap(),
                Cmd::Sel => self.sel(),
                Cmd::Nget => self.nget(),
                Cmd::Exec => self.exec(),
            },
        }
    }

    /// Execute the 'pop' command with a given stack.  Pops the top item off the
    /// stack and discards it.
    fn pop(&mut self) -> anyhow::Result<()> {
        if self.stack.is_empty() {
            return Err(Error::MissingToken.into());
        }
        self.stack.pop();
        Ok(())
    }

    /// Execute the 'add' command with a given stack.  Pops the two top integers off
    /// the stack and adds them together.  Pushes the result back onto the stack.
    fn add(&mut self) -> anyhow::Result<()> {
        self.binary_operation("add", |x, y| x + y)
    }

    /// Execute the 'sub' command with a given stack.  Pops the two top integers off
    /// the stack and does 'x - y' where 'x' was the second to top stack integer and
    /// 'y' was the top stack integer.  Pushes the result back onto the stack.
    fn sub(&mut self) -> anyhow::Result<()> {
        self.binary_operation("sub", |x, y| x - y)
    }

    /// Execute the 'mul' command with a given stack.  Pops the two top integers off
    /// the stack and multiplies them together.  Pushes the result back onto the stack.
    fn mul(&mut self) -> anyhow::Result<()> {
        self.binary_operation("mul", |x, y| x * y)
    }

    /// Execute the 'div' command with a given stack.  Pops the two top integers off
    /// the stack and does x / y where 'x' is the second to top stack integer and
    /// 'y' is the top stack integer.  Discards the remainder and pushes the result
    /// back onto the stack.
    fn div(&mut self) -> anyhow::Result<()> {
        self.binary_operation("div", |x, y| x / y)
    }

    /// Executed the 'rem' command with a given stack.  Pops the two top integers off
    /// the stack and does integer division pushing the remainder back onto the stack.
    fn rem(&mut self) -> anyhow::Result<()> {
        self.binary_operation("rem", |x, y| x - ((x / y) * y))
    }

    fn binary_operation(
        &mut self,
        name: &str,
        op: fn(isize, isize) -> isize,
    ) -> anyhow::Result<()> {
        if self.stack.len() < 2 {
            return Err(Error::MissingToken.into());
        }

        let ty = self.stack.pop().unwrap();
        let tx = self.stack.pop().unwrap();
        match (tx, ty) {
            (Token::Num(x), Token::Num(y)) => {
                let res = op(x, y);
                self.stack.push(Token::Num(res));
            }
            (x, y) => {
                return Err(Error::BinaryCmd {
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
    fn lt(&mut self) -> anyhow::Result<()> {
        self.binary_truth_operation("lt", |x, y| x < y)
    }

    /// Execute the 'gt' command with a given stack.  Pops the two top integers off
    /// the stack and does 'x > y' where 'x' was the second to top stack integer and
    /// 'y' was the top stack integer.  If true, pushes '1' onto the stack.  If
    /// false, pushes '0' onto the stack.
    fn gt(&mut self) -> anyhow::Result<()> {
        self.binary_truth_operation("gt", |x, y| x > y)
    }

    /// Execute the 'eq' command with a given stack.  Pops the two top integers off
    /// the stack and does 'x > y' where 'x' was the second to top stack integer and
    /// 'y' was the top stack integer.  If true, pushes '1' onto the stack.  If
    /// false, pushes '0' onto the stack.
    fn eq(&mut self) -> anyhow::Result<()> {
        self.binary_truth_operation("eq", |x, y| x == y)
    }

    fn binary_truth_operation(
        &mut self,
        name: &str,
        op: fn(isize, isize) -> bool,
    ) -> anyhow::Result<()> {
        if self.stack.len() < 2 {
            return Err(Error::MissingToken.into());
        }

        let ty = self.stack.pop().unwrap();
        let tx = self.stack.pop().unwrap();
        match (tx, ty) {
            (Token::Num(x), Token::Num(y)) => {
                let val = if op(x, y) { 1 } else { 0 };
                self.stack.push(Token::Num(val));
            }
            (x, y) => {
                return Err(Error::BinaryCmd {
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
    fn swap(&mut self) -> anyhow::Result<()> {
        if self.stack.len() < 2 {
            return Err(Error::MissingToken.into());
        }

        let x = self.stack.pop().unwrap();
        let y = self.stack.pop().unwrap();
        self.stack.push(x);
        self.stack.push(y);

        Ok(())
    }

    /// Execute the 'sel' command with a given stack.  Pops the two three integers
    /// off the stack, let us call them v1, v2, v3 (from top down).  If v3 == 0
    /// pushes v1, if v3 is non-zero pushes v2.
    fn sel(&mut self) -> anyhow::Result<()> {
        if self.stack.len() < 2 {
            return Err(Error::MissingToken.into());
        }

        let ty = self.stack.pop().unwrap();
        let tx = self.stack.pop().unwrap();
        let op = self.stack.pop().unwrap();
        match (op, tx, ty) {
            (Token::Num(op), x, y) => {
                if op == 0 {
                    self.stack.push(y);
                } else {
                    self.stack.push(x);
                }
            }
            (op, x, y) => {
                return Err(Error::TernaryCmd {
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
    fn nget(&mut self) -> anyhow::Result<()> {
        if self.stack.is_empty() {
            return Err(Error::MissingToken.into());
        }

        if let Token::Num(index) = self.stack.pop().unwrap() {
            let mut save = vec![];
            for _ in 0..index {
                if self.stack.is_empty() {
                    return Err(Error::MissingToken.into());
                }
                save.push(self.stack.pop().unwrap());
            }
            save.reverse();
            let new = save[0].clone();

            for t in save.into_iter() {
                self.stack.push(t);
            }
            self.stack.push(new);
        } else {
            return Err(Error::WrongToken.into());
        }

        Ok(())
    }

    /// Execute the 'exec' command with a given stack.  Pop the top item off the
    /// stack (it must be and executable sequence).  Pre-pend the sequence to
    /// the list of tokens making up the currently running program.
    fn exec(&mut self) -> anyhow::Result<()> {
        if self.stack.is_empty() {
            return Err(anyhow!(Error::MissingToken));
        }

        if let Token::Seq(seq) = self.stack.pop().unwrap() {
            self.prepend_sequence(&seq)?;
        } else {
            return Err(anyhow!(Error::WrongToken));
        }

        Ok(())
    }

    // Pre-pends the sequence to the program dequeue of executing tokens.
    fn prepend_sequence(&mut self, seq: &str) -> anyhow::Result<()> {
        let seq = &seq[1..seq.len() - 1]; // Remove parens, we know sequence is valid.
        let iter = PostfixIterator::new(&seq);

        let mut v = vec![];
        for t in iter {
            v.push(String::from(t));
        }

        for t in v.iter().rev() {
            let token = Token::new(&t)?;
            self.tokens.push_front(token);
        }
        Ok(())
    }
}

fn validate_program_string(s: &str) -> anyhow::Result<()> {
    validate_parenthesis(s)?;
    validate_length(s)?;
    validate_contains_postfix(s)?;

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
        return Err(anyhow!(Error::MissingOpeningParens));
    }

    if close_parens < open_parens {
        return Err(anyhow!(Error::MissingClosingParens));
    }

    Ok(())
}

fn validate_length(s: &str) -> anyhow::Result<()> {
    let v: Vec<&str> = s.split_ascii_whitespace().collect();
    if v.len() < 2 {
        return Err(anyhow!(Error::ProgramTooShort));
    }
    Ok(())
}

fn validate_contains_postfix(s: &str) -> anyhow::Result<()> {
    if !s.contains("postfix") {
        return Err(anyhow!(Error::MissingPostfix));
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

        // Handle executable sequence.
        if self.s.starts_with("(") {
            let index = index_of_closing_parenthesis(&self.s) + 1;

            let tmp = self.s.split_off(index);
            let token = self.s.clone();
            self.s = tmp;

            self.s = self.s.trim_start().to_string();

            return Some(token);
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

// `s` is guaranteed to be valid.
fn index_of_closing_parenthesis(s: &str) -> usize {
    let mut count = 0;

    for (i, c) in s.char_indices() {
        if c == '(' {
            count += 1;
        } else if c == ')' {
            count -= 1;
            if count == 0 {
                return i;
            }
        }
    }
    unreachable!()
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("missing opening parenthesis")]
    MissingOpeningParens,

    #[error("missing closing parenthesis")]
    MissingClosingParens,

    #[error("program is too short, most basic is: (postfix 0)")]
    ProgramTooShort,

    #[error("missing 'postfix'")]
    MissingPostfix,

    #[error("missing number of arguments")]
    MissingNumArgs,

    #[error("not enough tokens on stack")]
    MissingToken,

    #[error("incorrect token type on stack")]
    WrongToken,

    #[error("wrong tokens on stack: (`{x:?}`, `{y:?}`, `{cmd:?}`)")]
    BinaryCmd { x: String, y: String, cmd: String },

    #[error("wrong tokens on stack: (`{x:?}`, `{y:?}`, `{z:?}`, `{cmd:?}`)")]
    TernaryCmd {
        x: String,
        y: String,
        z: String,
        cmd: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic() {
        let s = "(postfix 0 1 2 add)";
        let _ = Program::new(s).expect("trivial add program");
    }

    #[test]
    fn parses_complex() {
        let s = "(postfix 0 1 2 3 add sub (1 gt (2 3 mul)) exec)";
        let _ = Program::new(s).expect("trivial add program");
    }

    // Runs a program and returns the value on top of stack.
    fn run(program: &str, args: Vec<isize>) -> isize {
        let mut p = Program::new(program).expect("valid program");
        p.run(args)
            .expect("valid program execution")
            .expect("token on stack")
            .value()
            .expect("value token")
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

    #[test]
    fn iterator_can_handle_simple_executable_sequences() {
        let s = "(1 sub)";
        let mut iter = PostfixIterator::new(&s);
        let got = iter.next().unwrap();

        assert_eq!(got, s)
    }

    #[test]
    fn iterator_can_handle_nested_executable_sequences() {
        let s = "(1 sub (1 sub))";
        let mut iter = PostfixIterator::new(&s);
        let got = iter.next().unwrap();

        assert_eq!(got, s)
    }

    #[test]
    fn can_pop() {
        let res = run("(postfix 0 1 2 3 pop)", vec![]);
        assert_eq!(res, 2);
    }

    #[test]
    fn can_add_with_empty_args() {
        let res = run("(postfix 0 1 2 add)", vec![]);
        assert_eq!(res, 3);
    }

    #[should_panic]
    #[test]
    fn can_add_with_error() {
        run("(postfix 0 1 add)", vec![]); // Remember first 0 is number of args.
    }

    #[test]
    fn can_add_with_one_arg() {
        let res = run("(postfix 1 2 add)", vec![3]);
        assert_eq!(res, 5);
    }

    #[test]
    fn can_add_with_two_arg() {
        let res = run("(postfix 2 add)", vec![2, 3]);
        assert_eq!(res, 5);
    }

    #[test]
    fn can_sub_with_empty_args() {
        let res = run("(postfix 0 2 1 sub)", vec![]);
        assert_eq!(res, 1);
    }

    #[test]
    fn can_sub_negative_result() {
        let res = run("(postfix 0 1 2 sub)", vec![]);
        assert_eq!(res, -1);
    }

    #[test]
    fn can_sub_with_one_arg() {
        let res = run("(postfix 1 2 sub)", vec![3]);
        assert_eq!(res, 1);
    }

    #[test]
    fn can_sub_with_two_arg() {
        let res = run("(postfix 2 sub)", vec![2, 3]);
        assert_eq!(res, 1);
    }

    #[test]
    fn only_top_of_stack_returned() {
        let res = run("(postfix 0 1 2 3)", vec![]);
        assert_eq!(res, 3);
    }

    #[test]
    fn can_mul_with_one_arg() {
        let res = run("(postfix 1 2 mul)", vec![3]);
        assert_eq!(res, 6);
    }

    #[test]
    fn can_div_with_one_arg() {
        let res = run("(postfix 1 2 div)", vec![7]);
        assert_eq!(res, 3);
    }

    #[test]
    fn can_get_remainder_with_one_arg() {
        let res = run("(postfix 1 2 rem)", vec![7]);
        assert_eq!(res, 1);
    }

    #[test]
    fn can_lt_true() {
        let res = run("(postfix 1 7 lt)", vec![2]);
        assert_eq!(res, 1);
    }

    #[test]
    fn lt_false() {
        let res = run("(postfix 1 2 lt)", vec![7]);
        assert_eq!(res, 0);
    }

    #[test]
    fn gt_true() {
        let res = run("(postfix 1 2 gt)", vec![7]);
        assert_eq!(res, 1);
    }

    #[test]
    fn gt_false() {
        let res = run("(postfix 1 7 gt)", vec![2]);
        assert_eq!(res, 0);
    }

    #[test]
    fn eq_true() {
        let res = run("(postfix 1 7 eq)", vec![7]);
        assert_eq!(res, 1);
    }

    #[test]
    fn can_eq_false() {
        let res = run("(postfix 1 7 eq)", vec![2]);
        assert_eq!(res, 0);
    }

    #[test]
    fn can_swap() {
        let res = run("(postfix 0 1 2 3 swap)", vec![]);
        assert_eq!(res, 2); // stack == (1, 3, 2)
    }

    #[test]
    fn can_select_0() {
        let res = run("(postfix 3 sel)", vec![8, 9, 0]);
        assert_eq!(res, 8); // stack == (8)
    }

    #[test]
    fn can_select_1() {
        let res = run("(postfix 3 sel)", vec![8, 9, 1]);
        assert_eq!(res, 9); // stack == (9)
    }

    #[test]
    fn can_nget_index_1() {
        let res = run("(postfix 0 1 2 3 1 nget)", vec![]);
        assert_eq!(res, 3); // stack == (1, 2, 3, 3)
    }

    #[test]
    fn can_nget_middle_index() {
        let res = run("(postfix 5 4 nget)", vec![1, 2, 3, 4, 5]);
        assert_eq!(res, 4); // stack == (5, 4, 3, 2, 1, 4)
    }

    #[test]
    fn can_nget_last_index() {
        let res = run("(postfix 3 6 7 5 nget)", vec![7, 8, 9]);
        assert_eq!(res, 9); // stack == (9, 8, 7, 6, 7, 9)
    }

    #[test]
    fn can_exec_basic_sequence() {
        let res = run("(postfix 0 2 (1 add) exec)", vec![]);
        assert_eq!(res, 3);
    }

    #[test]
    fn can_exec_complex_sequence() {
        let res = run("(postfix 0 1 (2 add (4 mul) exec) exec)", vec![]);
        assert_eq!(res, 12);
    }
}
