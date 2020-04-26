use crate::token::Token;
use anyhow::anyhow;
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

        Ok(Program { n_args, tokens })
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
}
