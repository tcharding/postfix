mod program;
mod token;

pub use program::Program;
pub use token::*;

/// Splits an input string into the two strings, program and args.
///
/// # Examples
///
/// ```
/// let s = "(postfix 1 2 add)[3]";
/// let (prog, args) = postfix::split(s).unwrap();
///
/// assert_eq!(prog, "(postfix 1 2 add)");
/// assert_eq!(args, "[3]");
/// ```
pub fn split(s: &str) -> anyhow::Result<(&str, &str)> {
    if let Some(index) = s.find('[') {
        return Ok((&s[..index], &s[index..]));
    }
    Err(Error::MissingClosingBracket.into())
}

/// Parse an args list into a list of integers, this is the argument string
/// passed to a PostFix program.
///
/// # Examples
/// ```
/// let args = "[1, 2, 3]";
/// let want = vec![1, 2, 3];
/// let got = postfix::parse_args_program_string(args).unwrap();
/// assert_eq!(got, want);
/// ```
pub fn parse_args_program_string(s: &str) -> anyhow::Result<Vec<isize>> {
    let mut v = vec![];

    let s = s.trim();

    if !s.starts_with('[') {
        return Err(Error::MissingOpeningBracket.into());
    }
    let s = &s[1..];

    if !s.ends_with(']') {
        return Err(Error::MissingClosingBracket.into());
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("missing opening square bracket")]
    MissingOpeningBracket,

    #[error("missing closing square bracket")]
    MissingClosingBracket,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_split_valid_input_line_with_empty_args() {
        let s = "(postfix 0 1 2 add)[]";
        let (prog, args) = split(s).expect("valid string");

        assert_eq!(prog, "(postfix 0 1 2 add)");
        assert_eq!(args, "[]");
    }

    #[test]
    fn can_split_valid_input_line_with_args() {
        let s = "(postfix 3 1 2 add)[1, 2, 3]";
        let (prog, args) = split(s).expect("valid string");

        assert_eq!(prog, "(postfix 3 1 2 add)");
        assert_eq!(args, "[1, 2, 3]");
    }
}
