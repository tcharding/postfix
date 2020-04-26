use postfix::Program;
use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::{env, process};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage(&args[0]);
        process::exit(1);
    }

    if &args[1] == "-f" || &args[1] == "--file" {
        run_file(&args[2])?;
    } else {
        run_line(&args[1])?;
    }

    Ok(())
}

fn usage(cmd: &str) {
    println!("Usage: {} '(postfix n-args <program>)[args]'", cmd);
}

fn run_file(s: &str) -> anyhow::Result<()> {
    let file = File::open(s)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if is_whitespace(&line) || is_comment(&line) {
            continue;
        }
        eprint!("{} -> ", line);
        run_line(&line)?;
        eprintln!("",);
    }

    Ok(())
}

fn is_whitespace(s: &str) -> bool {
    s == "" || s.starts_with(" ")
}

fn is_comment(s: &str) -> bool {
    s.starts_with("//")
}

fn run_line(s: &str) -> anyhow::Result<()> {
    let (program, args) = postfix::split(s)?;

    let mut program = Program::new(program)?;
    let args = postfix::parse_args_program_string(args)?;

    let result = program.run(args)?;

    match result {
        None => eprintln!("()"),
        Some(token) => eprintln!("{}", token),
    }

    Ok(())
}
