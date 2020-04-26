use postfix::Program;
use std::{env, process};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage(&args[0]);
        process::exit(1);
    }

    let s = &args[1];
    let (program, args) = postfix::split(s)?;

    let mut program = Program::new(program)?;
    let args = postfix::parse_args_program_string(args)?;

    let result = program.run(args)?;

    match result {
        None => println!("(empty stack)"),
        Some(token) => println!("result: {}", token),
    }

    Ok(())
}

fn usage(cmd: &str) {
    println!("Usage: {} '(postfix n-args <program>)[args]'", cmd);
}
