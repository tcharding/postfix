use std::env;
use std::process;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage(&args[0]);
        process::exit(1);
    }
    let line = &args[1];
    postfix::interpret(line)?;

    Ok(())
}

fn usage(cmd: &str) {
    println!("Usage: {} '(postfix n-args <program>)[args]'", cmd);
}
