use std::env;
use std::process;

fn main() -> anyhow::Result<()> {
    if let Some(path) = env::args().nth(1) {
        ive::run(path)
    } else {
        eprintln!("usage: {} <file>", env::args().next().unwrap());
        process::exit(1);
    }
}
