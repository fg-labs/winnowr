//! The `winnowr` command-line tool.

#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::Parser;
use mimalloc::MiMalloc;

use winnowr::cli::{Cli, run};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Written directly rather than logged, so that no log filter can hide why winnowr
            // failed.
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
