//! Romanize text on stdin (any script to Latin) using the ported
//! uroman tables.
//!
//! Usage:
//!   floravox-romanize [--lang ISO639-3] < words.txt

use std::io::{BufRead, Write};

fn main() {
    let mut lang: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--lang" => lang = args.next(),
            other => {
                eprintln!("unknown flag {other:?}");
                std::process::exit(2);
            }
        }
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let rom = floravox_g2p::uroman::romanize(line.trim(), lang.as_deref());
        writeln!(out, "{rom}").expect("write stdout");
    }
}
