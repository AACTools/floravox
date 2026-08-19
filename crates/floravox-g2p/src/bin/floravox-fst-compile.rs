//! Compile a TSV lexicon (`word\tph1 ph2 ...` per line) into a floravox FST
//! lexicon pair (`stem.fst` + `stem.pho`).
//!
//! Usage:
//!   floravox-fst-compile INPUT.tsv `OUTPUT_STEM`
//!
//! INPUT may be `-` for stdin. Lines starting with `#` are comments.

use std::io::Read;

fn main() {
    let mut args = std::env::args().skip(1);
    let input = if let Some(i) = args.next() {
        i
    } else {
        eprintln!("usage: floravox-fst-compile INPUT.tsv OUTPUT_STEM");
        std::process::exit(2);
    };
    let stem = args.next().unwrap_or_else(|| {
        eprintln!("usage: floravox-fst-compile INPUT.tsv OUTPUT_STEM");
        std::process::exit(2);
    });

    let text = if input == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).expect("read stdin");
        s
    } else {
        std::fs::read_to_string(&input).unwrap_or_else(|e| {
            eprintln!("cannot read {input}: {e}");
            std::process::exit(1);
        })
    };

    let mut rows: Vec<(String, String)> = Vec::new();
    let mut bad = 0usize;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((w, p)) = line.split_once('\t') {
            rows.push((w.trim().to_string(), p.trim().to_string()))
        } else {
            bad += 1;
            if bad <= 5 {
                eprintln!("line {}: not TSV, skipped: {line:?}", n + 1);
            }
        }
    }
    if bad > 5 {
        eprintln!("... and {} more bad lines", bad - 5);
    }

    match floravox_g2p::LexiconWriter::new(&stem).write(rows) {
        Ok(count) => {
            println!("wrote {stem}.fst + {stem}.pho ({count} entries)");
        }
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(1);
        }
    }
}
