//! Compile a pronunciation lexicon into a floravox FST lexicon pair
//! (`stem.fst` + `stem.pho`).
//!
//! Usage:
//!   floravox-fst-compile [--format FMT] INPUT OUTPUT_STEM
//!
//! FMT: auto (default) | cmudict | ipa-tsv | tsv
//!
//! * cmudict — `WORD  P HH R AH1 N` (converted ARPABET → IPA)
//! * ipa-tsv — `word\tIPA` with unsegmented IPA (WikiPron, gruut)
//! * tsv     — `word\tph1 ph2` pre-segmented phonemes
//!
//! INPUT may be `-` for stdin. Lines starting with `#` (or `;;;` for
//! CMUDICT) are comments; malformed lines are counted and skipped.

use std::io::Read;

use floravox_g2p::{ingest, SourceFormat};

fn usage_exit() -> ! {
    eprintln!(
        "usage: floravox-fst-compile [--format auto|cmudict|ipa-tsv|tsv] INPUT OUTPUT_STEM"
    );
    std::process::exit(2);
}

fn main() {
    let mut format_arg: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--format" {
            format_arg = match args.next() {
                Some(value) => Some(value),
                None => usage_exit(),
            };
        } else if let Some(value) = arg.strip_prefix("--format=") {
            format_arg = Some(value.to_string());
        } else {
            positional.push(arg);
        }
    }
    if positional.len() != 2 {
        usage_exit();
    }
    let input = positional.remove(0);
    let stem = positional.remove(0);

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

    let format = match format_arg.as_deref() {
        Some("cmudict") => SourceFormat::CmuDict,
        Some("ipa-tsv") => SourceFormat::IpaTsv,
        Some("tsv") => SourceFormat::Tsv,
        Some("auto") | None => SourceFormat::detect(&text),
        Some(other) => {
            eprintln!("unknown format {other:?} (auto|cmudict|ipa-tsv|tsv)");
            std::process::exit(2);
        }
    };

    let ingested = ingest::parse(&text, format);
    if ingested.skipped > 0 {
        eprintln!("skipped {} malformed lines", ingested.skipped);
    }
    if ingested.unknown > 0 {
        eprintln!(
            "skipped {} lines with unmapped ARPABET symbols",
            ingested.unknown
        );
    }

    match floravox_g2p::LexiconWriter::new(&stem).write(ingested.rows) {
        Ok(count) => {
            println!("wrote {stem}.fst + {stem}.pho ({count} entries, format {format:?})");
        }
        Err(e) => {
            eprintln!("compile failed: {e}");
            std::process::exit(1);
        }
    }
}
