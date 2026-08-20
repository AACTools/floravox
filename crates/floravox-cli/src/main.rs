//! floravox — diagnostic CLI.
//!
//! Subcommands:
//!   timeline SSML   Parse input and dump segments/spans (no model needed)
//!   synth           Synthesize to WAV + events JSON (requires a voice)

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match run(&args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e:#}");
            1
        }
    };
    std::process::exit(code);
}

fn run(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("timeline") => cmd_timeline(&args[1..]),
        Some("synth") => cmd_synth(&args[1..]),
        Some("g2p") => cmd_g2p(&args[1..]),
        Some("help") | None => {
            print_help();
            Ok(())
        }
        Some(other) => {
            bail!("unknown command {other:?}; try `floravox help`");
        }
    }
}

fn print_help() {
    eprintln!(
        "floravox — event-driven SSML TTS diagnostics\n\n\
         USAGE:\n  \
         floravox timeline [INPUT]          dump parsed segments & word spans\n  \
         floravox g2p --phonetisaurus S W…  phonemize words (no voice needed)\n  \
         floravox synth --model M --text T  synthesize to out.wav + events.json\n\n\
         timeline reads stdin when INPUT is absent or `-`.\n\
         synth also takes --lexicon STEM (compiled stem.fst/.pho),\n\
         --phonetisaurus STEM (model.fst + grapheme/phoneme tables) and\n\
         --byt5-encoder/--byt5-decoder for OOV, plus --file F for text."
    );
}

fn read_input(path: Option<&String>) -> Result<String> {
    match path {
        Some(p) if p != "-" => std::fs::read_to_string(p).with_context(|| format!("reading {p}")),
        _ => {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .context("reading stdin")?;
            Ok(s)
        }
    }
}

fn cmd_timeline(args: &[String]) -> Result<()> {
    let input = read_input(args.first())?;
    let doc = floravox_ssml::parse(&input)?;
    let mut out = std::io::stdout().lock();
    for w in &doc.warnings {
        writeln!(out, "warning: {w}")?;
    }
    for (i, seg) in doc.segments.iter().enumerate() {
        match seg {
            floravox_ssml::Segment::Words { words } => {
                for w in words {
                    writeln!(
                        out,
                        "[{i:3}] word  {:?} char {}..{} byte {}..{} prosody={:?} say_as={:?}",
                        w.text,
                        w.char_span.start,
                        w.char_span.end,
                        w.byte_span.start,
                        w.byte_span.end,
                        w.prosody.rate.map_or(1.0, |r| r),
                        w.say_as,
                    )?;
                }
            }
            floravox_ssml::Segment::Break { ms, char_pos, .. } => {
                writeln!(out, "[{i:3}] break {ms}ms (char {char_pos})")?;
            }
            floravox_ssml::Segment::Mark { name, char_pos, .. } => {
                writeln!(out, "[{i:3}] mark  {name:?} (char {char_pos})")?;
            }
            floravox_ssml::Segment::SentenceEnd { char_pos, .. } => {
                writeln!(out, "[{i:3}] sentence-end (char {char_pos})")?;
            }
            floravox_ssml::Segment::ParagraphEnd { char_pos, .. } => {
                writeln!(out, "[{i:3}] paragraph-end (char {char_pos})")?;
            }
        }
    }
    Ok(())
}

fn cmd_synth(args: &[String]) -> Result<()> {
    let mut model: Option<String> = None;
    let mut text: Option<String> = None;
    let mut text_file: Option<String> = None;
    let mut lexicon_stem: Option<String> = None;
    let mut phonetisaurus_stem: Option<String> = None;
    let mut byt5_encoder: Option<String> = None;
    let mut byt5_decoder: Option<String> = None;
    let mut out_wav = "out.wav".to_string();
    let mut out_events = "events.json".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => model = args.get(i + 1).cloned(),
            "--text" => text = args.get(i + 1).cloned(),
            "--file" => text_file = args.get(i + 1).cloned(),
            "--lexicon" => lexicon_stem = args.get(i + 1).cloned(),
            "--phonetisaurus" => phonetisaurus_stem = args.get(i + 1).cloned(),
            "--byt5-encoder" => byt5_encoder = args.get(i + 1).cloned(),
            "--byt5-decoder" => byt5_decoder = args.get(i + 1).cloned(),
            "--out" => out_wav = args.get(i + 1).cloned().unwrap_or(out_wav),
            "--events" => out_events = args.get(i + 1).cloned().unwrap_or(out_events),
            other => bail!("unknown synth flag {other:?}"),
        }
        i += 2;
    }
    let Some(model_path) = model else {
        bail!("--model PATH is required (path to .onnx or its stem)");
    };
    let input = match (text, text_file) {
        (Some(t), _) => t,
        (None, Some(f)) => std::fs::read_to_string(&f)?,
        (None, None) => bail!("provide --text or --file"),
    };

    #[cfg(feature = "onnx")]
    {
        let cached = floravox_g2p::CachedPhonemizer::new(
            build_phonemizer(
                lexicon_stem.as_deref(),
                phonetisaurus_stem.as_deref(),
                byt5_encoder.as_deref(),
                byt5_decoder.as_deref(),
            )?,
            1024,
        );
        let voice = floravox_core::synth::VoiceModel::load(&model_path)?;
        println!(
            "model: {} Hz, {} phonemes, durations output: {}",
            voice.config.sample_rate,
            voice.config.phoneme_id_map.len(),
            voice.config.has_durations
        );
        let synth = floravox_core::synth::Synthesizer::new(voice, cached);
        let (samples, events, rate) = synth.synthesize(&input)?;
        write_wav(&out_wav, &samples, rate)?;
        let events_json: Vec<serde_json::Value> = events
            .iter()
            .map(|e| serde_json::to_value(e).unwrap_or_default())
            .collect();
        std::fs::write(&out_events, serde_json::to_string_pretty(&events_json)?)?;
        let words = events
            .iter()
            .filter(|e| matches!(e, floravox_core::SynthesisEvent::WordBoundary(_)))
            .count();
        let ms = events
            .iter()
            .find_map(|e| match e {
                floravox_core::SynthesisEvent::Finished { total_ms, .. } => Some(*total_ms),
                _ => None,
            })
            .unwrap_or(0);
        println!(
            "wrote {} ({} samples, {} ms) + {} ({words} word events)",
            out_wav,
            samples.len(),
            ms,
            out_events
        );
        Ok(())
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (
            model_path,
            input,
            out_wav,
            out_events,
            lexicon_stem,
            phonetisaurus_stem,
            byt5_encoder,
            byt5_decoder,
        );
        bail!("floravox-cli was built without the `onnx` feature");
    }
}

/// Assemble the phonemizer: lexicon-backed when a stem is given, empty
/// lexicon otherwise. OOV duty, cheapest engine first: `Phonetisaurus`
/// WFST, `ByT5`, then letter-name spelling.
#[cfg(feature = "onnx")]
fn build_phonemizer(
    lexicon_stem: Option<&str>,
    phonetisaurus_stem: Option<&str>,
    byt5_encoder: Option<&str>,
    byt5_decoder: Option<&str>,
) -> Result<Box<dyn floravox_g2p::TokenPhonemizer + Send>> {
    if byt5_encoder.is_some() != byt5_decoder.is_some() {
        bail!("--byt5-encoder and --byt5-decoder go together");
    }
    let mut fallback: Box<dyn floravox_g2p::OovFallback + Send> =
        Box::new(floravox_g2p::RuleFallback::default());
    if let (Some(enc), Some(dec)) = (byt5_encoder, byt5_decoder) {
        let byt5 = floravox_g2p::Byt5G2p::load(enc, dec).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("byt5 fallback: {enc} + {dec}");
        fallback = Box::new(floravox_g2p::ChainedFallback(byt5, fallback));
    }
    if let Some(stem) = phonetisaurus_stem {
        let ph =
            floravox_g2p::PhonetisaurusG2p::open(stem).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!(
            "phonetisaurus fallback: {stem}.fst ({} states, {} arcs)",
            ph.num_states(),
            ph.num_arcs()
        );
        fallback = Box::new(floravox_g2p::ChainedFallback(ph, fallback));
    }
    Ok(match lexicon_stem {
        Some(stem) => {
            let lexicon =
                floravox_g2p::MmapLexicon::open(stem).map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("lexicon: {stem}.fst/.pho ({} entries)", lexicon.len());
            Box::new(floravox_g2p::LexiconPhonemizer::new(lexicon, fallback))
        }
        None => Box::new(floravox_g2p::LexiconPhonemizer::new(
            floravox_g2p::FstLexicon::from_rows(Vec::new())?,
            fallback,
        )),
    })
}

/// `floravox g2p` — phonemize words with a Phonetisaurus model, no voice
/// required.
fn cmd_g2p(args: &[String]) -> Result<()> {
    let mut phonetisaurus_stem: Option<String> = None;
    let mut words: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--phonetisaurus" => {
                phonetisaurus_stem = args.get(i + 1).cloned();
                if phonetisaurus_stem.is_none() {
                    bail!("--phonetisaurus needs a model stem");
                }
                i += 2;
            }
            other if other.starts_with("--") => bail!("unknown g2p flag {other:?}"),
            other => {
                words.push(other.to_string());
                i += 1;
            }
        }
    }
    let Some(stem) = &phonetisaurus_stem else {
        bail!("g2p requires --phonetisaurus STEM (model.fst + tables)");
    };
    if words.is_empty() {
        bail!("give at least one word");
    }
    let model =
        floravox_g2p::PhonetisaurusG2p::open(stem).map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!(
        "model: {} states, {} arcs",
        model.num_states(),
        model.num_arcs()
    );
    for word in &words {
        match model.phonemize(word) {
            Some(phonemes) => println!("{word}\t{}", phonemes.join(" ")),
            None => println!("{word}\t(no path)"),
        }
    }
    Ok(())
}

/// Minimal 16-bit PCM WAV writer (mono). Sample counts and lengths are
/// bounded by real utterance sizes.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut pcm: Vec<i16> = Vec::with_capacity(samples.len());
    for &s in samples {
        let v = s.clamp(-1.0, 1.0);
        pcm.push((v * f32::from(i16::MAX)) as i16);
    }
    let data_len = pcm.len() * 2;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len as u32).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits
    f.write_all(b"data")?;
    f.write_all(&(data_len as u32).to_le_bytes())?;
    for s in &pcm {
        f.write_all(&s.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}
