// Measure streaming behavior: time-to-first-audio-chunk vs total, plus
// chunk interleave timing, for multi-sentence text.
#![allow(clippy::all, clippy::pedantic)]

use floravox_core::synth::Synthesizer;
use floravox_g2p::{CachedPhonemizer, LexiconPhonemizer, MmapLexicon, RuleFallback};
use std::sync::mpsc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let model = std::env::var("MODEL").expect("MODEL=path");
    let lex = std::env::var("LEX").ok();
    let text = std::env::var("TEXT").unwrap_or_else(|_| {
        "De eerste zin klinkt vrij snel. De tweede zin volgt daarna zonder pauze. \
         En een derde zin maakt het verhaal compleet voor deze test."
            .into()
    });

    let backend = floravox_core::load_voice(&model)?;
    let phon = match lex {
        Some(stem) => CachedPhonemizer::new(
            LexiconPhonemizer::new(MmapLexicon::open(stem)?.to_mem(), RuleFallback::default()),
            1024,
        ),
        None => CachedPhonemizer::new(
            LexiconPhonemizer::new(
                floravox_g2p::FstLexicon::from_rows(Vec::new())?,
                RuleFallback::default(),
            ),
            1024,
        ),
    };
    let synth = Synthesizer::new(backend, phon);

    let t0 = Instant::now();
    let stream = synth.synthesize_stream(&text)?;
    let mut first_chunk: Option<Instant> = None;
    let mut n_chunks = 0;
    let mut n_events = 0;
    let mut total_samples = 0usize;

    let (done_tx, done_rx) = mpsc::channel();
    let audio = stream.audio;
    let events = stream.events;
    std::thread::spawn(move || {
        for _ev in events {
            n_events += 1;
        }
        done_tx.send(n_events).ok();
    });
    for chunk in audio {
        if first_chunk.is_none() {
            first_chunk = Some(Instant::now());
            println!("time to first audio chunk: {:?} (t0 +)", t0.elapsed());
        }
        n_chunks += 1;
        total_samples += chunk.samples.len();
    }
    let total = t0.elapsed();
    let evs = done_rx.recv().unwrap_or(0);
    println!(
        "total: {:?} | {} chunks, {} samples ({:.2}s audio), {} events",
        total,
        n_chunks,
        total_samples,
        total_samples as f32 / 16000.0,
        evs
    );
    if let Some(fc) = first_chunk {
        let rest = total - fc.duration_since(t0);
        println!(
            "first chunk at {:.0} ms of {:.0} ms total → {:.0}% of work happened AFTER playback could start",
            fc.duration_since(t0).as_millis(),
            total.as_millis(),
            100.0 * rest.as_secs_f32() / total.as_secs_f32()
        );
    }
    Ok(())
}
