#![allow(clippy::all, clippy::pedantic)]

// Full-pipeline bench (warm, in-process): synthesize() end to end.
use floravox_core::synth::Synthesizer;
use floravox_g2p::{CachedPhonemizer, LexiconPhonemizer, MmapLexicon, RuleFallback};

fn rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        // No /proc on macOS: point-in-time RSS via ps (KB). Sampled after
        // synthesis with the arena at its high-water mark, so it tracks
        // the Linux VmHWM number closely in practice.
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output();
        out.ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .map_or(0.0, |kb| kb / 1024.0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if let Some(v) = line.strip_prefix("VmHWM:") {
                return v
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse::<i64>()
                    .unwrap_or(0) as f64
                    / 1024.0;
            }
        }
        0.0
    }
}

fn main() -> anyhow::Result<()> {
    let model = std::env::var("MODEL").expect("MODEL=path");
    let text = std::env::var("TEXT").unwrap_or_else(|_| "De grenzen van mijn taal zijn de grenzen van mijn wereld. De tooverberg strekt zich uit over vele bladzijden.".into());
    let lex_stem = std::env::var("LEX").ok();

    let backend = floravox_core::load_voice(&model)?;
    let phon = match &lex_stem {
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

    // warmup (loads nothing extra; exercises full path)
    let _ = synth.synthesize(&text)?;
    let mut best = std::time::Duration::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let (samples, events, rate) = synth.synthesize(&text)?;
        best = best.min(t.elapsed());
        println!(
            "synthesize: {:.0} ms | {} samples ({:.2}s audio) | {} events | RTF {:.3}",
            best.as_millis(),
            samples.len(),
            samples.len() as f32 / rate as f32,
            events.len(),
            best.as_secs_f32() / (samples.len() as f32 / rate as f32),
        );
        println!("peak RSS: {:.0} MB", rss_mb());
    }
    Ok(())
}
