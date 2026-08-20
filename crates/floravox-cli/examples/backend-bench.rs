#![allow(clippy::all, clippy::pedantic)]

// Micro-bench: time VoiceBackend::run alone on the same id sequence.
// BENCH_IDS=n cargo run --release -p floravox-cli --example... (uses test harness instead)

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
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let model = std::env::var("MODEL").expect("MODEL=path");
    let before = rss_mb();
    let mut backend = floravox_core::load_voice(&model)?;
    eprintln!(
        "model load: RSS {before:.0} -> {:.0} MB (+{:.0})",
        rss_mb(),
        rss_mb() - before
    );
    let ids: Vec<i64> = (0..n as i64).map(|i| (i % 30) + 4).collect();
    // warmup
    let (audio, _) = backend.run(&ids, 1.0)?;
    let mut best = std::time::Duration::MAX;
    for i in 0..5 {
        let t = std::time::Instant::now();
        let (a, _) = backend.run(&ids, 1.0)?;
        best = best.min(t.elapsed());
        let _ = a.len();
        eprintln!("after run {i}: peak RSS {:.0} MB", rss_mb());
    }
    println!(
        "run({n} ids) -> {} samples ({:.2}s audio) in {:.0} ms | {:.0} ms per audio-second | RTF {:.3}",
        audio.len(),
        audio.len() as f32 / backend.config().sample_rate as f32,
        best.as_millis(),
        best.as_secs_f32() / (audio.len() as f32 / backend.config().sample_rate as f32) * 1000.0,
        best.as_secs_f32() / (audio.len() as f32 / backend.config().sample_rate as f32),
    );
    Ok(())
}
