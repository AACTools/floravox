# VoiceGarden-SPD integration

Goal: `voicegarden-spd` speaks with real word boundaries and fires SSIP
index marks (`700-SSIP EVENT INDEX-MARK`) at the exact audio sample, so
Orca-style clients get in-sync highlighting without IPC estimation.

## Wiring

floravox is a plain crate dependency — no FFI, no subprocess:

```toml
# VoiceGarden-SPD/crates/voicegarden-spd/Cargo.toml
[dependencies]
floravox-core = { path = "../../floravox/crates/floravox-core" }
```

## Playback loop with sample-accurate events

The module owns the audio device. Keep a **playback cursor** = number of
samples actually written to ALSA/Pulse/PipeWire, and drain the timeline as
you go:

```rust
use floravox_core::{EventTimeline, SynthesisEvent};

let mut timeline = EventTimeline::new();
let mut pcm_cursor: u64 = 0;

loop {
    let chunk = stream.audio.recv()?; // AudioChunk { samples, first_sample, .. }
    device.write(&chunk.samples)?;
    pcm_cursor = chunk.first_sample + chunk.samples.len() as u64;

    for event in timeline.drain_until(pcm_cursor) {
        match event {
            SynthesisEvent::WordBoundary(w) => {
                // word highlight: w.char_offset / w.char_len index the
                // original SSML the client sent
            }
            SynthesisEvent::MarkReached { name, .. } => {
                // speech-dispatcher C API:
                // module_report_event_index_mark(name.as_ptr());
            }
            SynthesisEvent::BreakStarted { .. } => { /* pause visual */ }
            SynthesisEvent::Finished { .. } => { /* stop iteration */ }
            _ => {}
        }
    }
}
```

Events arrive from a worker thread while audio streams; the timeline
orders them by sample so the drain never fires early.

## Notes

- Synthesis runs one utterance at a time behind a mutex (SSIP SPEAK is
  serial anyway). A second SPEAK while streaming should `stop()` first,
  which is `drop()` of both receivers.
- `length_scale` (speaking rate) is applied per prosody run, and the
  durations tensor already reflects it — timings stay correct at any rate.
- Unpatched voices (no `durations` output) still speak; events carry
  `estimated: true` so the module can decide whether to forward them.
