# floravox

Event-driven SSML text-to-speech engine for the VoiceGarden ecosystem —
permissively licensed (Apache-2.0 OR MIT), pure Rust, no Python runtime, no
GPL dependencies.

floravox gives neural TTS the two things edge deployments usually lack:

1. **First-class SSML** — `<break>`, `<prosody>`, `<phoneme>`, `<sub>`,
   `<mark>`, `<say-as>`, `<emphasis>` parsed with byte- and char-exact
   source-span tracking.
2. **Measured word & mark timing** — word boundaries and index marks carry
   sample-accurate positions derived from the acoustic model's own duration
   tensor, not estimates.

## How the timing works

Stock piper-family ONNX voices compute phoneme durations internally but
discard them. `python/add_durations_output.py` performs graph surgery on
any exported model — no PyTorch, no checkpoints — tapping the Ceil tensor
into a stable `"durations"` output:

```
sum(durations) × hop_length == audio samples   (validated exactly)
```

The Rust side folds those per-phoneme-id durations back onto word spans:

```
Δt_frame = hop / sample_rate          (e.g. 256/16000 = 16 ms)
word_start = Σ durations[ids < word_first_id] × hop
```

Unpatched (stock) models still work: timings fall back to a
proportional estimator and events are flagged `estimated: true`.

## Workspace

| Crate            | Purpose                                                   |
|------------------|-----------------------------------------------------------|
| `floravox-ssml`  | quick-xml parser, span tracker, tag handlers               |
| `floravox-g2p`   | mmap'd FST lexicons, LRU cache, OOV fallback trait         |
| `floravox-core`  | ort synthesis, duration folding, `EventTimeline`, events   |
| `floravox-cli`   | `floravox synth` / `floravox timeline` diagnostics         |

## Quick start

```console
# Patch a voice (once):
pip install -r python/requirements.txt
python python/add_durations_output.py voice.onnx --validate

# Synthesize with events:
cargo run -p floravox-cli -- synth \
  --model voice.onnx \
  --text '<speak>Hello <mark name="m1"/>world<break time="250ms"/>done</speak>' \
  --out out.wav --events events.json
```

`events.json` entries look like:

```json
{
  "type": "word_boundary",
  "text": "hello",
  "char_offset": 7, "char_len": 5,
  "byte_offset": 7, "byte_len": 5,
  "sample_start": 768, "sample_end": 4096,
  "ms_start": 48, "ms_end": 256,
  "estimated": false
}
```

Consumers that push PCM to an output device use
[`EventTimeline::drain_until`](crates/floravox-core/src/timeline.rs) with
their playback cursor; see `docs/voicegarden-spd.md` for the Speech
Dispatcher index-mark recipe.

## Current status

- [x] SSML parsing with exact spans (entities, `<sub>`, `<phoneme>`, marks)
- [x] FST lexicon format + compiler (`floravox-fst-compile`) + LRU cache
- [x] Duration graph surgery + validation (`python/`)
- [x] ort synthesis: measured word/mark timings, break splicing, estimation
      fallback, adaptive input styles (`scales` and split inputs)
- [x] End-to-end verified against `en_US-lessac-low` (16 kHz)
- [ ] Real lexicon data ingestion (CMUDict / gruut extraction)
- [ ] ByT5 / Phonetisaurus OOV fallback engines
- [ ] rust-tts-wrapper engine adapter
- [ ] VoiceGarden-SPD module integration

## Licensing

Code: Apache-2.0 OR MIT. Model weights and lexicon data keep their own
licenses — see `docs/licensing.md` (note: WikiPron-derived data is
CC BY-SA 4.0, which travels with the data, not the code).
