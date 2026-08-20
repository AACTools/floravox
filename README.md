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
| `floravox-g2p`   | mmap'd FST lexicons, LRU cache, OOV fallback trait, ingest |
| `floravox-core`  | ort synthesis, duration folding, `EventTimeline`, events   |
| `floravox-cli`   | `floravox synth` / `floravox timeline` diagnostics         |

## Building a lexicon

`floravox-fst-compile` ingests three source formats (auto-detected, or
pinned with `--format`):

| Format     | Shape                                    | Sources                          |
|------------|------------------------------------------|----------------------------------|
| `cmudict`  | `WORD  P HH R AH1 N`                     | CMUDict (ARPABET → IPA)          |
| `ipa-tsv`  | `word\thəˈloʊ` (unsegmented IPA)         | WikiPron downloads, gruut dumps  |
| `tsv`      | `word\tph1 ph2 ph3` (pre-segmented)      | hand-maintained lists            |

```console
# CMUDict (BSD-style, cmusphinx distribution):
curl -L -o cmudict.dict \
  https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
cargo run -p floravox-g2p --bin floravox-fst-compile -- cmudict.dict en_US

# Then:
cargo run -p floravox-cli -- synth --model voice.onnx --lexicon en_US \
  --text '<speak>Hello world</speak>' --out out.wav --events events.json
```

ARPABET conversion targets the piper/espeak en_US inventory: `AH0`/`ER0`
reduce to `ə`/`ɚ`, stress digits become standalone `ˈ`/`ˌ` symbols, and
`CH`/`JH` map to `tʃ`/`dʒ`.

## OOV: ByT5 neural fallback

Words missing from the lexicon can be phonemized by a ByT5 G2P model
(byte-level T5 — no tokenizer to keep in sync). Export a Hugging Face
checkpoint with optimum and pass the pair to `synth`:

```console
optimum-cli export onnx --model <hf-byt5-g2p-checkpoint> byt5/
cargo run -p floravox-cli -- synth --model voice.onnx --lexicon en_US \
  --byt5-encoder byt5/encoder_model.onnx --byt5-decoder byt5/decoder_model.onnx \
  --text '<speak>Hello world</speak>' --out out.wav --events events.json
```

Decoding is greedy with an EOS stop; the engine implements
`floravox_g2p::OovFallback` and chains down to letter-name spelling when
it produces nothing (`floravox_g2p::ChainedFallback`).

## Quick start

```console
# Patch a voice (once):
pip install -r python/requirements.txt
python python/add_durations_output.py voice.onnx --validate

# Synthesize with events:
cargo run -p floravox-cli -- synth \
  --model voice.onnx --lexicon en_US \
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
- [x] Lexicon data ingestion: CMUDict / IPA-TSV (WikiPron, gruut) converters
- [x] Duration graph surgery + validation (`python/`)
- [x] ort synthesis: measured word/mark timings, break splicing, estimation
      fallback, adaptive input styles (`scales` and split inputs)
- [x] End-to-end verified against `en_US-lessac-low` (16 kHz)
- [x] ByT5 ONNX OOV fallback engine (`floravox-g2p` `onnx` feature,
      greedy byte-level decoding, `ChainedFallback` to spelling)
- [ ] Phonetisaurus WFST OOV fallback
- [ ] rust-tts-wrapper engine adapter
- [ ] VoiceGarden-SPD module integration

## Licensing

Code: Apache-2.0 OR MIT. Model weights and lexicon data keep their own
licenses — see `docs/licensing.md` (note: WikiPron-derived data is
CC BY-SA 4.0, which travels with the data, not the code).
