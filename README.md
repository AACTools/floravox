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

## Supported voice families

| Family | Layout | Measured timings | Notes |
|---|---|---|---|
| piper VITS | `X.onnx` + `X.onnx.json` | yes (patched) | the original target |
| MMS VITS | `X.onnx` + `tokens.txt` (+ `config.json`) | yes (patched) | tensor names + config auto-detected |
| Matcha | acoustic `*.onnx` + `tokens.txt` + vocoder (`hifigan*`/`vocos*`) | yes (patched acoustic) | `sum(durations) == mel frames`, audio via vocoder |
| kokoro / zipvoice | — | — | not wired (kokoro's per-token durations are tappable; zipvoice is prompt-conditioned flow matching) |

Family, tensor naming, sample rate, and hop are discovered at load time
(graph inputs, sibling `tokens.txt`/`config.json`, embedded ONNX metadata,
or a piper-style `.onnx.json` sidecar).

## How the timing works

Stock piper/MMS VITS and Matcha ONNX voices compute phoneme durations
internally but discard them. `python/add_durations_output.py` performs
graph surgery on any exported model — no PyTorch, no checkpoints —
tapping the duration predictor's Ceil tensor into a stable
`"durations"` output:

```
sum(durations) × hop_length == audio samples   (VITS, validated exactly)
sum(durations) == mel frames                   (Matcha, audio via vocoder)
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

## OOV: Phonetisaurus WFST fallback

For OOV duty without `ort`, a Phonetisaurus n-gram transducer gives
lexicon-quality pronunciations from a few MB of weights. The decoder is a
clean-room Rust implementation of the OpenFst container format plus a
shortest-path search — no GPL code, no native dependencies, available in
frontend-only builds:

```console
# Any phonetisaurus model works; the cmudict downloads embed their
# symbol tables in the .fst itself:
cargo run -p floravox-cli -- g2p --phonetisaurus cmudict-20170708.o8.fst \
  hello world floravox
#   hello    HH EH1 L OW0
#   world    W ER1 L D
#   floravox F L AO1 R AH0 V AA0 K S

# Or wired into synthesis (tries phonetisaurus, then ByT5, then spelling):
cargo run -p floravox-cli -- synth --model voice.onnx --lexicon en_US \
  --phonetisaurus cmudict-20170708.o8.fst \
  --text '<speak>Hello world</speak>' --out out.wav --events events.json
```

Both on-disk layouts load: embedded symbol tables (single `.fst`) or
`model.fst` + `model.grapheme.table` + `model.phoneme.table`. Phonetisaurus
16-byte arcs and stock OpenFst 20-byte arcs are auto-detected, compound
symbols (`a|c` graphemes, `AH0|N` phonemes) are handled, and input casing
is inferred from the grapheme table. Note the output alphabet is whatever
the model was trained on (ARPABET for CMUDict models) — pair the model
with a voice that speaks that alphabet.

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
- [x] Phonetisaurus WFST OOV fallback (clean-room OpenFst reader +
      shortest-path decode, no `ort` dependency; validated against a
      1M-state CMUDict model)
- [x] Multi-family voices: `VoiceBackend` trait with piper/MMS VITS and
      Matcha+vocoder backends (family, tensor names, config auto-detected;
      both validated live with measured timings)
- [ ] rust-tts-wrapper engine adapter (`floravox-engine` branch)
- [ ] VoiceGarden-SPD module integration

## Licensing

Code: Apache-2.0 OR MIT. Model weights and lexicon data keep their own
licenses — see `docs/licensing.md` (note: WikiPron-derived data is
CC BY-SA 4.0, which travels with the data, not the code).
