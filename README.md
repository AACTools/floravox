# floravox

floravox is a text-to-speech engine written in Rust. You give it text and a voice model; it gives you audio plus the exact moment each word starts and ends.

It was built for the [VoiceGarden](https://github.com/AACTools) assistive-speech products, and for anyone who needs offline TTS with precise timing. The code is dual-licensed Apache-2.0 OR MIT. It is pure Rust, needs no Python runtime, and links no GPL code.

## The problem it solves

The good offline voices (piper, MMS, Matcha, Kokoro) ship as ONNX model files that take phonemes as input, not text. To build a product on top of them you need three extra pieces, and writing them yourself is months of work:

1. Text to phonemes (G2P). English is built in. German and twelve other languages work with bundles from [voicegarden-lexicons](https://github.com/AACTools/voicegarden-lexicons).
2. SSML. Screen readers and AAC apps send `<break>`, `<prosody rate>`, `<mark>` and friends. floravox parses them and tracks byte and character positions that point back into your original text.
3. Word timings that are measured, not guessed. Reading apps highlight the word being spoken as it plays. Most engines estimate timings from word length, and those estimates land roughly 100 to 300 ms off, which listeners notice. floravox uses the real numbers.

## How the timings work

Every supported model already computes how long each phoneme lasts, then throws that information away. `python/add_durations_output.py` patches the model file so the durations become a normal output. After patching:

```
sum(durations) × hop_length == audio samples   (piper and MMS VITS)
sum(durations) == mel frames                   (Matcha, audio comes from a vocoder)
sum(durations) × 600 == audio samples          (Kokoro, at 24 kHz)
```

floravox folds those phoneme durations back onto the words they belong to. Every word boundary and SSML mark then carries a sample-accurate position. Unpatched models still work; timings fall back to a proportional estimate and events are flagged `estimated: true`.

## Supported voices

| Family | Files on disk | Measured timings | Notes |
|---|---|---|---|
| piper VITS | `X.onnx` + `X.onnx.json` | yes, after patching | the original target |
| MMS VITS | `X.onnx` + `tokens.txt` (plus `config.json`) | yes, after patching | tensor names and config are found automatically |
| Matcha | acoustic `*.onnx` + `tokens.txt` + vocoder (`hifigan*` or `vocos*`) | yes, after patching the acoustic model | audio comes from the vocoder |
| Kokoro | `model.onnx` + `tokens.txt` + `voices.bin` | yes, after patching | 11 voices in en-v0.19, native speed control, `sum(durations) × 600 == samples` holds at every speed |
| zipvoice | not supported | no | it is a cloning model with no usable duration tensor |

You point `--model` at a directory or file and floravox works out which family it is, what the tensors are called, the sample rate, and the hop size. It reads the graph inputs, a sibling `tokens.txt` or `config.json`, ONNX metadata, or a piper-style `.onnx.json`, whichever is present.

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

Entries in `events.json` look like this:

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

If you push PCM to an output device yourself, call [`EventTimeline::drain_until`](crates/floravox-core/src/timeline.rs) with your playback cursor and it hands you every event that has come due. `docs/voicegarden-spd.md` shows the full recipe for speech-dispatcher index marks.

## Using it from other languages

All the crates are on crates.io at 0.5.1: `floravox-ssml`, `floravox-g2p`, `floravox-core`, `floravox-cli`.

If you do not write Rust, `floravox-capi` builds the G2P part as a C shared library (`libfloravox_capi.so`), with a header at `crates/floravox-capi/include/voicegarden-g2p.h`. That gives you phonemization from Python (ctypes), Node, C, C++, Java, C#, Go, and Dart without touching Rust.

| Crate | What it is |
|---|---|
| `floravox-ssml` | the SSML parser and span tracker |
| `floravox-g2p` | lexicons, the OOV fallback chain, and the ingest tools |
| `floravox-core` | voice backends, duration folding, the event timeline |
| `floravox-cli` | the `floravox synth` / `g2p` / `timeline` commands |
| `floravox-capi` | the C ABI shared library |

## G2P: turning words into phonemes

### English

English synthesis runs through [misaki](https://github.com/hexgrad/misaki), the same phonemizer Kokoro voices were trained with, via the Rust port [`misaki-rs`](https://crates.io/crates/misaki-rs). It is a default feature, MIT licensed, with dictionaries baked into the binary. Sentence context means heteronyms ("object" the noun versus the verb) and numbers ("123 dollars") come out right. Pick a dialect with `--misaki us` or `--misaki gb`.

```console
cargo run -p floravox-cli -- synth --model kokoro-model.onnx --misaki us \
  --text '<speak>Hello world</speak>' --out out.wav --events events.json
```

The crate's optional espeak fallback is switched off because it would link GPL code. Without it, unknown words are spelled out letter by letter, same as the built-in fallback.

### Other languages

Non-English voices use lexicon bundles from [voicegarden-lexicons](https://github.com/AACTools/voicegarden-lexicons). The important fact: gruut, the source of those lexicons, is the phonemizer piper's non-English voices were trained with, so its symbols line up with the voices. Tested on German: 236,000 symbols sampled against the piper German voice, none failed to resolve.

```console
python python/gruut2tsv.py lexicon.db de_DE.tsv
cargo run -p floravox-g2p --bin floravox-fst-compile -- de_DE.tsv de_DE
cargo run -p floravox-cli -- synth --model de_DE-thorsten.onnx \
  --lexicon de_DE --text '<speak>Guten Tag</speak>' \
  --out out.wav --events events.json
```

`floravox-fst-compile` also reads three raw formats directly (auto-detected, or pinned with `--format`):

| Format | Shape | Typical source |
|---|---|---|
| `cmudict` | `WORD  P HH R AH1 N` | CMUDict, converted from ARPABET to IPA |
| `ipa-tsv` | `word\thəˈloʊ` (IPA not yet split) | WikiPron downloads, gruut dumps |
| `tsv` | `word\tph1 ph2 ph3` (already split) | hand-maintained lists |

For CMUDict English the conversion targets the espeak-style inventory piper uses: `AH0` and `ER0` reduce to `ə` and `ɚ`, stress digits become standalone `ˈ` and `ˌ` marks, and `CH` and `JH` become `tʃ` and `dʒ`.

### Symbol resolution

Lexicons and G2P engines write composed symbols like `oʊ`, `aɪ`, and `ɝ`. Piper-style voices spell those as separate characters. floravox now resolves every symbol against the voice's own table (direct hit, then a substitution table, then splitting per character). This fixed a real bug: before, mismatched symbols were dropped silently, which deleted every diphthong, about 20% of symbols on a CMUDict sample. "night" lost its vowel. After the fix, 0% are dropped on the same test.

### Out-of-vocabulary words, two optional engines

Longer words the lexicon does not know can go through a Phonetisaurus WFST (a few MB, no ONNX runtime needed) or a ByT5 model (an ONNX pair, exported with optimum). Both plug into the same trait; whatever they cannot handle falls through to letter spelling.

```console
# Phonetisaurus, query mode:
cargo run -p floravox-cli -- g2p --phonetisaurus cmudict-20170708.o8.fst \
  hello world floravox
#   hello    HH EH1 L OW0
#   world    W ER1 L D
#   floravox F L AO1 R AH0 V AA0 K S
```

The Phonetisaurus decoder is a clean-room Rust implementation of the OpenFst file format plus a shortest-path search. Both layouts load (embedded symbol tables, or `model.fst` plus separate table files), and 16-byte and 20-byte arc encodings are detected automatically.

## How accurate is it?

Two audit tools ship in `python/`:

- `eval_timings.py` checks synthesized audio against the signal itself: breaks must land at silence edges, word boundaries must sit in energy dips, and it reports what the estimating fallback would have said instead. Results on patched voices: breaks land exactly at their silence edges, inter-word boundaries sit in dips, trailing silence tracks within about 5 ms, and the estimator would have been 94 to 308 ms off (median across families). One known wart: Kokoro emits roughly 640 ms of near-silence at the start that gets attributed to the first word, so highlight-from-start UIs should account for leading silence.
- `audit_g2p.py` compares G2P output against `espeak-ng --ipa`. Raw string comparison scores low (17% exact) but the differences are dialect, not errors: diphthong spelling and stress placement differ between dictionaries. The metric that matters is coverage of a real voice's symbol table, and that is 100% for both piper English (after symbol resolution) and Kokoro (after misaki normalization).

## Voice registry compatibility

Against the [sherpa-onnx-tts-models](https://github.com/AACTools/sherpa-onnx-tts-models) registry of 1,760 models: 1,138 MMS, 599 VITS, 5 Matcha, and 4 Kokoro models are drivable with measured timings. The remaining 14 (kitten, pocket, supertonic, zipvoice) are language-model or flow-matching models with no duration tensor; those stay on sherpa-onnx. Downloading and routing models is rust-tts-wrapper's job; floravox stays a synthesis engine that takes local files.

## Status

- [x] SSML parsing with exact spans (entities, `<sub>`, `<phoneme>`, marks)
- [x] FST lexicon format, compiler, and LRU cache
- [x] Lexicon ingestion: CMUDict, IPA-TSV, gruut
- [x] Duration graph surgery and validation for all supported families
- [x] Synthesis: measured word and mark timings, break splicing, estimation fallback, old and new piper input styles
- [x] Verified end to end against en_US-lessac, matcha-ljspeech, vits-mms-fra, kokoro-en-v0.19, and piper de_DE-thorsten
- [x] ByT5 ONNX OOV fallback (feature `onnx`)
- [x] Phonetisaurus WFST OOV fallback (no ort dependency, validated against a 1M-state CMUDict model)
- [x] Four voice families behind one `VoiceBackend` trait, detected at load time
- [x] English G2P: misaki pre-pass plus CMUDict and Phonetisaurus, 100% symbol coverage on both voice types
- [x] German G2P via gruut lexicons, 0.00% symbols dropped
- [x] CI on every push, plus a live voice-matrix workflow that downloads real models and checks measured events
- [x] Published on crates.io at 0.5.1
- [x] C ABI crate (`floravox-capi`) for non-Rust consumers
- [ ] More languages beyond gruut's thirteen (per-language Phonetisaurus WFSTs trained on the published lexicons; ByT5 where no lexicon exists)
- [ ] rust-tts-wrapper engine adapter (branch `floravox-engine` exists, tracks an older floravox and needs a bump)
- [ ] VoiceGarden-SPD module integration

## Licensing

Code is Apache-2.0 OR MIT. Model weights and lexicon data keep their own licenses; see `docs/licensing.md`. One note: data derived from WikiPron is CC BY-SA 4.0, which travels with the data, not the code.
