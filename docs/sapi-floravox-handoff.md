# VoiceGarden-SAPI: enabling the floravox engine — implementation guide

**Status: internal handoff doc — do not commit.**
**Audience: Windows team.** Everything below is verified working on Linux
SPD; the SAPI side is the same wrapper, so most of the work is wiring, not
new engineering.

---

## Background (2 minutes)

floravox is our offline TTS engine ([github.com/AACTools/floravox](https://github.com/AACTools/floravox),
Apache-2.0 OR MIT, pure Rust, no espeak-ng). It drives the same voice
models sherpa-onnx does (piper/MMS VITS, Matcha+Hifi-GAN, Kokoro) but adds:

- **Measured word timings** — from the acoustic model's duration tensor
  (voices patched with `floravox/python/add_durations_output.py`), not
  150-wpm estimates. This is the whole point: SAPI bookmark/bookmarkEvent
  and word-boundary events can track the actual audio.
- **Native SSML** — `<break>`, `<prosody>`, `<mark>`, `<phoneme>`,
  `<say-as>`, `<sub>` parsed locally. SAPI SSML can be passed through
  instead of stripped.
- **SpeechMarkdown** — converted in the wrapper's `speak()` path.
- **Streaming** — sentence-level inference passes; audio for sentence N
  is delivered while N+1 synthesises.

rust-tts-wrapper v0.4.1 ships the engine behind the `floravox` feature
(`floravox-lexicons` adds language routing). VoiceGarden-SPD has already
made floravox its default local engine (v0.4.0, PR #6 there).

**Per-model routing**: the registry crate (`sherpa-onnx-models` on
crates.io, from AACTools/sherpa-onnx-tts-models) now carries an `engines`
field on every model: `"floravox"` = drivable (vits/mms/matcha/kokoro,
1,746 models), `"sherpa-onnx"` = audio-LM families (kitten/pocket/
supertonic/zipvoice, 14 models). Route on that field.

---

## What to do on SAPI

### 1. Disable sherpa-onnx in SAPI (this is now the only safe option)

Agreed approach: **drop the `sherpaonnx` feature entirely** and let
floravox be the only local engine — exactly what VoiceGarden-SPD v0.4.0
shipped.

**Do not enable both and assume it links.** Update from the SPD v0.4.0
release: Windows sherpa downloads a *static* archive
(`win-x64-static-MT-Release-lib`) that embeds the whole onnxruntime, and
so does ort — enabling both features produced "multiple definition of
`onnxruntime::…`" at final link and killed the SPD aarch64 release.
An earlier note in this doc claimed the two "link cleanly together";
that verification was a **library-only build**, which never performs the
final link. On Windows MSVC the same collision is expected. If you must
keep sherpa for something (e.g. zipvoice cloning), either switch sherpa
to its `shared` link mode and ship one `onnxruntime.dll`, or build
separate binaries — but the clean path is dropping it and routing the
14 audio-LM models out of the catalogue (`engines` field says which).

### 2. Wrapper dependency

```toml
rust-tts-wrapper = { git = "https://github.com/AACTools/rust-tts-wrapper", tag = "v0.4.1", default-features = false, features = [
    "floravox-lexicons",   # floravox engine + published lexicon bundles
    # "cloud", ...         # whatever SAPI already enables
    # "sherpaonnx",        # drop this (or keep — see above)
] }
```

Note: `floravox-lexicons` implies `floravox`. The `speechmarkdown` feature
comes in via `cloud` as usual; floravox accepts the generic SSML dialect
it emits (whisper's `<amazon:effect>` is normalised to `<prosody>` inside
the wrapper's floravox engine).

### 3. Engine credentials

```json
{
  "modelsDir": "C:\\Users\\<user>\\AppData\\Local\\VoiceGarden\\models",
  "modelId": "piper-en_US-lessac-high",
  "lang": "en"
}
```

- `modelsDir`: a directory holding one sub-directory per voice (same
  layout VoiceGarden already downloads — `X.onnx` + `X.onnx.json`, or
  Matcha acoustic + `hifigan*`/`vocos*` vocoder, or Kokoro `model.onnx`
  + `tokens.txt` + `voices.bin`). Family/layout is auto-detected at load.
- `lang`: BCP-47 or ISO code. With `floravox-lexicons`, first use fetches
  the published bundle for that language (gruut-derived MIT lexicon +
  trained Phonetisaurus model, ~2–15 MB, cached under
  `%USERPROFILE%\.voicegarden\lexicons`). English voices also get misaki
  (Kokoro's own phonemizer) automatically; MMS-style character voices can
  take `"chars": "true"` instead of `lang`.
- `numThreads`: string, default 2 is fine (the engine defaults to 2
  intra-op threads; more grows memory without helping small models).

### 4. Voice enumeration

`FloravoxEngine::get_voices()` lists every voice directory under
`modelsDir` (id, display name, language, sample rate read from each
`X.onnx.json`). For SAPI's full catalogue you **must** use the registry
crate — without the `sherpaonnx` feature there is no `SherpaOnnxEngine`
to ask (SPD hit this and switched to the crate in v0.4.0):

```rust
let drivable = sherpa_onnx_models::models()
    .values()
    .filter(|m| m.engines == "floravox")
    .collect::<Vec<_>>();   // 1,746 download candidates
```

`sherpa-onnx-models = "0.1"` (crates.io) — typed `ModelInfo` with
`engines`, languages, license, url; data embedded at compile time.
Only registry-listed voices should be offered for download; local
`get_voices()` covers installed ones.

### 5. SAPI plumbing — the parts that pay off

- **Word boundaries**: the engine's `on_boundary` callback
  (`word, start_sec, end_sec, char_offset, char_len`) carries *measured*
  timings. Map to `ISpVoice::Word` events (SPVEI1 word boundary) as SAPI
  does today, but the positions now track the audio exactly, including
  through `<prosody rate>` changes (durations reflect the applied rate —
  no rescaling needed).
- **Bookmarks**: SAPI SSML `<MARK NAME="x"/>` maps 1:1 to floravox
  `<mark name="x"/>`; the engine emits a `MarkReached` event at the
  measured sample. If SAPI currently strips SSML before synthesis,
  stop: pass it through (minus marks if you time them yourself, SPD-style).
- **Voices patched with `add_durations_output.py` give measured timings;
  unpatched voices still work** with estimated timings (flagged
  `estimated: true` in the wrapper's boundary metadata). Recommend
  patching on download — it's a one-shot file transform, license-neutral
  (see floravox README).
- **Streaming**: `speak()` already streams chunks via `on_audio` (PCM16
  LE mono). SPD's measured numbers: first chunk ~2 s cold (model load),
  ~75% of synthesis overlaps playback on multi-sentence text.

### 6. Fallback (optional, recommended)

SPD attaches a sherpa fallback for load failures. If SAPI keeps
sherpa-onnx enabled, mirror `voicegarden-spd`'s pattern (PR #6, files
`pipeline.rs`/`voices.rs`): retry through the other engine *only when no
audio has flowed yet* (never double-speak). If SAPI drops sherpa
entirely, skip this — an engine load failure should surface as the SAPI
error path.

---

## Windows-specific checks (do these first)

0. **32-bit builds — floravox cannot cover them.** ort ships Windows
   artifacts for `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`
   only; there is **no i686 artifact** (this is why the wrapper pins
   sherpa's exact version for its own i686 publish job). If SAPI still
   ships a win32 flavour, keep sherpa for that target alone or drop the
   32-bit flavour — floravox will not build there.
1. **ort/onnxruntime on Windows**: floravox links onnxruntime via the
   `ort` crate, which downloads prebuilt MSVC artifacts at build time
   (DirectML builds available). Wrapper CI already builds `floravox`
   on `windows-latest` green, so the toolchain is fine; what needs eyes
   is SAPI's packaging and a **final-link** build of the actual SAPI
   binaries (lib-only builds hide link errors — see §1).
2. **ARM64 (Surface/others)**: artifacts exist for aarch64 Windows; run
   one Kokoro + one piper voice on ARM64 hardware before shipping.
3. **Lexicon fetch path**: `voicegarden-lexicons` caches under
   `%USERPROFILE%\.voicegarden\lexicons` (override with
   `VOICEGARDEN_LEXICON_DIR`). Confirm the ureq/rustls TLS stack is happy
   with your build (it's pure-Rust TLS; no schannel dependency).
4. **Voice paths on Windows**: credentials accept `C:\...` paths; the
   engine also honours `modelsDir` + `modelId` separately. Watch for
   `\` escaping in JSON credentials.

## Lessons from the SPD v0.4.0 release (same wrapper, same engine)

Three failures hit that release; here is the Windows relevance of each:

| SPD failure | Windows? |
|---|---|
| sherpa's static archive embeds onnxruntime → duplicate symbols vs ort at **final link** | **YES — expected.** Windows sherpa is static by default. Drop the feature (§1). |
| Missing C++ runtime linking once sherpa's build script was gone | No — MSVC CRT is ambient in the SAPI toolchain |
| ort artifacts built against glibc 2.38 / GCC 13 libstdc++ | No analog (MSVC artifacts both sides) — but see check 0: artifact *availability* per target is the real constraint |

Also from SPD: construct no `SherpaOnnxEngine` for registry lookups (§4),
and floravox became the default engine there with stable voice names
across the flip — saved selections survived. Worth mirroring both.

---

## Quick reference

| Thing | Where |
|---|---|
| floravox engine source | `rust-tts-wrapper/src/floravox_engine.rs` (v0.4.1) |
| SPD reference integration (PRs #5, #6) | `VoiceGarden-SPD/crates/voicegarden-spd/src/{voices,pipeline,callbacks}.rs` |
| Registry crate with `engines` field | `sherpa-onnx-models` 0.1.0 on crates.io |
| Lexicon bundles | github.com/AACTools/voicegarden-lexicons (14 languages) |
| Duration patching tool | `floravox/python/add_durations_output.py` |
| floravox SSML reference | `floravox/docs/ssml.md` |
| Wrapper examples (Rust) | `rust-tts-wrapper/examples/floravox-demo.rs`, `floravox-stream-demo.rs` |

## Suggested acceptance test

One piper voice (e.g. `piper-en_US-lessac-high`), patched: speak an SSML
document containing a bookmark mid-sentence, assert the SAPI bookmark
event fires within ~50 ms of the audio actually reaching that word (SPD
achieves mid-stream index marks at measured positions; the same
tolerance is realistic here). Then the same with `<prosody rate="0.5">`
wrapped around the bookmark — timing must stay correct without any
rescaling logic.
