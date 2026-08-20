# Licensing

## Code

All floravox crates are dual-licensed Apache-2.0 OR MIT. Dependencies:

| Dependency    | License              | Notes                          |
|---------------|----------------------|--------------------------------|
| quick-xml     | MIT                  |                                |
| fst           | MIT (Apache-2.0)     |                                |
| memmap2       | MIT OR Apache-2.0    |                                |
| ort           | MIT OR Apache-2.0    | links ONNX Runtime (MIT)       |
| serde/anyhow  | MIT OR Apache-2.0    |                                |
| misaki-rs     | MIT                  | default-features = false — its |
| uroman tables | Apache-2.0           | vendored from isi-nlp/uroman   |
|               |                      | (data/, + their LICENSE.txt);  |
|               |                      | a Rust reimplementation, no    |
|               |                      | uroman code is linked          |
|               |                      | optional espeak fallback (GPL) |
|               |                      | is NOT enabled                 |

No GPL/LGPL code is linked — the espeak-ng licensing problem that
motivated gruut does not arise here. (misaki-rs's pronunciation
dictionary includes corrections derived from espeak output; that is
data provenance, the same class as kokoro's own training data, and the
crate links no espeak code with the fallback feature off.)

## Data artifacts (each carries its own license)

| Artifact                  | Source license     | Travels with            |
|---------------------------|--------------------|-------------------------|
| CMUDict-derived lexicons  | BSD-2-Clause-ish   | the `.fst`/`.pho` pair  |
| WikiPron-derived lexicons | **CC BY-SA 4.0**   | the `.fst`/`.pho` pair  |
| gruut lexicon extractions | MIT                | compiled artifacts      |
| Piper voice models        | per-voice (check the voice repo) | `.onnx`/`.onnx.json` |

**WikiPron caveat:** CC BY-SA is a data license, not code — compiled FSTs
derived from WikiPron data remain CC BY-SA and must be distributed with
attribution (and share-alike terms for the data). They do not infect the
Rust code that reads them, but ship them as separate downloadable
artifacts with their own NOTICE rather than embedding them in binaries.

## Model patching

`add_durations_output.py` only rewrites ONNX graphs; the patched model
keeps the original voice license. Patched files should be named
distinctly (e.g. `voice.onnx` → keep hash) so provenance is traceable.
