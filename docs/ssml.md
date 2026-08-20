# SSML support in floravox

floravox parses SSML locally (no network, no cloud dialect). Every word,
break, and mark keeps byte- and character-exact spans pointing back into
your original string, so client-side highlighting lines up with the
source text even when tags shift offsets.

This document describes what the parser actually accepts, what each tag
does during synthesis, and the edge cases. It reflects the code, not the
W3C spec: where the two differ, the difference is called out.

## Input shapes

Plain text is accepted and treated as one utterance:

```text
Hello world
```

SSML is detected by a leading `<speak>` (a full envelope is optional —
floravox does not require `version` or `xmlns` attributes, unlike some
cloud endpoints):

```xml
<speak>Hello <break time="250ms"/>world</speak>
```

XML entities (`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`, numeric
references) are decoded, and the decoded text occupies the span the
entity occupied in the source.

Malformed input never fails the whole document: the parser records a
warning, does its best with the rest, and exposes warnings on the
document (`floravox timeline` prints them; the synthesizer ignores them).

## Element reference

### `<break time="250ms"/>`

Silence spliced into the audio at an exact sample position.

- `time`: `120ms`, `1.5s`, `250` (bare number = milliseconds).
- `strength` (used only when `time` is absent): `none` 0 ms,
  `x-weak` 50, `weak` 100, `medium` 250, `strong` 500, `x-strong` 800.
- Breaks are emitted as `BreakStarted`/`BreakEnded` events at their
  exact sample offsets; a `<mark>` immediately before a break fires at
  the break's start.

### `<mark name="m1"/>`

A positional event, not audio. When the audio reaches the mark's sample,
a `MarkReached` event fires carrying the name. Marks placed:

- before a word → fire at that word's first sample,
- between words → fire at the next word's start,
- before a `<break>` → fire at the break's start,
- at the very end (no following speech) → fire at the current offset.

This is the primitive screen readers need for SSIP index marks, and it
works identically on measured and estimated timing paths.

### `<prosody rate="..." pitch="..." volume="...">`

Scoped multiplier applied to enclosed text; nesting overlays (inner
values win per attribute).

- `rate`: `0.8` (20% slower), `+10%`, `-15%`, or names
  `x-slow` (0.5) `slow` (0.75) `medium` (1.0) `fast` (1.25)
  `x-fast` (1.5).
  Rate changes split synthesis into separate inference passes, so
  timing stays measured and exact at every rate.
- `pitch`: accepted and parsed (`+3st`, `10%`, names), but VITS-family
  voices have no pitch control — the value is recorded on the word spans
  for downstream engines and ignored by the current backends.
- `volume`: parsed (`+2dB`, `0.9`, `loud`) and applied by the audio
  path in the wrapper layer (linear scaling); the core emits unscaled
  samples.

Unparseable values are ignored with a warning, not an error.

### `<emphasis level="strong|moderate|reduced">`

Convenience wrapper implemented as a prosody preset:

| level | rate | pitch | volume |
|---|---|---|---|
| strong | 0.97 | 1.15 | 1.20 |
| moderate | 0.99 | 1.06 | 1.10 |
| reduced | 0.92 | 0.90 | 0.80 |

(pitch again recorded but not rendered by VITS-family backends.)

### `<say-as interpret-as="...">`

- `characters` / `spell-out`: the enclosed text is spelled out — each
  character is phonemized separately (e.g. `abc` → "ay bee see"). This
  is the fully supported mode and the one AAC spelling boards use.
- `cardinal` / `number`, `ordinal`: recorded on the word spans. With the
  misaki English frontend, numbers are expanded to words before
  synthesis regardless, so `42` is spoken as "forty two" naturally; the
  attributes mark intent for frontends that need it.
- anything else (`date`, `time`, `currency`, `telephone`): recorded,
  no current behavioral change — expansion is left to the input text
  or a document frontend.

### `<phoneme ph="h ə l oʊ">hello</phoneme>`

Bypasses G2P entirely for the enclosed word: the `ph` symbols are used
directly (whitespace-separated, resolved against the voice's own
inventory through floravox's symbol resolution — composed symbols like
`oʊ` are split, homoglyphs corrected, unknown symbols dropped with the
rest intact).

This is the escape hatch for forcing a pronunciation, and the primary
way to drive voices whose G2P you don't have (e.g. feeding kokoro's
character alphabet before the misaki frontend existed).

### `<sub alias="World Wide Web">WWW</sub>`

The enclosed text is *replaced* by the alias for pronunciation, while
word events still carry the original text and its spans — highlighting
shows `WWW` while the voice says "World Wide Web".

### `<voice name="...">`

Recorded on word spans (per-word `voice` field). The synthesizer does
not switch voices mid-utterance; the attribute is surfaced for wrapper
layers that route utterances by voice.

### `<speak xml:lang="...">`

The envelope's `xml:lang` is recorded on the document and used for G2P
routing by consumers (the rust-tts-wrapper floravox engine picks the
voicegarden-lexicons bundle for that language when no explicit language
is configured). It does not switch voices mid-document and is not
validated beyond UTF-8.

### `<s>`, `<p>`

Sentence and paragraph boundaries. Each emits a positional event, and —
since the streaming change — sentence-final punctuation in plain text
also splits synthesis passes, so audio starts flowing per sentence
whether or not you tag them.

### `<audio src="...">`

Ignored with a warning (pre-recorded audio splicing is out of scope for
the core; the wrapper layer is the right place if ever needed).

## What is NOT supported

- `<audio>` playback (parsed, warned, ignored).
- `<lexicon>` references — pronunciation dictionaries are floravox FST
  lexicons supplied at engine setup, not per-document URIs.
- `<meta>`, `<metadata>`, SSML `xml:lang` switching mid-document.
- W3C `morphology`/`token`/`w` fine-grained token elements.

Unknown tags are parsed as transparent containers (their text still
renders); unknown attributes are ignored silently. This keeps documents
authored for cloud engines loadable rather than rejected.

## Timing notes

- Word and mark positions are **measured** (derived from the acoustic
  model's duration tensor) on patched voices; events carry
  `estimated: false`. On stock voices a proportional estimator is used
  and events carry `estimated: true` with identical structure.
- `<break>` splicing happens at the sample level, so a break never
  shifts measured word boundaries that follow it — they are re-based
  by exactly the break's length.
- With rate ≠ 1 the durations tensor reflects the actual rendered rate,
  so measured timings remain exact at any prosody rate.
