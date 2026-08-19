//! # floravox-ssml
//!
//! Streaming SSML / plain-text parser with byte- and char-exact source span
//! tracking, built on `quick-xml`.
//!
//! Every word token emitted by [`parse`] carries its exact offsets into the
//! raw input string, so downstream consumers (highlighting, index marks,
//! lipsync) can map audio timing back onto original text — even when the input
//! contains XML entities (`&amp;`), `<sub alias>` replacements, or
//! `<phoneme>` overrides.
//!
//! ```
//! use floravox_ssml::{parse, Segment};
//!
//! let doc = parse("<speak>Hello <mark name=\"m1\"/>world</speak>").unwrap();
//! assert!(matches!(doc.segments[1], Segment::Mark { ref name, .. } if name == "m1"));
//! ```

use std::fmt;
use std::ops::Range;

/// Effective prosody for a word or segment, as resolved multipliers
/// (1.0 = unchanged). `None` fields inherit the engine default.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Prosody {
    /// Speaking-rate multiplier (0.8 = 20% slower).
    pub rate: Option<f32>,
    /// Pitch multiplier (1.05 = +5%).
    pub pitch: Option<f32>,
    /// Volume multiplier (1.2 = louder).
    pub volume: Option<f32>,
}

impl Prosody {
    /// Merge `other` on top of `self` (non-`None` fields win).
    fn overlay(self, other: Self) -> Self {
        Self {
            rate: other.rate.or(self.rate),
            pitch: other.pitch.or(self.pitch),
            volume: other.volume.or(self.volume),
        }
    }

    /// True when every field is `None` (engine defaults).
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.rate.is_none() && self.pitch.is_none() && self.volume.is_none()
    }
}

/// How a word should be spoken, from `<say-as interpret-as="...">`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SayAs {
    /// No `<say-as>` in scope.
    #[default]
    None,
    /// Spell out character by character (`interpret-as="characters"`).
    Characters,
    /// Read as cardinal number (`interpret-as="cardinal"` / `"number"`).
    Cardinal,
    /// Read as ordinal number (`interpret-as="ordinal"`).
    Ordinal,
    /// Date / time / currency / telephone — recorded for future expansion.
    Other,
}

/// A word token with exact source mapping.
///
/// Byte and char offsets index into the **raw input string** (`char_end` /
/// `byte_end` exclusive). When a word contains XML entities the byte span
/// covers the full raw source of the decoded text.
#[derive(Debug, Clone, PartialEq)]
pub struct WordSpan {
    /// The word as written (highlight/display target; e.g. `"WWW"`).
    pub text: String,
    /// The text that should be spoken (after `<sub alias>` substitution;
    /// equals `text` unless inside a `<sub>`).
    pub spoken: String,
    /// Character range in the raw input.
    pub char_span: Range<usize>,
    /// Byte range in the raw input (UTF-8).
    pub byte_span: Range<usize>,
    /// Explicit phoneme override from `<phoneme ph="f ə n ɛ t ɪ k s">`.
    /// Each element is one IPA symbol.
    pub phonemes: Option<Vec<String>>,
    /// Effective prosody snapshot for this word.
    pub prosody: Prosody,
    /// Effective `<say-as>` mode for this word.
    pub say_as: SayAs,
    /// Nearest enclosing `<voice name="...">`, if any.
    pub voice: Option<String>,
}

impl WordSpan {
    /// Length in characters of the raw source span.
    #[must_use]
    pub fn char_len(&self) -> usize {
        self.char_span.end - self.char_span.start
    }

    /// Length in bytes of the raw source span.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.byte_span.end - self.byte_span.start
    }
}

/// A flat, ordered representation of the input after SSML resolution.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SsmlDocument {
    /// Ordered segments (words, breaks, marks, structural markers).
    pub segments: Vec<Segment>,
    /// Non-fatal issues encountered while parsing (unknown tags, malformed
    /// values, ...). Input is always parsed leniently.
    pub warnings: Vec<String>,
}

impl SsmlDocument {
    /// All word spans in document order.
    #[must_use]
    pub fn words(&self) -> Vec<&WordSpan> {
        self.segments
            .iter()
            .filter_map(|s| match s {
                Segment::Words { words } => Some(words.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// The text as it will be spoken (sub aliases applied, tags stripped).
    #[must_use]
    pub fn spoken_text(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for seg in &self.segments {
            if let Segment::Words { words } = seg {
                parts.extend(words.iter().map(|w| w.spoken.clone()));
            }
        }
        parts.join(" ")
    }
}

/// One element of the flattened document.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    /// A run of words sharing one prosody context.
    Words {
        /// Word tokens with source spans.
        words: Vec<WordSpan>,
    },
    /// An explicit `<break time="500ms"/>`.
    Break {
        /// Pause length in milliseconds.
        ms: u32,
        /// Character position of the tag's `<` in the raw input.
        char_pos: usize,
        /// Byte position of the tag's `<` in the raw input.
        byte_pos: usize,
    },
    /// An SSML `<mark name="..."/>`; the engine emits an index-mark event at
    /// the exact sample where speech reaches this point.
    Mark {
        /// Mark name as given by the client.
        name: String,
        /// Character position of the tag's `<` in the raw input.
        char_pos: usize,
        /// Byte position of the tag's `<` in the raw input.
        byte_pos: usize,
    },
    /// A sentence boundary (`</s>`), usable for streaming segmentation.
    SentenceEnd {
        /// Character position of the tag in the raw input.
        char_pos: usize,
        /// Byte position of the tag in the raw input.
        byte_pos: usize,
    },
    /// A paragraph boundary (`</p>`).
    ParagraphEnd {
        /// Character position of the tag in the raw input.
        char_pos: usize,
        /// Byte position of the tag in the raw input.
        byte_pos: usize,
    },
}

/// Error type: parsing is lenient, so failures are near-impossible by design.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SSML parse error: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

/// Parse SSML or plain text into an [`SsmlDocument`].
///
/// Plain text (nothing that looks like markup) is treated as a single text
/// run. Input that looks like XML but fails to parse falls back to plain-text
/// treatment with a warning: a TTS frontend should always speak *something*.
#[allow(clippy::missing_errors_doc)]
pub fn parse(input: &str) -> Result<SsmlDocument, ParseError> {
    if input.contains('<') {
        match parse_xml(input) {
            ParseOutcome::Ok(doc) => return Ok(doc),
            ParseOutcome::Fallback(mut bad) => {
                let mut plain = SsmlDocument::default();
                push_plain_words(&mut plain, input.as_bytes(), 0, &OffsetMap::identity(input));
                bad.warnings.insert(
                    0,
                    "input looked like XML but did not parse cleanly; treated as plain text".into(),
                );
                plain.warnings = bad.warnings;
                return Ok(plain);
            }
        }
    }
    let mut doc = SsmlDocument::default();
    push_plain_words(&mut doc, input.as_bytes(), 0, &OffsetMap::identity(input));
    Ok(doc)
}

enum ParseOutcome {
    Ok(SsmlDocument),
    Fallback(SsmlDocument),
}

/// Byte-offset → char-offset conversion table for one input.
struct OffsetMap {
    /// `bytes_to_chars[i]` = char index of byte `i` (`input.len()` + 1 entries).
    bytes_to_chars: Vec<usize>,
}

impl OffsetMap {
    fn identity(input: &str) -> Self {
        Self::build(input)
    }

    fn build(input: &str) -> Self {
        let bytes = input.as_bytes();
        let mut map = Vec::with_capacity(bytes.len() + 1);
        map.push(0usize);
        let mut starts = 0usize;
        for (b, byte) in bytes.iter().enumerate() {
            // A UTF-8 char start is byte 0 or any non-continuation byte.
            if b == 0 || (byte & 0xC0) != 0x80 {
                starts += 1;
            }
            map.push(starts);
        }
        Self {
            bytes_to_chars: map,
        }
    }

    fn char_at(&self, byte: usize) -> usize {
        self.bytes_to_chars
            .get(byte)
            .copied()
            .unwrap_or_else(|| self.bytes_to_chars.last().copied().unwrap_or(0))
    }
}

struct Override {
    is_phoneme: bool,
    /// Byte offset just after the start tag (`>`).
    inner_start: usize,
    /// Byte offset just before the matching end tag (`<`), set as text arrives.
    inner_end: usize,
    phonemes: Option<Vec<String>>,
    alias: Option<String>,
}

struct ParserState {
    prosody_stack: Vec<Prosody>,
    say_as_stack: Vec<SayAs>,
    voice_stack: Vec<String>,
    override_active: Option<Override>,
    pending: Vec<WordSpan>,
    warnings: Vec<String>,
    fatal: bool,
    /// Byte offset just past the last text run seen (for slice recovery).
    last_text_end: usize,
    /// Accumulates words across Text/GeneralRef event splits.
    assembler: WordAssembler,
}

impl ParserState {
    fn new() -> Self {
        Self {
            prosody_stack: vec![Prosody::default()],
            say_as_stack: vec![SayAs::None],
            voice_stack: Vec::new(),
            override_active: None,
            pending: Vec::new(),
            warnings: Vec::new(),
            fatal: false,
            last_text_end: 0,
            assembler: WordAssembler::default(),
        }
    }

    fn prosody(&self) -> Prosody {
        *self.prosody_stack.last().unwrap_or(&Prosody::default())
    }

    fn say_as(&self) -> SayAs {
        self.say_as_stack.last().copied().unwrap_or_default()
    }

    fn voice(&self) -> Option<String> {
        self.voice_stack.last().cloned()
    }
}

fn parse_xml(input: &str) -> ParseOutcome {
    use quick_xml::events::Event;

    let mut doc = SsmlDocument::default();
    let mut st = ParserState::new();
    let offsets = OffsetMap::build(input);
    let mut reader = quick_xml::Reader::from_str(input);
    reader.config_mut().expand_empty_elements = false;
    reader.config_mut().trim_text(false);

    loop {
        // Position at the END of the previous event == start of the next.
        let event_start = reader.buffer_position() as usize;
        let event = match reader.read_event() {
            Ok(ev) => ev,
            Err(e) => {
                st.warnings
                    .push(format!("XML error near byte {event_start}: {e}"));
                st.fatal = true;
                break;
            }
        };
        let event_end = reader.buffer_position() as usize;

        match event {
            Event::Start(tag) => {
                let name = local_name(&tag);
                handle_open(
                    &mut st,
                    &mut doc,
                    &tag,
                    &name,
                    event_start,
                    event_end,
                    false,
                    &offsets,
                );
            }
            Event::Empty(tag) => {
                let name = local_name(&tag);
                handle_open(
                    &mut st,
                    &mut doc,
                    &tag,
                    &name,
                    event_start,
                    event_start,
                    true,
                    &offsets,
                );
            }
            Event::End(tag) => {
                let name = String::from_utf8_lossy(tag.name().as_ref()).into_owned();
                handle_close(&mut st, &mut doc, &name, event_start, input, &offsets);
            }
            Event::Text(t) => {
                // Raw (still escaped) source bytes are exactly this slice.
                let raw = &input[event_start..event_end.min(input.len())];
                debug_assert_eq!(raw.as_bytes(), t.as_ref(), "quick-xml text slice mismatch");
                st.on_text(raw.as_bytes(), input, &offsets);
            }
            Event::GeneralRef(g) => {
                // quick-xml splits text runs at entities; the event carries
                // the entity body without `&`/`;`, while the raw source
                // slice below includes them (which is what span tracking
                // needs).
                let raw = &input[event_start..event_end.min(input.len())];
                debug_assert!(
                    raw.len() >= 2 && raw.as_bytes()[1..raw.len() - 1] == *g.as_ref(),
                    "quick-xml genref slice mismatch"
                );
                st.on_text(raw.as_bytes(), input, &offsets);
            }
            Event::CData(t) => {
                // The event spans `<![CDATA[` + content + `]]>`; recover the
                // inner raw slice from the source.
                let start = event_start + b"<![CDATA[".len();
                let end = event_end.saturating_sub(b"]]>".len()).max(start);
                let raw = &input[start..end.min(input.len())];
                let _ = t;
                st.on_text(raw.as_bytes(), input, &offsets);
            }
            Event::Comment(_) | Event::Decl(_) | Event::PI(_) | Event::DocType(_) => {}
            Event::Eof => break,
        }
    }

    st.flush(&mut doc, &offsets);
    doc.warnings = st.warnings;
    if st.fatal {
        ParseOutcome::Fallback(doc)
    } else {
        ParseOutcome::Ok(doc)
    }
}

impl ParserState {
    fn on_text(&mut self, raw: &[u8], input: &str, _offsets: &OffsetMap) {
        let run_start = locate_run(input, self.last_text_end, raw);
        self.last_text_end = run_start + raw.len();
        if let Some(ov) = self.override_active.as_mut() {
            ov.inner_end = self.last_text_end;
            return;
        }
        self.assembler.push_run(raw, run_start);
    }

    /// Flush accumulated text into pending word spans. Called at tag
    /// boundaries so words split across Text/GeneralRef events merge.
    fn drain_text(&mut self, offsets: &OffsetMap) {
        let words = self.assembler.finish();
        for w in words {
            self.pending.push(WordSpan {
                char_span: offsets.char_at(w.byte_span.start)..offsets.char_at(w.byte_span.end),
                byte_span: w.byte_span,
                text: w.text.clone(),
                spoken: w.text,
                phonemes: None,
                prosody: self.prosody(),
                say_as: self.say_as(),
                voice: self.voice(),
            });
        }
    }

    fn flush(&mut self, doc: &mut SsmlDocument, offsets: &OffsetMap) {
        self.drain_text(offsets);
        if !self.pending.is_empty() {
            doc.segments.push(Segment::Words {
                words: std::mem::take(&mut self.pending),
            });
        }
    }
}

/// Find where `raw` (a raw text-run byte slice) occurs in `input`, searching
/// from `from`. Text runs are located by forward scan because quick-xml does
/// not hand out source slices.
fn locate_run(input: &str, from: usize, raw: &[u8]) -> usize {
    if raw.is_empty() {
        return from;
    }
    let hay = input.as_bytes();
    let start = from.min(hay.len());
    if hay.len() >= raw.len() && hay[start..].starts_with(raw) {
        return start;
    }
    // Fallback: full scan (rare; only if `from` drifted).
    memfind(hay, raw).unwrap_or(start)
}

fn memfind(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Handle a start (or empty) tag.
#[allow(clippy::too_many_arguments)]
fn handle_open(
    st: &mut ParserState,
    doc: &mut SsmlDocument,
    tag: &quick_xml::events::BytesStart<'_>,
    name: &str,
    tag_start: usize,
    tag_end: usize,
    is_empty_form: bool,
    offsets: &OffsetMap,
) {
    // Scoped tags close the current word run so prosody changes are visible.
    match name {
        "break" | "mark" | "prosody" | "emphasis" | "say-as" | "voice" | "phoneme" | "sub" => {
            st.flush(doc, offsets);
        }
        _ => {}
    }
    match name {
        "break" => {
            let ms = attr(tag, "time")
                .and_then(|t| parse_time(&t))
                .or_else(|| attr(tag, "strength").and_then(|s| parse_break_strength(&s)))
                .unwrap_or_else(|| {
                    st.warnings
                        .push("break without usable time/strength; using 0 ms".into());
                    0
                });
            doc.segments.push(Segment::Break {
                ms,
                char_pos: offsets.char_at(tag_start),
                byte_pos: tag_start,
            });
        }
        "mark" => {
            let mark_name = attr(tag, "name").unwrap_or_default();
            doc.segments.push(Segment::Mark {
                name: mark_name,
                char_pos: offsets.char_at(tag_start),
                byte_pos: tag_start,
            });
        }
        "prosody" => {
            let mut p = Prosody::default();
            if let Some(v) = attr(tag, "rate") {
                match parse_rate(&v) {
                    Some(r) => p.rate = Some(r),
                    None => st.warnings.push(format!("ignored prosody rate {v:?}")),
                }
            }
            if let Some(v) = attr(tag, "pitch") {
                match parse_pitch(&v) {
                    Some(r) => p.pitch = Some(r),
                    None => st.warnings.push(format!("ignored prosody pitch {v:?}")),
                }
            }
            if let Some(v) = attr(tag, "volume") {
                match parse_volume(&v) {
                    Some(r) => p.volume = Some(r),
                    None => st.warnings.push(format!("ignored prosody volume {v:?}")),
                }
            }
            st.prosody_stack.push(st.prosody().overlay(p));
        }
        "emphasis" => {
            let level = attr(tag, "level").unwrap_or_else(|| "moderate".into());
            let p = match level.as_str() {
                "strong" => Prosody {
                    rate: Some(0.97),
                    pitch: Some(1.15),
                    volume: Some(1.2),
                },
                "moderate" => Prosody {
                    rate: Some(0.99),
                    pitch: Some(1.06),
                    volume: Some(1.1),
                },
                "reduced" => Prosody {
                    rate: Some(0.92),
                    pitch: Some(0.9),
                    volume: Some(0.8),
                },
                other => {
                    st.warnings
                        .push(format!("unknown emphasis level {other:?}; using default"));
                    Prosody::default()
                }
            };
            st.prosody_stack.push(st.prosody().overlay(p));
        }
        "say-as" => {
            let mode = match attr(tag, "interpret-as").as_deref() {
                Some("characters" | "spell-out") => SayAs::Characters,
                Some("cardinal" | "number") => SayAs::Cardinal,
                Some("ordinal") => SayAs::Ordinal,
                Some(_) => SayAs::Other,
                None => SayAs::Other,
            };
            st.say_as_stack.push(mode);
        }
        "voice" => {
            st.voice_stack.push(attr(tag, "name").unwrap_or_default());
        }
        "phoneme" => {
            let ph = attr(tag, "ph")
                .map(|s| s.split_whitespace().map(str::to_owned).collect::<Vec<_>>());
            match ph {
                Some(phonemes) => {
                    st.override_active = Some(Override {
                        is_phoneme: true,
                        inner_start: tag_end,
                        inner_end: tag_end,
                        phonemes: Some(phonemes),
                        alias: None,
                    });
                }
                None => st
                    .warnings
                    .push("phoneme tag without ph attribute; override ignored".into()),
            }
        }
        "sub" => match attr(tag, "alias") {
            Some(alias) => {
                st.override_active = Some(Override {
                    is_phoneme: false,
                    inner_start: tag_end,
                    inner_end: tag_end,
                    phonemes: None,
                    alias: Some(alias),
                });
            }
            None => st
                .warnings
                .push("sub tag without alias attribute; override ignored".into()),
        },
        "audio" => {
            st.warnings
                .push("audio tag ignored (pre-recorded audio not supported)".into());
        }
        _ => {}
    }
    // Self-closing variants of scoped tags need an immediate pop.
    if is_empty_form {
        match name {
            "prosody" | "emphasis" => {
                st.prosody_stack.pop();
            }
            "say-as" => {
                st.say_as_stack.pop();
            }
            "voice" => {
                st.voice_stack.pop();
            }
            _ => {}
        }
    }
}

/// Handle an end tag.
fn handle_close(
    st: &mut ParserState,
    doc: &mut SsmlDocument,
    name: &str,
    tag_start: usize,
    input: &str,
    offsets: &OffsetMap,
) {
    match name {
        "prosody" | "emphasis" => {
            st.flush(doc, offsets);
            st.prosody_stack.pop();
        }
        "say-as" => {
            st.flush(doc, offsets);
            st.say_as_stack.pop();
        }
        "voice" => {
            st.flush(doc, offsets);
            st.voice_stack.pop();
        }
        "phoneme" | "sub" => {
            if let Some(ov) = st.override_active.take() {
                let inner = &input[ov.inner_start..ov.inner_end.min(input.len())];
                let decoded: Vec<String> = decode_words(inner.as_bytes(), ov.inner_start)
                    .into_iter()
                    .map(|w| w.text)
                    .collect();
                if decoded.is_empty() {
                    st.warnings
                        .push(format!("{name} element with no inner text; ignored"));
                } else {
                    let byte_span = ov.inner_start..ov.inner_end.min(input.len());
                    let joined = decoded.join(" ");
                    let spoken = if ov.is_phoneme {
                        joined.clone()
                    } else {
                        ov.alias.clone().unwrap_or(joined.clone())
                    };
                    st.pending.push(WordSpan {
                        text: joined,
                        spoken,
                        char_span: offsets.char_at(byte_span.start)..offsets.char_at(byte_span.end),
                        byte_span,
                        phonemes: ov.phonemes,
                        prosody: st.prosody(),
                        say_as: st.say_as(),
                        voice: st.voice(),
                    });
                }
            }
        }
        "s" => {
            st.flush(doc, offsets);
            doc.segments.push(Segment::SentenceEnd {
                char_pos: offsets.char_at(tag_start),
                byte_pos: tag_start,
            });
        }
        "p" => {
            st.flush(doc, offsets);
            doc.segments.push(Segment::ParagraphEnd {
                char_pos: offsets.char_at(tag_start),
                byte_pos: tag_start,
            });
        }
        _ => {}
    }
}

fn local_name(tag: &quick_xml::events::BytesStart<'_>) -> String {
    String::from_utf8_lossy(tag.name().as_ref()).into_owned()
}

fn attr(tag: &quick_xml::events::BytesStart<'_>, key: &str) -> Option<String> {
    tag.attributes()
        .flatten()
        .find(|a| a.key.as_ref() == key.as_bytes())
        .and_then(|a| a.unescape_value().ok().map(std::borrow::Cow::into_owned))
}

/// Push plain (non-XML) text as a single word run.
fn push_plain_words(doc: &mut SsmlDocument, raw: &[u8], start: usize, offsets: &OffsetMap) {
    let mut words = Vec::new();
    for w in decode_words(raw, start) {
        words.push(WordSpan {
            char_span: offsets.char_at(w.byte_span.start)..offsets.char_at(w.byte_span.end),
            byte_span: w.byte_span,
            text: w.text.clone(),
            spoken: w.text,
            phonemes: None,
            prosody: Prosody::default(),
            say_as: SayAs::None,
            voice: None,
        });
    }
    if !words.is_empty() {
        doc.segments.push(Segment::Words { words });
    }
}

/// A decoded word with its raw source byte span (entities included).
struct DecodedWord {
    text: String,
    byte_span: Range<usize>,
}

/// Assembles words across multiple raw text runs (quick-xml splits text at
/// entity references), preserving exact byte spans in the original input.
#[derive(Default)]
struct WordAssembler {
    words: Vec<DecodedWord>,
    text: String,
    start: Option<usize>,
    end: usize,
}

impl WordAssembler {
    fn push_run(&mut self, raw: &[u8], run_start: usize) {
        let mut i = 0;
        while i < raw.len() {
            let b = raw[i];
            if b == b'&' {
                if let Some(semi) = raw[i + 1..].iter().position(|&c| c == b';') {
                    let ent = &raw[i..i + semi + 2];
                    let decoded = decode_entity(ent);
                    for ch in decoded.chars() {
                        if ch.is_whitespace() {
                            self.flush_word(run_start + i);
                        } else {
                            if self.start.is_none() {
                                self.start = Some(run_start + i);
                            }
                            self.text.push(ch);
                        }
                    }
                    self.end = run_start + i + semi + 2;
                    i += semi + 2;
                    continue;
                }
            }
            let len = utf8_len(b);
            let chunk = &raw[i..(i + len).min(raw.len())];
            let s = String::from_utf8_lossy(chunk).into_owned();
            for ch in s.chars() {
                if ch.is_whitespace() {
                    self.flush_word(run_start + i);
                } else {
                    if self.start.is_none() {
                        self.start = Some(run_start + i);
                    }
                    self.text.push(ch);
                }
            }
            self.end = run_start + i + len;
            i += len;
        }
    }

    fn flush_word(&mut self, pos: usize) {
        if self.text.is_empty() {
            self.start = None;
        } else {
            let start = self.start.take().unwrap_or(pos);
            self.words.push(DecodedWord {
                text: std::mem::take(&mut self.text),
                byte_span: start..self.end.max(pos),
            });
        }
    }

    /// Return assembled words and reset for the next text sequence.
    fn finish(&mut self) -> Vec<DecodedWord> {
        if !self.text.is_empty() {
            let start = self.start.take().unwrap_or(self.end);
            let end = self.end;
            self.words.push(DecodedWord {
                text: std::mem::take(&mut self.text),
                byte_span: start..end,
            });
        }
        std::mem::take(&mut self.words)
    }
}

/// Decode one raw text run into words (single-run convenience).
fn decode_words(raw: &[u8], run_start: usize) -> Vec<DecodedWord> {
    let mut a = WordAssembler::default();
    a.push_run(raw, run_start);
    a.finish()
}

fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// Decode a single `&...;` entity to its replacement text.
fn decode_entity(ent: &[u8]) -> String {
    if ent.len() < 3 {
        return String::from_utf8_lossy(ent).into_owned();
    }
    let inner = &ent[1..ent.len() - 1];
    match inner {
        b"amp" => "&".into(),
        b"lt" => "<".into(),
        b"gt" => ">".into(),
        b"quot" => "\"".into(),
        b"apos" => "'".into(),
        b"nbsp" => "\u{a0}".into(),
        _ => {
            if let Some(hex) = inner
                .strip_prefix(b"#x")
                .or_else(|| inner.strip_prefix(b"#X"))
            {
                u32::from_str_radix(&String::from_utf8_lossy(hex), 16)
                    .ok()
                    .and_then(char::from_u32)
                    .map(String::from)
                    .unwrap_or_default()
            } else if let Some(dec) = inner.strip_prefix(b"#") {
                String::from_utf8_lossy(dec)
                    .parse::<u32>()
                    .ok()
                    .and_then(char::from_u32)
                    .map(String::from)
                    .unwrap_or_default()
            } else {
                String::new()
            }
        }
    }
}

/// Parse an SSML time value (`"500ms"`, `"2s"`, `"1.5s"`, bare number = ms).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn parse_time(t: &str) -> Option<u32> {
    let t = t.trim();
    if let Some(v) = t.strip_suffix("ms") {
        v.trim().parse::<f64>().ok().map(|f| f.max(0.0) as u32)
    } else if let Some(v) = t.strip_suffix('s') {
        v.trim()
            .parse::<f64>()
            .ok()
            .map(|f| (f * 1000.0).max(0.0) as u32)
    } else {
        t.parse::<f64>().ok().map(|f| f.max(0.0) as u32)
    }
}

/// W3C break strengths mapped to default millisecond pauses.
#[must_use]
pub fn parse_break_strength(s: &str) -> Option<u32> {
    match s {
        "none" => Some(0),
        "x-weak" => Some(50),
        "weak" => Some(100),
        "medium" => Some(250),
        "strong" => Some(500),
        "x-strong" => Some(800),
        _ => None,
    }
}

/// Parse a rate value: ratio (`"0.8"`), percentage (`"80%"`), relative
/// (`"+20%"` / `"-20%"`), or W3C names.
#[must_use]
pub fn parse_rate(r: &str) -> Option<f32> {
    let r = r.trim();
    let named = match r {
        "x-slow" => Some(0.5),
        "slow" => Some(0.75),
        "medium" | "default" => Some(1.0),
        "fast" => Some(1.25),
        "x-fast" => Some(1.5),
        _ => None,
    };
    named.or_else(|| parse_relative_percent(r)).or_else(|| {
        let v = r.parse::<f32>().ok()?;
        (0.1..=10.0).contains(&v).then_some(v)
    })
}

/// Parse a pitch value: percentage, semitones (`"+3st"`), or W3C names.
#[must_use]
pub fn parse_pitch(p: &str) -> Option<f32> {
    let p = p.trim();
    let named = match p {
        "x-low" => Some(0.7),
        "low" => Some(0.85),
        "medium" | "default" => Some(1.0),
        "high" => Some(1.15),
        "x-high" => Some(1.3),
        _ => None,
    };
    if let Some(v) = named {
        return Some(v);
    }
    if let Some(st) = p.strip_suffix("st") {
        return st
            .trim()
            .parse::<f32>()
            .ok()
            .map(|n| 2.0_f32.powf(n / 12.0));
    }
    parse_relative_percent(p)
}

/// Parse a volume value: percentage, relative, W3C names, or 0–100 number.
#[must_use]
pub fn parse_volume(v: &str) -> Option<f32> {
    let v = v.trim();
    let named = match v {
        "silent" => Some(0.0),
        "x-soft" => Some(0.3),
        "soft" => Some(0.6),
        "medium" | "default" => Some(1.0),
        "loud" => Some(1.3),
        "x-loud" => Some(1.6),
        _ => None,
    };
    if let Some(x) = named {
        return Some(x);
    }
    if v.ends_with('%') {
        let n = v.strip_suffix('%')?.trim().parse::<f32>().ok()?;
        return Some((n / 100.0).clamp(0.0, 2.0));
    }
    let n = v.parse::<f32>().ok()?;
    Some((n / 100.0).clamp(0.0, 2.0))
}

/// `"+20%"` / `"-20%"` / `"80%"` → multiplier against 1.0.
fn parse_relative_percent(s: &str) -> Option<f32> {
    let s = s.trim();
    let pct = s.strip_suffix('%')?.trim();
    if let Some(rel) = pct.strip_prefix('+') {
        let r: f32 = rel.trim().parse().ok()?;
        return Some((1.0 + r / 100.0).max(0.05));
    }
    if let Some(rel) = pct.strip_prefix('-') {
        let r: f32 = rel.trim().parse().ok()?;
        return Some((1.0 - r / 100.0).max(0.05));
    }
    let n: f32 = pct.parse().ok()?;
    Some((n / 100.0).max(0.05))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_words() {
        let doc = parse("Hello world").unwrap();
        let words = doc.words();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[0].char_span, 0..5);
        assert_eq!(words[1].char_span, 6..11);
        assert_eq!(words[1].byte_span, 6..11);
    }

    #[test]
    fn speak_root_strips() {
        let doc = parse("<speak>Hello world</speak>").unwrap();
        assert_eq!(doc.words().len(), 2);
        assert_eq!(doc.spoken_text(), "Hello world");
    }

    #[test]
    fn break_time_variants() {
        let doc = parse(
            "<speak>a<break time=\"500ms\"/>b<break time=\"2s\"/>c<break time=\"1.5s\"/>d</speak>",
        )
        .unwrap();
        let breaks: Vec<u32> = doc
            .segments
            .iter()
            .filter_map(|s| match s {
                Segment::Break { ms, .. } => Some(*ms),
                _ => None,
            })
            .collect();
        assert_eq!(breaks, vec![500, 2000, 1500]);
    }

    #[test]
    fn break_strength() {
        let doc = parse("<speak>a<break strength=\"strong\"/>b</speak>").unwrap();
        assert!(doc
            .segments
            .iter()
            .any(|s| matches!(s, Segment::Break { ms: 500, .. })));
    }

    #[test]
    fn mark_name_and_position() {
        let ssml = "<speak>Hi <mark name=\"m1\"/>there</speak>";
        let doc = parse(ssml).unwrap();
        match &doc.segments[1] {
            Segment::Mark {
                name,
                char_pos,
                byte_pos,
            } => {
                assert_eq!(name, "m1");
                // "<speak>" (7) + "Hi " (3) → the mark tag's '<' sits at byte 10.
                assert_eq!(*byte_pos, 10);
                assert_eq!(*char_pos, 10);
                assert_eq!(&ssml[(*byte_pos)..=(*byte_pos)], "<");
            }
            other => panic!("expected mark, got {other:?}"),
        }
        // words after the mark still parse
        assert_eq!(doc.words()[1].text, "there");
    }

    #[test]
    fn prosody_resolution() {
        let doc = parse(
            "<speak><prosody rate=\"80%\">slow <prosody pitch=\"+5%\">both</prosody></prosody>normal</speak>",
        )
        .unwrap();
        let words = doc.words();
        assert_eq!(words[0].prosody.rate, Some(0.8));
        assert_eq!(words[0].prosody.pitch, None);
        assert_eq!(words[1].prosody.rate, Some(0.8));
        assert_eq!(words[1].prosody.pitch, Some(1.05));
        assert!(words[2].prosody.is_default());
    }

    #[test]
    fn phoneme_override() {
        let doc =
            parse("<speak>Say <phoneme ph=\"f ə n ɛ t ɪ k s\">phonetics</phoneme> now</speak>")
                .unwrap();
        let words = doc.words();
        assert_eq!(words.len(), 3);
        assert_eq!(words[1].text, "phonetics");
        let ph = words[1].phonemes.as_deref().expect("override present");
        assert_eq!(ph, &["f", "ə", "n", "ɛ", "t", "ɪ", "k", "s"]);
    }

    #[test]
    fn sub_alias() {
        let ssml = "<speak>Visit <sub alias=\"World Wide Web\">WWW</sub> today</speak>";
        let doc = parse(ssml).unwrap();
        let words = doc.words();
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "Visit");
        assert_eq!(words[1].text, "WWW");
        assert_eq!(words[1].spoken, "World Wide Web");
        // Byte span covers the ORIGINAL inner text ("WWW"), not the alias.
        let start = ssml.find("WWW").expect("WWW in source");
        assert_eq!(words[1].byte_span, start..start + 3);
    }

    #[test]
    fn entities_preserve_spans() {
        // "AT&T" written with &amp; — 12 raw bytes for the word
        let ssml = "<speak>AT&amp;T rules</speak>";
        let doc = parse(ssml).unwrap();
        let words = doc.words();
        assert_eq!(words[0].text, "AT&T");
        assert_eq!(words[0].byte_len(), "AT&amp;T".len());
        assert_eq!(words[0].char_len(), "AT&amp;T".len()); // all-ASCII source
        let span = words[0].byte_span.clone();
        assert_eq!(&ssml[span], "AT&amp;T");
        assert_eq!(words[1].text, "rules");
    }

    #[test]
    fn numeric_entity() {
        let doc = parse("<speak>&#65;&#66; cd</speak>").unwrap();
        assert_eq!(doc.words()[0].text, "AB");
    }

    #[test]
    fn unicode_char_vs_byte() {
        // "<speak>" is 7 ASCII bytes; h=byte7/char7, é=bytes8-9/char8 …
        let ssml = "<speak>héllo wörld</speak>";
        let doc = parse(ssml).unwrap();
        let w = &doc.words()[0];
        assert_eq!(w.text, "héllo");
        assert_eq!(w.byte_span, 7..13); // h(1) é(2) l l o = 6 bytes
        assert_eq!(w.char_span, 7..12); // 5 characters
        let w2 = &doc.words()[1];
        assert_eq!(w2.text, "wörld");
        assert_eq!(w2.byte_span, 14..20); // w ö(2) r l d = 6 bytes
        assert_eq!(w2.char_span, 13..18);
    }

    #[test]
    fn say_as_characters() {
        let doc =
            parse("<speak>code <say-as interpret-as=\"characters\">ABC</say-as></speak>").unwrap();
        assert_eq!(doc.words()[1].say_as, SayAs::Characters);
    }

    #[test]
    fn malformed_falls_back_to_plain() {
        let doc = parse("<speak><unclosed malformed").unwrap();
        assert!(!doc.warnings.is_empty());
        assert!(doc.spoken_text().contains("unclosed"));
    }

    #[test]
    fn unknown_tags_transparent() {
        let doc = parse("<speak>Hello <marvelous>brave</marvelous> world</speak>").unwrap();
        assert_eq!(doc.spoken_text(), "Hello brave world");
        assert!(doc.warnings.is_empty());
    }

    #[test]
    fn sentence_and_paragraph_markers() {
        let doc = parse("<speak><p>One<s>two</s></p></speak>").unwrap();
        assert!(doc
            .segments
            .iter()
            .any(|s| matches!(s, Segment::SentenceEnd { .. })));
        assert!(doc
            .segments
            .iter()
            .any(|s| matches!(s, Segment::ParagraphEnd { .. })));
    }

    #[test]
    fn empty_element_prosody_no_hang() {
        let doc = parse("<speak><prosody rate=\"slow\"/>word</speak>").unwrap();
        assert_eq!(doc.words().len(), 1);
        assert!(doc.words()[0].prosody.is_default());
    }

    #[test]
    fn cdata_text_spoken() {
        let doc = parse("<speak><![CDATA[Hello there]]></speak>").unwrap();
        assert_eq!(doc.spoken_text(), "Hello there");
    }

    #[test]
    fn time_value_parsing() {
        assert_eq!(parse_time("500ms"), Some(500));
        assert_eq!(parse_time("2s"), Some(2000));
        assert_eq!(parse_time("1.5s"), Some(1500));
        assert_eq!(parse_time("300"), Some(300));
        assert_eq!(parse_time("junk"), None);
    }

    #[test]
    fn rate_parsing_variants() {
        assert_eq!(parse_rate("0.8"), Some(0.8));
        assert_eq!(parse_rate("80%"), Some(0.8));
        assert_eq!(parse_rate("+20%"), Some(1.2));
        assert_eq!(parse_rate("-20%"), Some(0.8));
        assert_eq!(parse_rate("slow"), Some(0.75));
        assert_eq!(parse_rate("wat"), None);
    }

    #[test]
    fn pitch_semitones() {
        let p = parse_pitch("+3st").unwrap();
        assert!((p - 2.0_f32.powf(3.0 / 12.0)).abs() < 1e-6);
        assert_eq!(parse_pitch("x-high"), Some(1.3));
    }
}
