//! ONNX acoustic synthesis via `ort`, with duration-aware event emission.
//!
//! Voice families (piper/MMS VITS, Matcha+vocoder) live in
//! [`crate::backends`] behind the [`crate::VoiceBackend`] trait. When the
//! model has been patched by `python/add_durations_output.py` (extra
//! `"durations"` output), word and mark events carry **measured** sample
//! positions. Stock models fall back to [`crate::estimate`] timings
//! flagged `estimated: true`.

use crate::backends::VoiceBackend;
use crate::estimate::estimate_timings;
use crate::events::{SynthesisEvent, WordTiming};
use crate::{fold_word_timings, sample_at_id_index};
use anyhow::anyhow;
use floravox_g2p::TokenPhonemizer;
use floravox_ssml::{parse as parse_ssml, Segment, WordSpan};
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

/// Samples per streamed audio chunk.
const CHUNK_SAMPLES: usize = 8192;

/// Audio chunks buffered between the synthesis worker and the consumer
/// (8 × 8192 samples ≈ 4 s at 16 kHz): enough to ride out one inference
/// without underrunning, small enough to bound ahead-of-playback work.
const CHANNEL_CHUNKS: usize = 8;

/// Symbols piper-style models reserve for control ids.
const BOS: &str = "^";
const EOS: &str = "$";
const PAD: &str = "_";

/// Sequence framing a model family expects around the phoneme ids.
///
/// VITS exports (piper, MMS) and Matcha use piper's layout:
/// `BOS, PAD, (ids…, PAD)*, EOS` with `space, PAD` between words. Kokoro
/// is char-level with none of that: bare tokens with `space` between
/// words.
#[derive(Debug, Clone, Default)]
pub struct ControlSymbols {
    /// Start-of-sequence symbol (piper `^`).
    pub bos: Option<String>,
    /// End-of-sequence symbol (piper `$`).
    pub eos: Option<String>,
    /// Interleaved pad symbol (piper `_`).
    pub pad: Option<String>,
    /// Word separator symbol (piper/kokoro ` `).
    pub space: Option<String>,
}

impl ControlSymbols {
    /// Piper-style framing (`^`, `$`, `_`, ` `); `build_ids` skips any
    /// symbol missing from the model's map (MMS has no `^`/`$`/space).
    #[must_use]
    pub fn piper() -> Self {
        Self {
            bos: Some(BOS.into()),
            eos: Some(EOS.into()),
            pad: Some(PAD.into()),
            space: Some(" ".into()),
        }
    }

    /// Kokoro framing: no control wrapping, spaces between words.
    #[must_use]
    pub fn kokoro() -> Self {
        Self {
            bos: None,
            eos: None,
            pad: None,
            space: Some(" ".into()),
        }
    }
}

/// Fully-resolved voice parameters.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Frame shift in samples (durations are in frames).
    pub hop_length: u32,
    /// Symbol → ids.
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
    /// Decoder randomness.
    pub noise_scale: f32,
    /// Base length scale (rate control).
    pub length_scale: f32,
    /// Duration-predictor randomness.
    pub noise_scale_w: f32,
    /// Speaker id for multi-speaker models.
    pub speaker_id: Option<i64>,
    /// Whether the graph exposes a durations output.
    pub has_durations: bool,
    /// Old piper exports take one `[noise_scale, length_scale, noise_scale_w]`
    /// tensor named `scales`; newer ones take three separate inputs.
    pub uses_scales: bool,
    /// Sequence framing (control symbols) the family expects.
    pub framing: ControlSymbols,
    /// The voice's token table is a character inventory (MMS-style
    /// `frontend=characters` voices) rather than a phoneme map: single
    /// characters, no BOS/EOS markers, no multi-symbol phonemes. Callers
    /// without an explicit frontend choice should use
    /// [`CharFrontend`] for these.
    pub is_char_table: bool,
}

/// A block of audio with its absolute position in the utterance.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// f32 PCM samples.
    pub samples: Vec<f32>,
    /// Absolute sample index of `samples[0]`.
    pub first_sample: u64,
    /// Sample rate.
    pub sample_rate: u32,
}

/// Synthesizer combining a voice model with a phonemizer.
///
/// Clone-safe for streaming: synthesis runs on a worker thread sharing the
/// model and phonemizer behind a mutex (one utterance at a time).
pub struct Synthesizer<G: TokenPhonemizer> {
    inner: Arc<Mutex<Inner<G>>>,
}

struct Inner<G> {
    backend: Box<dyn VoiceBackend>,
    g2p: G,
    pre_pass: Option<Box<dyn DocumentPhonemizer>>,
}

/// Document-level phonemizer: assigns phonemes to a run of words with
/// sentence context. POS-aware engines (misaki) need whole sentences to
/// disambiguate heteronyms and expand numbers; per-word `TokenPhonemizer`
/// calls cannot provide that.
///
/// Assignments only fill words whose `phonemes` are still `None` (SSML
/// `<phoneme>` overrides always win) and skip character-mode `say-as`.
pub trait DocumentPhonemizer: Send {
    /// Fill `word.phonemes` for the run (in place).
    fn assign_phonemes(&mut self, words: &mut [WordSpan]);
}

/// [`DocumentPhonemizer`] that assigns each character of each word as
/// its own symbol, for voices trained on romanized text rather than
/// phonemes (MMS and friends: every MMS voice's `tokens.txt` is a
/// per-language character inventory). Symbol resolution in `build_ids`
/// maps the characters through the voice's own table, and the
/// per-symbol pad insertion gives MMS's char-pad-char framing.
pub struct CharFrontend {
    /// Lowercase input first (character inventories are mostly lowercase).
    pub lowercase: bool,
    /// Romanize input with uroman first (non-Latin scripts -> Latin
    /// characters). An optional ISO 639-3 code selects language-specific
    /// rules.
    pub romanize: Option<&'static str>,
}

impl DocumentPhonemizer for CharFrontend {
    fn assign_phonemes(&mut self, words: &mut [WordSpan]) {
        for w in words {
            if w.phonemes.is_some() || w.say_as == floravox_ssml::SayAs::Characters {
                continue;
            }
            #[cfg(feature = "uroman")]
            let text = if let Some(lang) = self.romanize {
                floravox_g2p::uroman::romanize(&w.spoken, Some(lang))
            } else {
                w.spoken.clone()
            };
            #[cfg(not(feature = "uroman"))]
            let text = w.spoken.clone();
            let text = if self.lowercase {
                text.to_lowercase()
            } else {
                text
            };
            let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            if !chars.is_empty() {
                w.phonemes = Some(chars);
            }
        }
    }
}

/// [`DocumentPhonemizer`] backed by [`floravox_g2p::MisakiG2p`] (feature
/// `misaki`) — the phonemizer Kokoro voices were trained with.
#[cfg(feature = "misaki")]
pub struct MisakiPrePass(pub floravox_g2p::MisakiG2p);

#[cfg(feature = "misaki")]
impl DocumentPhonemizer for MisakiPrePass {
    fn assign_phonemes(&mut self, words: &mut [WordSpan]) {
        let texts: Vec<&str> = words.iter().map(|w| w.spoken.as_str()).collect();
        let results = self.0.phonemize_words(&texts);
        for (w, ph) in words.iter_mut().zip(results) {
            if w.phonemes.is_none() && w.say_as != floravox_ssml::SayAs::Characters {
                if let Some(p) = ph.filter(|p| !p.is_empty()) {
                    w.phonemes = Some(p);
                }
            }
        }
    }
}

/// One planned synthesis step.
enum PlanItem {
    /// A run of words sharing one rate.
    Words {
        words: Vec<WordSpan>,
        rate: f32,
        marks: Vec<(String, i64)>,
    },
    /// An explicit pause.
    Break { ms: u64 },
    /// Sentence boundary event.
    SentenceEnd,
    /// Paragraph boundary event.
    ParagraphEnd,
}

/// Handle returned by [`Synthesizer::synthesize_stream`].
pub struct StreamingSynthesis {
    /// Audio chunks in order; ends when the channel closes.
    pub audio: Receiver<AudioChunk>,
    /// Events, roughly in sample order; ends when the channel closes.
    pub events: Receiver<SynthesisEvent>,
}

impl StreamingSynthesis {
    /// Collect everything into memory: `(samples, events, sample_rate)`.
    /// # Errors
    ///
    /// Never fails in practice; kept `Result` for forward compatibility.
    pub fn collect(self) -> anyhow::Result<(Vec<f32>, Vec<SynthesisEvent>, u32)> {
        let mut samples = Vec::new();
        let mut events = Vec::new();
        let mut rate = 22_050u32;
        for chunk in self.audio {
            rate = chunk.sample_rate;
            samples.extend(chunk.samples.iter());
        }
        for ev in self.events {
            events.push(ev);
        }
        events.sort_by_key(SynthesisEvent::sample);
        Ok((samples, events, rate))
    }
}

impl<G: TokenPhonemizer + Send + 'static> Synthesizer<G> {
    /// Combine a loaded voice backend with a phonemizer
    /// (see []).
    pub fn new(backend: Box<dyn VoiceBackend>, g2p: G) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                backend,
                g2p,
                pre_pass: None,
            })),
        }
    }

    /// Attach a document-level phonemizer pre-pass (e.g.
    /// [`MisakiPrePass`]); it runs before the per-word chain on every
    /// same-rate word run.
    /// # Panics
    ///
    /// Panics when the internal lock is poisoned (only possible after a
    /// panicked worker thread).
    #[must_use]
    pub fn with_document_phonemizer(self, pre_pass: Box<dyn DocumentPhonemizer>) -> Self {
        self.inner
            .lock()
            .expect("freshly constructed synthesizer is unlocked")
            .pre_pass = Some(pre_pass);
        self
    }

    /// Synthesize to completion in memory: `(samples, events, sample_rate)`.
    /// # Errors
    ///
    /// Propagates parsing or ONNX inference failures.
    pub fn synthesize(&self, input: &str) -> anyhow::Result<(Vec<f32>, Vec<SynthesisEvent>, u32)> {
        self.synthesize_stream(input)?.collect()
    }

    /// Synthesize SSML or plain text, streaming audio chunks and events over
    /// channels. Synthesis runs on a worker thread; dropping the receivers
    /// cancels it.
    /// # Errors
    ///
    /// Propagates SSML parse failures (planning stage only; synthesis
    /// errors terminate the streams).
    pub fn synthesize_stream(&self, input: &str) -> anyhow::Result<StreamingSynthesis> {
        let plan = {
            let inner = self
                .inner
                .lock()
                .map_err(|_| anyhow!("synth lock poisoned"))?;
            plan_document(&inner.backend.config().phoneme_id_map, input)?
        };
        let inner = Arc::clone(&self.inner);
        // Bounded audio channel: with a real-time consumer, the worker
        // synthesizes at most CHANNEL_CHUNKS ahead of playback instead of
        // buffering (and arena-growing for) the whole utterance. Dropping
        // the receiver still cancels; sends just block until then.
        let (audio_tx, audio_rx) = mpsc::sync_channel(CHANNEL_CHUNKS);
        let (event_tx, event_rx) = mpsc::channel();

        std::thread::spawn(move || {
            let Ok(mut inner) = inner.lock() else { return };
            let Inner {
                backend,
                g2p,
                pre_pass,
            } = &mut *inner;
            let _ = synth_worker(
                backend.as_mut(),
                g2p,
                pre_pass.as_mut(),
                &plan,
                &audio_tx,
                &event_tx,
            );
        });

        Ok(StreamingSynthesis {
            audio: audio_rx,
            events: event_rx,
        })
    }
}

/// Split a document into plan items at breaks, sentence/paragraph ends, and
/// rate changes. Pure function of the phoneme map + input (no model needed).
fn plan_document(map: &HashMap<String, Vec<i64>>, input: &str) -> anyhow::Result<Vec<PlanItem>> {
    let _ = map;
    let doc = parse_ssml(input)?;
    let mut plan: Vec<PlanItem> = Vec::new();
    let mut pending_marks: Vec<(String, i64)> = Vec::new();

    let take_marks = |pending: &mut Vec<(String, i64)>| std::mem::take(pending);

    for seg in &doc.segments {
        match seg {
            Segment::Words { words } => {
                if words.is_empty() {
                    continue;
                }
                // Split at sentence-final punctuation so each sentence
                // becomes its own inference pass: audio for sentence N
                // streams out while sentence N+1 is still synthesizing
                // (and the ORT arena sees bounded pass shapes). Explicit
                // <s>/<p> tags already produced their own segments.
                // Each sentence/cap fragment is its own inference pass
                // (no same-rate re-merging): pass shape bounds the ORT
                // arena, and sentence-level streaming needs the split.
                for chunk in split_sentences(words) {
                    let rate = chunk[0].prosody.rate.unwrap_or(1.0);
                    plan.push(PlanItem::Words {
                        words: chunk,
                        rate,
                        marks: take_marks(&mut pending_marks),
                    });
                    if let Some(PlanItem::Words { words, .. }) = plan.last() {
                        if words.last().is_some_and(|w| is_sentence_final(&w.text)) {
                            plan.push(PlanItem::SentenceEnd);
                        }
                    }
                }
            }
            Segment::Break { ms, .. } => {
                // Marks waiting in front of a break fire at the break start.
                if !pending_marks.is_empty() {
                    push_or_attach(&mut plan, std::mem::take(&mut pending_marks));
                }
                plan.push(PlanItem::Break { ms: u64::from(*ms) });
            }
            Segment::Mark { name, char_pos, .. } => {
                pending_marks.push((name.clone(), i64::try_from(*char_pos).unwrap_or(-1)));
            }
            Segment::SentenceEnd { .. } => {
                close_words(&mut plan);
                if !matches!(plan.last(), Some(PlanItem::SentenceEnd)) {
                    plan.push(PlanItem::SentenceEnd);
                }
            }
            Segment::ParagraphEnd { .. } => {
                close_words(&mut plan);
                plan.push(PlanItem::ParagraphEnd);
            }
        }
    }
    // Trailing marks attach to a final zero-word item.
    if !pending_marks.is_empty() {
        push_or_attach(&mut plan, pending_marks);
    }
    if std::env::var_os("FLORAVOX_DEBUG_PLAN").is_some() {
        let lens: Vec<usize> = plan
            .iter()
            .filter_map(|p| match p {
                PlanItem::Words { words, .. } => Some(words.len()),
                _ => None,
            })
            .collect();
        eprintln!("[plan] runs: {lens:?}");
    }
    Ok(plan)
}

/// Maximum words per inference pass. Long unsplit text (comma-spliced
/// clauses, transcription output) would otherwise make one giant pass,
/// and the ORT arena grows with the longest pass shape (measured: a
/// 19-word run peaked 255 MB vs 166 MB for the same words in two
/// passes). Runs over the cap split at the nearest comma, else at a
/// word boundary; sentence boundaries always win first. Override with
/// `FLORAVOX_MAX_PASS_WORDS` (0 disables capping).
fn max_pass_words() -> usize {
    std::env::var("FLORAVOX_MAX_PASS_WORDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16)
}

/// Split a word run at sentence-final words (`.`, `!`, `?`, `...`,
/// CJK/Armenian/Thai terminals), then cap remaining runs at
/// [`max_pass_words`] by splitting at commas (preferred) or word
/// boundaries. The terminator stays with its sentence.
fn split_sentences(words: &[WordSpan]) -> Vec<Vec<WordSpan>> {
    let mut out: Vec<Vec<WordSpan>> = Vec::new();
    let mut cur: Vec<WordSpan> = Vec::new();
    for w in words {
        cur.push(w.clone());
        if is_sentence_final(&w.text) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    // Cap pass length: split over-long runs at the comma nearest the cap
    // (a natural pause), falling back to the word boundary at the cap.
    let cap = max_pass_words();
    if cap > 0 {
        let mut capped: Vec<Vec<WordSpan>> = Vec::with_capacity(out.len());
        for run in out {
            if run.len() <= cap {
                capped.push(run);
                continue;
            }
            let mut start = 0usize;
            while start < run.len() {
                let remaining = run.len() - start;
                if remaining <= cap {
                    capped.push(run[start..].to_vec());
                    break;
                }
                let window = &run[start..start + cap];
                let cut = window
                    .iter()
                    .rposition(|w| w.text.trim_end().ends_with(','))
                    .map_or(start + cap, |i| start + i + 1);
                capped.push(run[start..cut].to_vec());
                start = cut;
            }
        }
        out = capped;
    }
    if out.is_empty() {
        out.push(Vec::new());
    }
    out
}

/// True when `text` ends with a sentence-terminal character.
fn is_sentence_final(text: &str) -> bool {
    text.trim_end().chars().next_back().is_some_and(|c| {
        matches!(
            c,
            '.' | '!'
                | '?'
                | '\u{2026}'
                | '\u{3002}'
                | '\u{ff01}'
                | '\u{ff1f}'
                | '\u{0589}'
                | '\u{17d4}'
                | '\u{17d1}'
                | '\u{104b}'
        )
    })
}

/// End the current open Words item so the next one starts fresh.
fn close_words(_plan: &mut Vec<PlanItem>) {
    // Words items merge only when adjacent; inserting a boundary item
    // (SentenceEnd/ParagraphEnd) between them prevents merging.
}

/// Attach marks to the last item (or a synthetic empty one).
fn push_or_attach(plan: &mut Vec<PlanItem>, marks: Vec<(String, i64)>) {
    match plan.last_mut() {
        Some(PlanItem::Words { marks: m, .. }) => m.extend(marks),
        _ => plan.push(PlanItem::Words {
            words: Vec::new(),
            rate: 1.0,
            marks,
        }),
    }
}

/// Worker: execute the plan, sending audio and events.
#[allow(clippy::too_many_lines)]
fn synth_worker<G: TokenPhonemizer>(
    model: &mut dyn VoiceBackend,
    g2p: &mut G,
    mut pre_pass: Option<&mut Box<dyn DocumentPhonemizer>>,
    plan: &[PlanItem],
    audio_tx: &mpsc::SyncSender<AudioChunk>,
    event_tx: &mpsc::Sender<SynthesisEvent>,
) -> anyhow::Result<()> {
    let rate = model.config().sample_rate;
    let hop = model.config().hop_length;

    let send_chunk = |samples: &[f32], offset: &mut u64| -> bool {
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            if audio_tx
                .send(AudioChunk {
                    samples: chunk.to_vec(),
                    first_sample: *offset,
                    sample_rate: rate,
                })
                .is_err()
            {
                return false; // consumer dropped: cancelled
            }
            *offset += chunk.len() as u64;
        }
        true
    };

    let mut offset: u64 = 0;
    event_tx.send(SynthesisEvent::Started).ok();

    for item in plan {
        match item {
            PlanItem::Break { ms } => {
                event_tx
                    .send(SynthesisEvent::BreakStarted {
                        ms: *ms,
                        sample: offset,
                    })
                    .ok();
                let silence_len = u64::from(rate) * ms / 1000;
                let silence = vec![0.0_f32; silence_len as usize];
                if !send_chunk(&silence, &mut offset) {
                    return Ok(());
                }
                event_tx
                    .send(SynthesisEvent::BreakEnded { sample: offset })
                    .ok();
            }
            PlanItem::SentenceEnd => {
                event_tx
                    .send(SynthesisEvent::SentenceEnd { sample: offset })
                    .ok();
            }
            PlanItem::ParagraphEnd => {
                event_tx
                    .send(SynthesisEvent::ParagraphEnd { sample: offset })
                    .ok();
            }
            PlanItem::Words {
                words,
                rate: unit_rate,
                marks,
            } => {
                let mut owned: Vec<WordSpan>;
                let words: &[WordSpan] = if let Some(pp) = pre_pass.as_mut() {
                    owned = words.clone();
                    pp.assign_phonemes(&mut owned);
                    &owned
                } else {
                    words.as_slice()
                };
                if words.is_empty() {
                    // Marks with no following speech fire at current offset.
                    for (name, char_offset) in marks {
                        event_tx
                            .send(SynthesisEvent::MarkReached {
                                name: name.clone(),
                                sample: offset,
                                ms: offset * 1000 / u64::from(rate),
                                char_offset: *char_offset,
                            })
                            .ok();
                    }
                    continue;
                }

                let (ids, groups) = build_ids(
                    &model.config().phoneme_id_map,
                    &model.config().framing,
                    g2p,
                    words,
                );
                if ids.is_empty() {
                    continue;
                }

                let length_scale = model.config().length_scale / unit_rate.max(0.1);
                let (audio, durations) = model.run(&ids, length_scale)?;
                let d_ok = durations.as_ref().is_some_and(|d| d.len() == ids.len());

                // Word timings (measured or estimated).
                let mut timings: Vec<WordTiming> = match &durations {
                    Some(d) if d_ok => {
                        let samples = fold_word_timings(d, &groups, hop);
                        words
                            .iter()
                            .zip(samples)
                            .map(|(w, (s, e))| WordTiming {
                                text: w.text.clone(),
                                byte_offset: w.byte_span.start,
                                byte_len: w.byte_len(),
                                char_offset: w.char_span.start,
                                char_len: w.char_len(),
                                sample_start: offset + s,
                                sample_end: offset + e,
                                ms_start: (offset + s) * 1000 / u64::from(rate),
                                ms_end: (offset + e) * 1000 / u64::from(rate),
                                estimated: false,
                            })
                            .collect()
                    }
                    _ => estimate_timings(words, audio.len() as u64, rate)
                        .into_iter()
                        .map(|mut t| {
                            t.sample_start += offset;
                            t.sample_end += offset;
                            t.ms_start = t.sample_start * 1000 / u64::from(rate);
                            t.ms_end = t.sample_end * 1000 / u64::from(rate);
                            t
                        })
                        .collect(),
                };

                // Leading-silence trim: models that emit a long quiet
                // stretch before the first phoneme (kokoro: ~640 ms)
                // would attribute it to the first word, firing
                // first-word highlighting early. Shift the first word's
                // start forward to the first audible sample when the
                // gap is large (>= 200 ms) and measured.
                if !timings.is_empty() {
                    let t0 = &mut timings[0];
                    let gap_samples = t0.sample_start.saturating_sub(offset);
                    if gap_samples >= u64::from(rate) / 5 {
                        #[allow(clippy::cast_possible_truncation, clippy::useless_conversion)]
                        let span = (t0.sample_start.saturating_sub(offset)).min(audio.len() as u64)
                            as usize;
                        let cut = first_audible_offset(&audio, span, rate);
                        if cut > 0 {
                            let new_start = t0.sample_start + u64::from(cut);
                            if new_start < t0.sample_end {
                                t0.sample_start = new_start;
                                t0.ms_start = new_start * 1000 / u64::from(rate);
                            }
                        }
                    }
                }

                // Marks fire at the unit's first word start.
                let mark_sample = durations.as_ref().map_or(offset, |d| {
                    offset + sample_at_id_index(d, groups[0].start, hop)
                });
                for (name, char_offset) in marks {
                    event_tx
                        .send(SynthesisEvent::MarkReached {
                            name: name.clone(),
                            sample: mark_sample,
                            ms: mark_sample * 1000 / u64::from(rate),
                            char_offset: *char_offset,
                        })
                        .ok();
                }

                if !send_chunk(&audio, &mut offset) {
                    return Ok(());
                }
                for t in timings {
                    event_tx.send(SynthesisEvent::WordBoundary(t)).ok();
                }
            }
        }
    }

    event_tx
        .send(SynthesisEvent::Finished {
            total_samples: offset,
            total_ms: offset * 1000 / u64::from(rate),
        })
        .ok();
    Ok(())
}

/// Resolve a phoneme symbol against a model's id map, splitting
/// compounds the map doesn't carry.
///
/// Lexicons and G2P engines emit composed symbols (`oʊ`, `aɪ`, `ɜː`,
/// `t͡ʃ`); espeak-style inventories (all piper/MMS/kokoro voices) spell
/// them as separate symbols instead. Without this, unknown symbols were
/// silently dropped — 20% of symbols on a `CMUDict` lexicon sample, deleting
/// whole diphthongs. Resolution order:
///
/// 1. direct hit;
/// 2. substitution table for precomposed stragglers (`ɝ` → `ɜ` + `˞`);
/// 3. per-character split (combining length marks and modifiers resolve
///    as their own symbols; tie bars are skipped).
///
/// Returns empty when nothing resolves — the symbol is dropped, as before,
/// but only when it truly has no representation in the voice.
/// Precomposed symbols espeak-style inventories spell differently.
const SUBST: &[(&str, &str)] = &[("ɝ", "ɜ˞")];

/// ASCII homoglyphs some sources emit where IPA letters are meant
/// (`g` vs `ɡ` U+0261 — gruut lexicons do this).
const HOMOGLYPHS: &[(char, char)] = &[('g', 'ɡ')];

fn resolve_phoneme_ids<'a>(map: &'a HashMap<String, Vec<i64>>, sym: &str) -> Vec<&'a Vec<i64>> {
    if let Some(ids) = map.get(sym) {
        return vec![ids];
    }
    for (from, to) in SUBST {
        if sym == *from {
            return to
                .chars()
                .filter_map(|c| map.get(c.encode_utf8(&mut [0u8; 4])))
                .collect();
        }
    }
    let mut out = Vec::new();
    for ch in sym.chars() {
        if matches!(ch, '\u{0361}' | '\u{035C}') {
            continue; // tie bars: the parts carry the phoneme
        }
        let mut buf = [0u8; 4];
        let key = ch.encode_utf8(&mut buf);
        if let Some(ids) = map.get(key) {
            out.push(ids);
            continue;
        }
        // ASCII homoglyph retry (g -> ɡ), then combining marks the
        // voice doesn't carry (diacritics like the non-syllabic breve
        // in `aɪ̯`) are dropped rather than failing the whole symbol.
        if let Some((_, to)) = HOMOGLYPHS.iter().find(|(from, _)| *from == ch) {
            if let Some(ids) = map.get(to.encode_utf8(&mut [0u8; 4])) {
                out.push(ids);
                continue;
            }
        }
        if !ch.is_alphabetic() && !ch.is_numeric() {
            continue;
        }
        return Vec::new();
    }
    out
}

/// Samples of leading silence in `audio[..span]`, counting from 0:
/// advances while 10-ms blocks stay under 2% of the block-peak RMS.
/// Returns 0 when no significant leading silence exists.
fn first_audible_offset(audio: &[f32], span: usize, rate: u32) -> u32 {
    let win = (rate as usize / 100).max(1); // 10 ms
    if span <= win || audio.is_empty() {
        return 0;
    }
    let peak = audio
        .iter()
        .take(span)
        .fold(0.0_f32, |m, &s| m.max(s.abs()));
    let thresh = peak * 0.02;
    let mut pos = 0usize;
    while pos + win <= span {
        let block = &audio[pos..pos + win];
        #[allow(clippy::cast_precision_loss)]
        let rms = (block.iter().map(|s| s * s).sum::<f32>() / block.len() as f32).sqrt();
        if rms > thresh {
            break;
        }
        pos += win;
    }
    u32::try_from(pos).unwrap_or(0)
}

/// Build the phoneme-id sequence for a word run.
///
/// Layout follows `framing`: piper-style `BOS, PAD, (phoneme ids…, PAD)*,
/// EOS` with `space, PAD` between words; kokoro-style bare tokens with
/// `space` between words. Control symbols missing from the map are
/// skipped instead of guessed.
fn build_ids<G: TokenPhonemizer>(
    map: &HashMap<String, Vec<i64>>,
    framing: &ControlSymbols,
    g2p: &mut G,
    words: &[WordSpan],
) -> (Vec<i64>, Vec<std::ops::Range<usize>>) {
    let pad = framing.pad.as_deref().and_then(|s| map.get(s));
    let bos = framing.bos.as_deref().and_then(|s| map.get(s));
    let eos = framing.eos.as_deref().and_then(|s| map.get(s));
    let space = framing.space.as_deref().and_then(|s| map.get(s));

    let mut ids: Vec<i64> = Vec::new();
    ids.extend_from_slice(bos.unwrap_or(&Vec::new()));
    if let Some(p) = pad {
        ids.extend_from_slice(p);
    }

    let mut groups = Vec::with_capacity(words.len());
    for (wi, word) in words.iter().enumerate() {
        if wi > 0 {
            if let Some(sp) = space {
                ids.extend_from_slice(sp);
            }
            if let Some(p) = pad {
                ids.extend_from_slice(p);
            }
        }
        let phonemes: Vec<String> = match &word.phonemes {
            Some(ph) => ph.clone(),
            None => phonemize_word(g2p, word),
        };
        let start = ids.len();
        for p in &phonemes {
            for pid in resolve_phoneme_ids(map, p) {
                ids.extend_from_slice(pid);
                if let Some(p2) = pad {
                    ids.extend_from_slice(p2);
                }
            }
        }
        groups.push(start..ids.len());
    }
    ids.extend_from_slice(eos.unwrap_or(&Vec::new()));
    (ids, groups)
}

/// Phonemize one word span, honoring say-as modes.
fn phonemize_word<G: TokenPhonemizer>(g2p: &mut G, word: &WordSpan) -> Vec<String> {
    match word.say_as {
        floravox_ssml::SayAs::Characters => word
            .spoken
            .chars()
            .flat_map(|c| g2p.phonemize_token(&c.to_string()))
            .collect(),
        _ => g2p.phonemize_token(&word.spoken),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> HashMap<String, Vec<i64>> {
        // Tiny phoneme map covering the test sentence.
        [
            ("_", vec![0]),
            ("^", vec![1]),
            ("$", vec![2]),
            (" ", vec![3]),
            ("h", vec![4]),
            ("ɛ", vec![5]),
            ("l", vec![6]),
            ("o", vec![7]),
            ("w", vec![8]),
            ("ɜː", vec![9]),
            ("d", vec![10]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    #[test]
    fn plan_splits_on_break_and_keeps_marks() {
        let plan = plan_document(
            &map(),
            "<speak>a<mark name=\"m\"/><break time=\"250ms\"/>b</speak>",
        )
        .unwrap();
        let has_break = plan
            .iter()
            .any(|p| matches!(p, PlanItem::Break { ms: 250 }));
        assert!(has_break);
        // the mark attaches in front of the break
        let attached = plan.iter().any(|p| match p {
            PlanItem::Words { marks, .. } => marks.iter().any(|(m, _)| m == "m"),
            _ => false,
        });
        assert!(attached);
    }

    #[test]
    fn single_sentence_is_one_item() {
        let plan = plan_document(&map(), "hello world").unwrap();
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn plan_splits_plain_text_into_sentence_passes() {
        // Streaming: each sentence-final word ends a Words item (its own
        // inference pass) with a SentenceEnd between, so audio for
        // sentence N streams while N+1 synthesizes.
        let plan = plan_document(&map(), "Eins hier. Zwei dort! Drei ueberall? Kein Ende").unwrap();
        let words_items: Vec<&Vec<floravox_ssml::WordSpan>> = plan
            .iter()
            .filter_map(|p| match p {
                PlanItem::Words { words, .. } => Some(words),
                _ => None,
            })
            .collect();
        assert_eq!(words_items.len(), 4, "3 sentence + trailing run");
        assert!(words_items[0].last().is_some_and(|w| w.text.ends_with('.')));
        let ends = plan
            .iter()
            .filter(|p| matches!(p, PlanItem::SentenceEnd))
            .count();
        assert_eq!(ends, 3);
    }

    #[test]
    fn long_runs_are_capped_at_commas_or_boundaries() {
        // 19 words, no sentence punctuation: must split into runs <= 16.
        std::env::set_var("FLORAVOX_MAX_PASS_WORDS", "16");
        let words: Vec<&str> = "a b c d e f g h i j k l m n o p q r s"
            .split_whitespace()
            .collect();
        let spans: Vec<floravox_ssml::WordSpan> = words
            .iter()
            .map(|w| floravox_ssml::WordSpan {
                text: (*w).into(),
                spoken: (*w).into(),
                char_span: 0..1,
                byte_span: 0..1,
                phonemes: None,
                prosody: floravox_ssml::Prosody::default(),
                say_as: floravox_ssml::SayAs::None,
                voice: None,
            })
            .collect();
        let runs = split_sentences(&spans);
        assert!(
            runs.iter().all(|r| r.len() <= 16),
            "run lengths: {:?}",
            runs.iter().map(Vec::len).collect::<Vec<_>>()
        );
        // comma preference: split at the comma inside the window
        let comma_words = "w1 , w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18";
        std::env::set_var("FLORAVOX_MAX_PASS_WORDS", "8");
        let spans2: Vec<floravox_ssml::WordSpan> = comma_words
            .split_whitespace()
            .map(|w| floravox_ssml::WordSpan {
                text: w.into(),
                spoken: w.into(),
                char_span: 0..1,
                byte_span: 0..1,
                phonemes: None,
                prosody: floravox_ssml::Prosody::default(),
                say_as: floravox_ssml::SayAs::None,
                voice: None,
            })
            .collect();
        let runs2 = split_sentences(&spans2);
        assert!(runs2[0].last().is_some_and(|w| w.text == ","));
        std::env::remove_var("FLORAVOX_MAX_PASS_WORDS");
    }

    #[test]
    fn explicit_s_tags_do_not_double_sentence_ends() {
        let plan = plan_document(&map(), "<s>Eins.</s><s>Zwei.</s>").unwrap();
        let ends = plan
            .iter()
            .filter(|p| matches!(p, PlanItem::SentenceEnd))
            .count();
        assert_eq!(ends, 2, "one per </s>, not doubled");
    }

    #[test]
    fn plan_splits_on_rate_change() {
        let plan = plan_document(
            &map(),
            "<speak>a<prosody rate=\"fast\">b</prosody> c</speak>",
        )
        .unwrap();
        let word_items: usize = plan
            .iter()
            .filter(|p| matches!(p, PlanItem::Words { .. }))
            .count();
        // default | fast | back to default
        assert_eq!(word_items, 3);
    }

    #[test]
    fn resolve_splits_compound_symbols() {
        // espeak-style inventory: single chars + modifiers only.
        let m: HashMap<String, Vec<i64>> = [
            ("o", vec![7]),
            ("ʊ", vec![8]),
            ("a", vec![9]),
            ("ɪ", vec![10]),
            ("ɜ", vec![11]),
            ("˞", vec![12]),
            ("ː", vec![13]),
            ("ʃ", vec![14]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let ids = |v: Vec<&Vec<i64>>| v.into_iter().flatten().copied().collect::<Vec<i64>>();
        assert_eq!(ids(resolve_phoneme_ids(&m, "oʊ")), vec![7, 8]); // diphthong split
        assert_eq!(ids(resolve_phoneme_ids(&m, "aɪ")), vec![9, 10]);
        assert_eq!(ids(resolve_phoneme_ids(&m, "ɝ")), vec![11, 12]); // substitution
        assert_eq!(ids(resolve_phoneme_ids(&m, "ɜː")), vec![11, 13]); // length mark
        assert!(resolve_phoneme_ids(&m, "x").is_empty()); // truly unknown
    }

    #[test]
    fn resolve_handles_homoglyphs_and_loose_diacritics() {
        // gruut lexicon shapes against a piper-style inventory.
        let m: HashMap<String, Vec<i64>> = [
            ("ɡ", vec![1]),
            ("a", vec![2]),
            ("ː", vec![3]),
            ("ɪ", vec![4]),
            ("t", vec![5]),
            ("s", vec![6]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let ids = |v: Vec<&Vec<i64>>| v.into_iter().flatten().copied().collect::<Vec<i64>>();
        assert_eq!(ids(resolve_phoneme_ids(&m, "g")), vec![1]); // ASCII homoglyph
        assert_eq!(ids(resolve_phoneme_ids(&m, "aɪ̯")), vec![2, 4]); // breve dropped
        assert_eq!(ids(resolve_phoneme_ids(&m, "t͡s")), vec![5, 6]); // tie bar
        assert_eq!(ids(resolve_phoneme_ids(&m, "aː")), vec![2, 3]); // length
    }

    /// Fake backend for worker-pipeline tests: silence audio with
    /// per-id durations.
    struct FakeBackend {
        resolved: ResolvedConfig,
    }

    impl FakeBackend {
        fn new(phoneme_map: HashMap<String, Vec<i64>>) -> Self {
            Self {
                resolved: ResolvedConfig {
                    sample_rate: 16_000,
                    hop_length: 256,
                    phoneme_id_map: phoneme_map,
                    noise_scale: 0.667,
                    length_scale: 1.0,
                    noise_scale_w: 0.8,
                    speaker_id: None,
                    has_durations: true,
                    uses_scales: false,
                    framing: ControlSymbols::piper(),
                    is_char_table: false,
                },
            }
        }
    }

    impl crate::backends::VoiceBackend for FakeBackend {
        fn config(&self) -> &ResolvedConfig {
            &self.resolved
        }

        fn run(
            &mut self,
            ids: &[i64],
            _length_scale: f32,
        ) -> anyhow::Result<(Vec<f32>, Option<Vec<i64>>)> {
            let d = vec![10_i64; ids.len()];
            let n = usize::try_from(d.iter().sum::<i64>()).unwrap_or(0) * 256;
            Ok((vec![0.0; n], Some(d)))
        }
    }

    #[test]
    fn char_frontend_assigns_per_character() {
        use floravox_ssml::Prosody;
        let mk = |spoken: &str| WordSpan {
            text: spoken.into(),
            spoken: spoken.into(),
            char_span: 0..spoken.len(),
            byte_span: 0..spoken.len(),
            phonemes: None,
            prosody: Prosody::default(),
            say_as: floravox_ssml::SayAs::None,
            voice: None,
        };
        let mut words = vec![mk("Bonjour"), mk("le")];
        CharFrontend {
            lowercase: true,
            romanize: None,
        }
        .assign_phonemes(&mut words);
        assert_eq!(
            words[0].phonemes,
            Some(
                ["b", "o", "n", "j", "o", "u", "r"]
                    .map(String::from)
                    .to_vec()
            )
        );
        assert_eq!(
            words[1].phonemes,
            Some(["l", "e"].map(String::from).to_vec())
        );
        // existing phonemes (SSML overrides) are left alone
        words[0].phonemes = Some(vec!["x".into()]);
        CharFrontend {
            lowercase: true,
            romanize: None,
        }
        .assign_phonemes(&mut words);
        assert_eq!(words[0].phonemes, Some(vec!["x".into()]));
    }

    #[test]
    fn pre_pass_assigns_phonemes_before_the_chain() {
        // Pre-pass assigns "h" (id 4 in the test map); the per-word chain
        // would return "chain" (not in the map → all symbols dropped →
        // no audio). If measured word events arrive, the pre-pass won.
        struct Fixed;
        impl super::DocumentPhonemizer for Fixed {
            fn assign_phonemes(&mut self, words: &mut [WordSpan]) {
                for w in words {
                    w.phonemes = Some(vec!["h".into()]);
                }
            }
        }
        struct Chain;
        impl floravox_g2p::TokenPhonemizer for Chain {
            fn phonemize_token(&mut self, _t: &str) -> Vec<String> {
                vec!["chain".into()]
            }
        }

        let mut backend = FakeBackend::new(map());
        let plan = plan_document(&map(), "hello world").unwrap();
        let (audio_tx, audio_rx) = mpsc::sync_channel(CHANNEL_CHUNKS);
        let (event_tx, event_rx) = mpsc::channel();
        let mut pre: Box<dyn DocumentPhonemizer> = Box::new(Fixed);
        synth_worker(
            &mut backend,
            &mut Chain,
            Some(&mut pre),
            &plan,
            &audio_tx,
            &event_tx,
        )
        .unwrap();
        drop(audio_tx);
        drop(event_tx);
        let events: Vec<_> = event_rx.iter().collect();
        let words: Vec<&crate::WordTiming> = events
            .iter()
            .filter_map(|e| match e {
                SynthesisEvent::WordBoundary(w) => Some(w),
                _ => None,
            })
            .collect();
        assert_eq!(words.len(), 2, "pre-pass words did not synthesize");
        assert!(words.iter().all(|w| !w.estimated));
        let chunks: Vec<_> = audio_rx.iter().collect();
        assert!(
            !chunks.is_empty(),
            "no audio: pre-pass symbols were dropped"
        );
    }

    #[test]
    fn build_ids_matches_piper_layout() {
        struct Fixed;
        impl TokenPhonemizer for Fixed {
            fn phonemize_token(&mut self, t: &str) -> Vec<String> {
                match t {
                    "hello" => vec!["h".into(), "ɛ".into(), "l".into(), "o".into()],
                    "world" => vec!["w".into(), "ɜː".into(), "l".into(), "d".into()],
                    _ => vec![],
                }
            }
        }
        let m = map();
        let doc = parse_ssml("hello world").unwrap();
        let words: Vec<WordSpan> = doc
            .segments
            .iter()
            .flat_map(|s| match s {
                Segment::Words { words } => words.clone(),
                _ => Vec::new(),
            })
            .collect();
        let mut g2p = Fixed;
        let (ids, groups) = build_ids(&m, &ControlSymbols::piper(), &mut g2p, &words);
        // BOS PAD | h PAD ɛ PAD l PAD o PAD | space PAD | w PAD ɜː PAD l PAD d PAD | EOS
        assert_eq!(ids.len(), 2 + 8 + 2 + 8 + 1);
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(ids[1], 0); // PAD
        assert_eq!(*ids.last().unwrap(), 2); // EOS
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].start, 2);
        assert_eq!(groups[0].end, 10);
        assert_eq!(groups[1].start, 12);
        assert_eq!(groups[1].end, 20);
    }
}
