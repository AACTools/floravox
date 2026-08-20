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

/// Symbols piper-style models reserve for control ids.
const BOS: &str = "^";
const EOS: &str = "$";
const PAD: &str = "_";

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
}

/// One planned synthesis step.
enum PlanItem {
    /// A run of words sharing one rate.
    Words {
        words: Vec<WordSpan>,
        rate: f32,
        marks: Vec<String>,
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
            inner: Arc::new(Mutex::new(Inner { backend, g2p })),
        }
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
        let (audio_tx, audio_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();

        std::thread::spawn(move || {
            let Ok(mut inner) = inner.lock() else { return };
            let Inner { backend, g2p } = &mut *inner;
            let _ = synth_worker(backend.as_mut(), g2p, &plan, &audio_tx, &event_tx);
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
    let mut pending_marks: Vec<String> = Vec::new();

    let take_marks = |pending: &mut Vec<String>| std::mem::take(pending);

    for seg in &doc.segments {
        match seg {
            Segment::Words { words } => {
                if words.is_empty() {
                    continue;
                }
                let rate = words[0].prosody.rate.unwrap_or(1.0);
                match plan.last_mut() {
                    Some(PlanItem::Words {
                        words: w, rate: r, ..
                    }) if (*r - rate).abs() < 1e-6 => {
                        w.extend(words.iter().cloned());
                    }
                    _ => plan.push(PlanItem::Words {
                        words: words.clone(),
                        rate,
                        marks: take_marks(&mut pending_marks),
                    }),
                }
            }
            Segment::Break { ms, .. } => {
                // Marks waiting in front of a break fire at the break start.
                if !pending_marks.is_empty() {
                    push_or_attach(&mut plan, std::mem::take(&mut pending_marks));
                }
                plan.push(PlanItem::Break { ms: u64::from(*ms) });
            }
            Segment::Mark { name, .. } => pending_marks.push(name.clone()),
            Segment::SentenceEnd { .. } => {
                close_words(&mut plan);
                plan.push(PlanItem::SentenceEnd);
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
    Ok(plan)
}

/// End the current open Words item so the next one starts fresh.
fn close_words(_plan: &mut Vec<PlanItem>) {
    // Words items merge only when adjacent; inserting a boundary item
    // (SentenceEnd/ParagraphEnd) between them prevents merging.
}

/// Attach marks to the last item (or a synthetic empty one).
fn push_or_attach(plan: &mut Vec<PlanItem>, marks: Vec<String>) {
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
    plan: &[PlanItem],
    audio_tx: &mpsc::Sender<AudioChunk>,
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
                if words.is_empty() {
                    // Marks with no following speech fire at current offset.
                    for name in marks {
                        event_tx
                            .send(SynthesisEvent::MarkReached {
                                name: name.clone(),
                                sample: offset,
                                ms: offset * 1000 / u64::from(rate),
                            })
                            .ok();
                    }
                    continue;
                }

                let (ids, groups) = build_ids(&model.config().phoneme_id_map.clone(), g2p, words);
                if ids.is_empty() {
                    continue;
                }

                let length_scale = model.config().length_scale / unit_rate.max(0.1);
                let (audio, durations) = model.run(&ids, length_scale)?;
                let d_ok = durations.as_ref().is_some_and(|d| d.len() == ids.len());

                // Word timings (measured or estimated).
                let timings: Vec<WordTiming> = match &durations {
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

                // Marks fire at the unit's first word start.
                let mark_sample = durations.as_ref().map_or(offset, |d| {
                    offset + sample_at_id_index(d, groups[0].start, hop)
                });
                for name in marks {
                    event_tx
                        .send(SynthesisEvent::MarkReached {
                            name: name.clone(),
                            sample: mark_sample,
                            ms: mark_sample * 1000 / u64::from(rate),
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

/// Build the phoneme-id sequence for a word run.
///
/// Piper-style layout: `BOS, PAD, (phoneme ids…, PAD)*, EOS`, with
/// `space, PAD` between words. Control symbols missing from the map
/// (MMS alphabets have no `^`/`$`/space) are skipped instead of guessed,
/// so those models get a flat id sequence.
fn build_ids<G: TokenPhonemizer>(
    map: &HashMap<String, Vec<i64>>,
    g2p: &mut G,
    words: &[WordSpan],
) -> (Vec<i64>, Vec<std::ops::Range<usize>>) {
    let pad = map.get(PAD);
    let bos = map.get(BOS);
    let eos = map.get(EOS);
    let space = map.get(" ");

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
            if let Some(pid) = map.get(p.as_str()) {
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
            PlanItem::Words { marks, .. } => marks.iter().any(|m| m == "m"),
            _ => false,
        });
        assert!(attached);
    }

    #[test]
    fn plan_merges_same_rate_words() {
        let plan = plan_document(&map(), "hello world").unwrap();
        assert_eq!(plan.len(), 1);
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
        let (ids, groups) = build_ids(&m, &mut g2p, &words);
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
