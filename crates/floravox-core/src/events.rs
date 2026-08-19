//! Event types emitted during synthesis and playback.

use serde::{Deserialize, Serialize};

/// A single synchronization event with its position in the audio stream.
///
/// All sample positions are absolute from the start of the utterance,
/// including any `<break>` silence inserted between segments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SynthesisEvent {
    /// Speech audio is about to start.
    Started,
    /// A word begins/ends at the given positions.
    WordBoundary(WordTiming),
    /// An SSML `<mark name="..."/>` was reached.
    MarkReached {
        /// Mark name as supplied by the client.
        name: String,
        /// Absolute sample position where speech reaches this mark.
        sample: u64,
        /// Same position in milliseconds.
        ms: u64,
    },
    /// A `<break>` pause starts.
    BreakStarted {
        /// Pause length in milliseconds.
        ms: u64,
        /// Absolute sample where the pause begins.
        sample: u64,
    },
    /// A `<break>` pause ends.
    BreakEnded {
        /// Absolute sample where speech resumes.
        sample: u64,
    },
    /// A sentence (`</s>`) boundary was crossed.
    SentenceEnd {
        /// Absolute sample position.
        sample: u64,
    },
    /// A paragraph (`</p>`) boundary was crossed.
    ParagraphEnd {
        /// Absolute sample position.
        sample: u64,
    },
    /// Synthesis finished; no more audio or events follow.
    Finished {
        /// Total audio length in samples.
        total_samples: u64,
        /// Total audio length in milliseconds.
        total_ms: u64,
    },
}

/// Word boundary timing with full source mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordTiming {
    /// The word as written (highlight target).
    pub text: String,
    /// UTF-8 byte offset of the word in the source input.
    pub byte_offset: usize,
    /// Byte length of the word's raw source span.
    pub byte_len: usize,
    /// Character offset of the word in the source input.
    pub char_offset: usize,
    /// Character length of the word's raw source span.
    pub char_len: usize,
    /// Absolute sample where the word starts.
    pub sample_start: u64,
    /// Absolute sample where the word ends (exclusive).
    pub sample_end: u64,
    /// Start in milliseconds.
    pub ms_start: u64,
    /// End in milliseconds.
    pub ms_end: u64,
    /// True when timings come from the estimation fallback instead of the
    /// model's durations output (unpatched voice).
    pub estimated: bool,
}

impl SynthesisEvent {
    /// The absolute sample position this event fires at.
    #[must_use]
    pub fn sample(&self) -> u64 {
        match self {
            Self::Started => 0,
            Self::WordBoundary(w) => w.sample_start,
            Self::MarkReached { sample, .. }
            | Self::BreakStarted { sample, .. }
            | Self::BreakEnded { sample }
            | Self::SentenceEnd { sample }
            | Self::ParagraphEnd { sample } => *sample,
            Self::Finished { total_samples, .. } => *total_samples,
        }
    }

    /// The event position in milliseconds (rounded).
    #[must_use]
    pub fn ms(&self) -> u64 {
        match self {
            Self::WordBoundary(w) => w.ms_start,
            Self::MarkReached { ms, .. } | Self::Finished { total_ms: ms, .. } => *ms,
            Self::BreakStarted { ms, .. } => *ms,
            _ => 0,
        }
    }
}
