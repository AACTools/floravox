//! # floravox-core
//!
//! Orchestrates the floravox pipeline:
//!
//! 1. [`floravox_ssml`] parses input into word spans with exact source
//!    offsets.
//! 2. [`floravox_g2p`] converts words to phonemes.
//! 3. The ONNX acoustic model (feature `onnx`) emits audio plus a
//!    per-phoneme-id **durations** tensor.
//! 4. [`fold_word_timings`] folds durations back onto word spans, and
//!    [`EventTimeline`] lets a playback thread fire events at the exact
//!    sample it has reached — the primitive VoiceGarden-SPD needs for
//!    SSIP `700-SSIP EVENT INDEX-MARK` delivery.
//!
//! ```
//! use floravox_core::{SynthesisEvent, EventTimeline};
//!
//! let mut timeline = EventTimeline::new();
//! timeline.push(22050, SynthesisEvent::MarkReached { name: "m1".into(), sample: 22050, ms: 1000 });
//! // Playback thread has written 44100 samples:
//! let fired = timeline.drain_until(44100);
//! assert_eq!(fired.len(), 1);
//! ```

pub mod estimate;
pub mod events;
#[cfg(feature = "onnx")]
pub mod synth;
pub mod timeline;

pub use events::{SynthesisEvent, WordTiming};
pub use timeline::EventTimeline;

/// Fold a durations tensor (mel frames per phoneme-id) into sample-accurate
/// word timings.
///
/// `durations` is indexed exactly like the phoneme-id input sequence
/// (including any pad ids the model inserts). `groups` maps each word to its
/// id index range within that sequence.
///
/// Returns per-group `(sample_start, sample_end)` tuples.
#[must_use]
pub fn fold_word_timings(
    durations: &[i64],
    groups: &[std::ops::Range<usize>],
    hop_length: u32,
) -> Vec<(u64, u64)> {
    let mut prefix: Vec<u64> = Vec::with_capacity(durations.len() + 1);
    prefix.push(0);
    let mut acc = 0u64;
    for &d in durations {
        acc += u64::try_from(d.max(0))
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::from(hop_length));
        prefix.push(acc);
    }
    groups
        .iter()
        .map(|g| {
            let start_idx = g.start.min(prefix.len().saturating_sub(1));
            let end_idx = g.end.min(prefix.len().saturating_sub(1));
            (prefix[start_idx], prefix[end_idx])
        })
        .collect()
}

/// Frame position (in samples) of a specific id index — used for `<mark>`
/// placement between words.
#[must_use]
pub fn sample_at_id_index(durations: &[i64], index: usize, hop_length: u32) -> u64 {
    durations
        .iter()
        .take(index)
        .map(|&d| u64::try_from(d.max(0)).unwrap_or(u64::MAX))
        .sum::<u64>()
        * u64::from(hop_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_word_groups() {
        // 5 ids; word A owns ids 0..2, word B owns ids 3..5
        let durations = [10i64, 5, 5, 20, 10];
        let groups = vec![0..2, 2..5];
        let samples = fold_word_timings(&durations, &groups, 256);
        assert_eq!(samples[0], (0, (10 + 5) * 256));
        assert_eq!(samples[1], ((10 + 5) * 256, (10 + 5 + 5 + 20 + 10) * 256));
    }

    #[test]
    fn sample_at_index() {
        let durations = [10i64, 5, 5];
        assert_eq!(sample_at_id_index(&durations, 0, 256), 0);
        assert_eq!(sample_at_id_index(&durations, 2, 256), 15 * 256);
        assert_eq!(sample_at_id_index(&durations, 99, 256), 20 * 256);
    }
}
