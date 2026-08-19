//! Proportional timing fallback for voices without a durations output.
//!
//! When the ONNX model is a stock (unpatched) piper voice, audio arrives
//! without alignment data. This module distributes the observed audio length
//! across words weighted by phoneme count (or character count when no
//! phonemes are known) — the same spirit as rust-tts-wrapper's
//! `estimate_word_boundaries`, but anchored to the real utterance length so
//! the timeline neither over- nor under-runs.

use crate::events::WordTiming;
use floravox_ssml::WordSpan;

/// Compute word timings for `words` across `total_samples` of audio.
///
/// Weights: phoneme count when available, else character count, minimum 1.
#[must_use]
pub fn estimate_timings(
    words: &[WordSpan],
    total_samples: u64,
    sample_rate: u32,
) -> Vec<WordTiming> {
    if words.is_empty() {
        return Vec::new();
    }
    let weights: Vec<u64> = words
        .iter()
        .map(|w| {
            u64::try_from(
                w.phonemes
                    .as_ref()
                    .map_or_else(|| w.spoken.chars().count(), std::vec::Vec::len),
            )
            .unwrap_or(1)
            .max(1)
        })
        .collect();
    let total_weight: u64 = weights.iter().sum();
    let rate = u64::from(sample_rate.max(1));

    let mut out = Vec::with_capacity(words.len());
    let mut cursor = 0u64;
    for (i, w) in words.iter().enumerate() {
        let is_last = i + 1 == words.len();
        let len = if is_last {
            total_samples.saturating_sub(cursor)
        } else {
            (total_samples.saturating_mul(weights[i]) / total_weight.max(1)).max(1)
        };
        out.push(WordTiming {
            text: w.text.clone(),
            byte_offset: w.byte_span.start,
            byte_len: w.byte_len(),
            char_offset: w.char_span.start,
            char_len: w.char_len(),
            sample_start: cursor,
            sample_end: cursor + len,
            ms_start: cursor * 1000 / rate,
            ms_end: (cursor + len) * 1000 / rate,
            estimated: true,
        });
        cursor += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use floravox_ssml::Prosody;

    fn span(text: &str, n_phonemes: Option<usize>) -> WordSpan {
        WordSpan {
            text: text.into(),
            spoken: text.into(),
            char_span: 0..text.len(),
            byte_span: 0..text.len(),
            phonemes: n_phonemes.map(|n| (0..n).map(|i| i.to_string()).collect()),
            prosody: Prosody::default(),
            say_as: floravox_ssml::SayAs::None,
            voice: None,
        }
    }

    #[test]
    fn covers_full_range() {
        let words = vec![span("aa", None), span("bbbb", None), span("cc", None)];
        let timings = estimate_timings(&words, 22050, 22050);
        assert_eq!(timings[0].sample_start, 0);
        assert_eq!(timings.last().unwrap().sample_end, 22050);
        for pair in timings.windows(2) {
            assert_eq!(pair[0].sample_end, pair[1].sample_start);
        }
        assert!(timings.iter().all(|t| t.estimated));
    }

    #[test]
    fn phoneme_weights_beat_char_weights() {
        let words = vec![span("long", Some(10)), span("short", Some(2))];
        let timings = estimate_timings(&words, 1200, 22050);
        let a = timings[0].sample_end - timings[0].sample_start;
        let b = timings[1].sample_end - timings[1].sample_start;
        assert!(a > b);
    }
}
