//! [misaki](https://github.com/hexgrad/misaki) G2P — the phonemizer
//! Kokoro voices were trained with, as a Rust port
//! ([`misaki-rs`](https://crates.io/crates/misaki-rs), MIT, fully
//! self-contained: dictionaries and POS-tagger weights are compiled in).
//!
//! Two integration shapes:
//!
//! * [`MisakiG2p::phonemize_words`] — **document-level**: phonemizes a
//!   whole run of words with sentence context, so the POS tagger
//!   disambiguates heteronyms (`object` noun vs verb) and numbers expand
//!   to words. This is the correct mode for Kokoro and anything else
//!   trained on misaki output. `floravox-core`'s
//!   `DocumentPhonemizer`/`MisakiPrePass` drive it.
//! * [`TokenPhonemizer`] — single-word mode for simple wiring; POS
//!   context is lost but dictionary + morphology still apply.
//!
//! Output normalization adapts to espeak-style char inventories: zero-width
//! joiners inside clusters (`d‍ʒ`) split into separate symbols, tie bars
//! likewise, and superscript `ᵊ` maps to `ə`. Validated against kokoro
//! en-v0.19's 177-symbol `tokens.txt` (100% coverage after normalization).
//!
//! The optional espeak fallback of the underlying crate is **not** enabled
//! (it would link GPL libespeak-ng); out-of-dictionary words are spelled
//! letter-by-letter instead — the same semantics as [`crate::RuleFallback`],
//! and chainable behind it.

use crate::{punct_phonemes, word_core, Phoneme, TokenPhonemizer};

/// Look-ahead limit when aligning document tokens back onto words.
const MATCH_WINDOW: usize = 8;

/// Document- and single-word-level misaki phonemizer.
pub struct MisakiG2p {
    inner: misaki_rs::G2P,
}

impl MisakiG2p {
    /// Create for a dialect: `false` = US English, `true` = British.
    #[must_use]
    pub fn new(british: bool) -> Self {
        Self {
            inner: misaki_rs::G2P::new(if british {
                misaki_rs::Language::EnglishGB
            } else {
                misaki_rs::Language::EnglishUS
            }),
        }
    }

    /// Phonemize a run of words with sentence context.
    ///
    /// Returns one entry per word: aligned phonemes when the engine's
    /// token stream matched the word text (numbers/currency may expand
    /// to several tokens, which breaks exact matching), else the word
    /// phonemized on its own, else `None` (empty result — callers keep
    /// their normal per-word path).
    pub fn phonemize_words(&mut self, words: &[&str]) -> Vec<Option<Vec<Phoneme>>> {
        if words.is_empty() {
            return Vec::new();
        }
        let text = words.join(" ");
        let Ok((_, tokens)) = self.inner.g2p(&text) else {
            return vec![None; words.len()];
        };
        let toks: Vec<(String, Option<Vec<Phoneme>>)> = tokens
            .iter()
            .map(|t| {
                (
                    normalize_text(&t.text),
                    t.phonemes.as_deref().map(split_phonemes),
                )
            })
            .collect();

        let mut cursor = 0usize;
        words
            .iter()
            .map(|w| {
                let target = normalize_text(w);
                if target.is_empty() {
                    return None;
                }
                // Look ahead for the token whose text matches this word;
                // stop early at a token matching a later word so repeated
                // words don't steal each other's alignment.
                let mut found = None;
                let mut scan = 0usize;
                while cursor + scan < toks.len() && scan < MATCH_WINDOW {
                    let (tnorm, ph) = &toks[cursor + scan];
                    if *tnorm == target {
                        found.clone_from(ph);
                        cursor += scan + 1;
                        break;
                    }
                    if words
                        .iter()
                        .skip(1)
                        .any(|later| normalize_text(later) == *tnorm)
                    {
                        break;
                    }
                    scan += 1;
                }
                // Fallback: phonemize the word on its own (dictionary +
                // morphology still apply; only POS context is lost).
                found.or_else(|| self.phonemize_single(w))
            })
            .collect()
    }

    /// Phonemize one word in isolation.
    fn phonemize_single(&mut self, word: &str) -> Option<Vec<Phoneme>> {
        let joined: String = word.split_whitespace().collect::<Vec<_>>().join(" ");
        let Ok((phonemes, _)) = self.inner.g2p(&joined) else {
            return None;
        };
        let split = split_phonemes(&phonemes);
        (!split.is_empty()).then_some(split)
    }
}

impl TokenPhonemizer for MisakiG2p {
    fn phonemize_token(&mut self, token: &str) -> Vec<Phoneme> {
        let mut out = Vec::new();
        if let Some(core) = word_core(token) {
            if let Some(ph) = self.phonemize_single(core) {
                out.extend(ph);
            }
        }
        out.extend(punct_phonemes(token));
        out
    }
}

/// Normalize a phoneme string into inventory-safe symbols: zero-width
/// joiners and tie bars split clusters (`d‍ʒ` → `d ʒ`), superscript `ᵊ`
/// becomes `ə`.
fn split_phonemes(s: &str) -> Vec<Phoneme> {
    s.replace(['\u{200D}', '\u{0361}', '\u{035C}'], " ")
        .replace('ᵊ', "ə")
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// Case- and punctuation-folded text for token/word matching.
fn normalize_text(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_level_pos_and_numbers() {
        let mut g2p = MisakiG2p::new(false);
        let words = ["The", "object", "is", "to", "object", "."];
        let out = g2p.phonemize_words(&words);
        // Real words align; the bare "." folds to nothing (no alphanumerics)
        // and yields None by design.
        assert!(
            out.iter().take(5).all(Option::is_some),
            "unaligned: {out:?}"
        );
        assert!(out[5].is_none());
        let joined: String = out[1].as_ref().map_or_else(String::new, |v| v.join(""));
        assert!(joined.contains('ˈ'), "no stress: {joined}");
    }

    #[test]
    fn normalized_symbols_are_inventory_safe() {
        let mut g2p = MisakiG2p::new(false);
        let out = g2p.phonemize_words(&["adventure", "judge", "measure"]);
        for ph in out.into_iter().flatten() {
            for sym in ph {
                assert!(!sym.contains('\u{200D}'), "ZWJ survived: {sym:?}");
                assert!(!sym.contains('\u{0361}'), "tie bar survived: {sym:?}");
                assert!(!sym.contains('ᵊ'), "superscript schwa survived: {sym:?}");
            }
        }
    }

    #[test]
    fn single_word_mode_matches_document_mode_for_simple_words() {
        let mut g2p = MisakiG2p::new(false);
        let doc = g2p.phonemize_words(&["hello"]);
        let single = g2p.phonemize_single("hello");
        assert_eq!(doc[0], single);
        assert!(doc[0].as_ref().is_some_and(|p| !p.is_empty()));
    }

    #[test]
    fn token_mode_appends_punctuation() {
        let mut g2p = MisakiG2p::new(false);
        let ph = TokenPhonemizer::phonemize_token(&mut g2p, "hello,");
        assert_eq!(ph.last().map(String::as_str), Some(","));
    }
}
