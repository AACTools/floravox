//! Ingest third-party pronunciation data into lexicon rows.
//!
//! Three source formats are understood:
//!
//! * **CMUDICT** — `WORD  P HH R AH1 N` (whitespace-separated ARPABET with
//!   stress digits). Converted to IPA targeting the `piper/espeak en_US`
//!   phoneme inventory: `AH0`/`ER0` reduce to `ə`/`ɚ`, other stress marks
//!   become standalone `ˈ`/`ˌ` symbols, and affricates map to `tʃ`/`dʒ`.
//! * **IPA TSV** — `word\tIPA` with unsegmented IPA (`WikiPron`
//!   downloads, `gruut` extractions). The right-hand side is clustered
//!   into phoneme symbols by [`ipa_tokens`].
//! * **TSV** — `word\tph1 ph2 ph3` with pre-segmented phonemes (the native
//!   format of [`crate::FstLexicon::from_tsv`], tolerated variant).
//!
//! All parsers are tolerant: malformed lines are counted in
//! [`Ingested::skipped`] (and unmapped ARPABET symbols in
//! [`Ingested::unknown`]) instead of aborting the whole file.
//!
//! ```
//! use floravox_g2p::ingest;
//!
//! let ing = ingest::parse(
//!     "HELLO  HH AH0 L OW1\n",
//!     ingest::SourceFormat::CmuDict,
//! );
//! assert_eq!(ing.rows, vec![("HELLO".into(), "h ə l ˈ oʊ".into())]);
//! ```

use crate::Phoneme;

/// Source lexicon format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    /// CMUDICT: `WORD  P HH R AH1 N` (ARPABET + stress digits).
    CmuDict,
    /// `word\tIPA` with unsegmented IPA (`WikiPron`, `gruut` extractions).
    IpaTsv,
    /// `word\tph1 ph2 ph3` with pre-segmented phonemes.
    Tsv,
}

impl SourceFormat {
    /// Guess the format of `text` by scanning its first non-comment lines.
    /// TSV is the fallback when nothing more specific matches.
    #[must_use]
    pub fn detect(text: &str) -> Self {
        let mut cmudict = 0usize;
        let mut ipa = 0usize;
        let mut other = 0usize;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            if line_is_cmudict(line) {
                cmudict += 1;
            } else if let Some((_, right)) = line.split_once('\t') {
                // Unsegmented IPA (`WikiPron` style) has no spaces on the
                // right-hand side; pre-segmented TSV phonemes do.
                if !right.is_ascii() && !right.chars().any(char::is_whitespace)
                {
                    ipa += 1;
                } else {
                    other += 1;
                }
            } else {
                other += 1;
            }
            if cmudict + ipa + other >= 2000 {
                break;
            }
        }
        if cmudict > 0 && cmudict >= ipa {
            Self::CmuDict
        } else if ipa > 0 {
            Self::IpaTsv
        } else {
            Self::Tsv
        }
    }
}

/// Result of ingesting a source file.
#[derive(Debug, Default)]
pub struct Ingested {
    /// `(word, space-separated phonemes)` rows, ready for
    /// [`crate::LexiconWriter::write`].
    pub rows: Vec<(String, String)>,
    /// Malformed lines (empty word side, no phonemes, missing tab).
    pub skipped: usize,
    /// CMUDICT lines dropped because they contain unmapped ARPABET symbols.
    pub unknown: usize,
}

/// Parse `text` according to `format`.
#[must_use]
pub fn parse(text: &str, format: SourceFormat) -> Ingested {
    match format {
        SourceFormat::CmuDict => parse_cmudict(text),
        SourceFormat::IpaTsv => parse_ipa_tsv(text),
        SourceFormat::Tsv => parse_tsv(text),
    }
}

/// Parse CMUDICT text (`WORD  P HH R AH1 N` per line, `;;;` comments,
/// `WORD(2)` variant suffixes).
#[must_use]
pub fn parse_cmudict(text: &str) -> Ingested {
    let mut ing = Ingested::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let Some(word) = fields.next() else {
            ing.skipped += 1;
            continue;
        };
        let phones: Vec<&str> = fields.collect();
        if phones.is_empty() {
            ing.skipped += 1;
            continue;
        }
        let word = strip_variant(word);
        match cmudict_phonemes(&phones) {
            Some(ph) => ing.rows.push((word.to_owned(), ph.join(" "))),
            None => ing.unknown += 1,
        }
    }
    ing
}

/// Parse `word\tIPA` text with unsegmented IPA on the right-hand side.
#[must_use]
pub fn parse_ipa_tsv(text: &str) -> Ingested {
    let mut ing = Ingested::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((word, ipa)) = line.split_once('\t') else {
            ing.skipped += 1;
            continue;
        };
        let tokens = ipa_tokens(ipa.trim());
        if word.trim().is_empty() || tokens.is_empty() {
            ing.skipped += 1;
            continue;
        }
        ing.rows.push((word.trim().to_owned(), tokens.join(" ")));
    }
    ing
}

/// Parse `word\tph1 ph2` text with pre-segmented phonemes (`#` comments,
/// blank lines, and malformed lines are counted and skipped).
#[must_use]
pub fn parse_tsv(text: &str) -> Ingested {
    let mut ing = Ingested::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match line.split_once('\t') {
            Some((w, p)) if !w.trim().is_empty() && !p.trim().is_empty() => {
                ing.rows.push((w.trim().to_owned(), p.trim().to_owned()));
            }
            _ => ing.skipped += 1,
        }
    }
    ing
}

/// Convert one ARPABET symbol (without stress digit) to IPA.
///
/// The mapping targets the `piper/espeak en_US` inventory (`ɹ` for R,
/// diphthongs as single symbols like `eɪ`/`oʊ`). `AH`/`ER` stress
/// reductions are handled separately in [`parse_cmudict`].
#[must_use]
pub fn arpabet_to_ipa(phone: &str) -> Option<&'static str> {
    Some(match phone {
        "AA" => "ɑ",
        "AE" => "æ",
        "AH" => "ʌ",
        "AO" => "ɔ",
        "AW" => "aʊ",
        "AY" => "aɪ",
        "B" => "b",
        "CH" => "tʃ",
        "D" => "d",
        "DH" => "ð",
        "EH" => "ɛ",
        "ER" => "ɝ",
        "EY" => "eɪ",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "IH" => "ɪ",
        "IY" => "i",
        "JH" => "dʒ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "OW" => "oʊ",
        "OY" => "ɔɪ",
        "P" => "p",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "UH" => "ʊ",
        "UW" => "u",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        _ => return None,
    })
}

/// Segment an unsegmented IPA string into phoneme symbols.
///
/// * Base characters start a symbol; combining marks (U+0300–U+036F) and
///   IPA modifier letters (U+02B0–U+02FF, e.g. `ː` `ʰ`) attach to the
///   previous symbol.
/// * `ˈ` `ˌ` `.` are emitted as standalone prosody symbols, matching the
///   piper phoneme convention.
/// * English diphthongs (`aɪ aʊ eɪ oʊ ɔɪ`) are merged into single symbols.
/// * Anything else (spaces, punctuation, tie bars) is dropped.
#[must_use]
pub fn ipa_tokens(ipa: &str) -> Vec<Phoneme> {
    let mut clustered: Vec<Phoneme> = Vec::new();
    // Set while a tie bar (U+0361/U+035C) is pending: the next base
    // character joins the previous symbol ("t͡ʃ" → one symbol).
    let mut glued = false;
    for ch in ipa.chars() {
        if matches!(ch, 'ˈ' | 'ˌ' | '.') {
            clustered.push(ch.to_string());
            glued = false;
        } else if is_tie_bar(ch) {
            if let Some(last) = clustered.last_mut() {
                last.push(ch);
                glued = true;
            }
        } else if is_ipa_modifier(ch) {
            if let Some(last) = clustered.last_mut() {
                last.push(ch);
            }
            glued = false;
        } else if ch.is_alphabetic() {
            if glued {
                if let Some(last) = clustered.last_mut() {
                    last.push(ch);
                    glued = false;
                    continue;
                }
            }
            clustered.push(ch.to_string());
        } else {
            glued = false;
        }
    }
    merge_diphthongs(clustered)
}

/// True for IPA tie bars gluing two bases into one affricate symbol.
fn is_tie_bar(ch: char) -> bool {
    matches!(u32::from(ch), 0x0361 | 0x035C)
}

/// True for IPA modifier letters and combining diacritics that attach to the
/// preceding symbol (length `ː`/`ˑ`, aspiration `ʰ`, velarization `ˠ`, ...).
/// The stress marks `ˈ`/`ˌ` live in this range too but are matched earlier.
fn is_ipa_modifier(ch: char) -> bool {
    matches!(u32::from(ch), 0x02B0..=0x02FF | 0x0300..=0x036F | 0x207F)
}

/// Adjacent symbol pairs kept as single diphthong symbols (`piper en_US`).
const DIPHTHONGS: [&str; 5] = ["aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ"];

fn merge_diphthongs(tokens: Vec<Phoneme>) -> Vec<Phoneme> {
    let mut merged: Vec<Phoneme> = Vec::with_capacity(tokens.len());
    let mut iter = tokens.into_iter().peekable();
    while let Some(tok) = iter.next() {
        if let Some(next) = iter.peek() {
            let pair = format!("{tok}{next}");
            if DIPHTHONGS.contains(&pair.as_str()) {
                merged.push(pair);
                iter.next();
                continue;
            }
        }
        merged.push(tok);
    }
    merged
}

/// `WORD(2)` → `WORD` (CMUDICT pronunciation-variant suffix).
fn strip_variant(word: &str) -> &str {
    if word.ends_with(')') {
        if let Some(open) = word.rfind('(') {
            return &word[..open];
        }
    }
    word
}

/// Convert one line's ARPABET symbols (with optional stress digits) to IPA.
/// Returns `None` when any symbol is unmapped.
fn cmudict_phonemes(phones: &[&str]) -> Option<Vec<Phoneme>> {
    let mut out: Vec<Phoneme> = Vec::with_capacity(phones.len() + 2);
    for phone in phones {
        let (base, stress) = split_stress(phone);
        match base {
            // Unstressed schwa variants: AH0 → ə, ER0 → ɚ.
            "AH" => out.push(if stress == Some(0) { "ə" } else { "ʌ" }.into()),
            "ER" => out.push(if stress == Some(0) { "ɚ" } else { "ɝ" }.into()),
            other => {
                let ipa = arpabet_to_ipa(other)?;
                match stress {
                    Some(1) => out.push("ˈ".into()),
                    Some(2) => out.push("ˌ".into()),
                    _ => {}
                }
                out.push(ipa.into());
            }
        }
    }
    Some(out)
}

/// `AH1` → `("AH", Some(1))`, `T` → `("T", None)`.
fn split_stress(phone: &str) -> (&str, Option<u8>) {
    match phone.as_bytes().last() {
        Some(b'0') => (&phone[..phone.len() - 1], Some(0)),
        Some(b'1') => (&phone[..phone.len() - 1], Some(1)),
        Some(b'2') => (&phone[..phone.len() - 1], Some(2)),
        _ => (phone, None),
    }
}

/// True when a line looks like `WORD(1)  P HH R AH1 N` with every phone in
/// the ARPABET inventory (used for format auto-detection).
fn line_is_cmudict(line: &str) -> bool {
    let mut fields = line.split_whitespace();
    let Some(word) = fields.next() else {
        return false;
    };
    let word = strip_variant(word);
    if word.is_empty()
        || !word
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '\'' || c == '-' || c == '.')
    {
        return false;
    }
    let mut phones = 0;
    for phone in fields {
        let (base, _) = split_stress(phone);
        if arpabet_to_ipa(base).is_none() {
            return false;
        }
        phones += 1;
    }
    phones > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FstLexicon, LexiconPhonemizer, RuleFallback};

    const CMUDICT_SAMPLE: &str = "\
;;; Canonical CMUDict Format
HELLO  HH AH0 L OW1
WATER  W AO1 T ER0
WATER(1)  W AA1 T ER0
CONSTRUCTOR  K AH0 N S T R AH1 K T ER0
'ROUND  R AW1 N D
GARBAGE LINE ZK XQ
NOTABWORD
";

    #[test]
    fn cmudict_basics() {
        let ing = parse_cmudict(CMUDICT_SAMPLE);
        assert_eq!(ing.rows.len(), 5);
        assert_eq!(ing.unknown, 1); // ZK XQ
        assert_eq!(ing.skipped, 1); // no phonemes
        assert_eq!(ing.rows[0], ("HELLO".into(), "h ə l ˈ oʊ".into()));
        // Variant suffix stripped; both WATER rows collapse to one entry.
        assert_eq!(ing.rows[1], ("WATER".into(), "w ˈ ɔ t ɚ".into()));
        assert_eq!(ing.rows[2], ("WATER".into(), "w ˈ ɑ t ɚ".into()));
        // Apostrophe kept, primary stress mark, AH0 reduction.
        assert_eq!(ing.rows[4], ("'ROUND".into(), "ɹ ˈ aʊ n d".into()));
    }

    #[test]
    fn cmudict_stress_reductions() {
        let ing = parse_cmudict("BUTTER  B AH1 T ER2\n");
        assert_eq!(ing.rows[0].1, "b ʌ t ɝ");
        let ing = parse_cmudict("AGAIN  AH0 G EH1 N\n");
        assert_eq!(ing.rows[0].1, "ə ɡ ˈ ɛ n");
    }

    #[test]
    fn cmudict_roundtrip_through_lexicon() {
        let ing = parse_cmudict(CMUDICT_SAMPLE);
        let lex = FstLexicon::from_rows(ing.rows).unwrap();
        let mut g2p = LexiconPhonemizer::new(lex, RuleFallback::default());
        assert_eq!(g2p.phonemize_word("hello"), vec!["h", "ə", "l", "ˈ", "oʊ"]);
        // Variant rows: last wins (AA pronunciation of water).
        assert_eq!(g2p.phonemize_word("WATER"), vec!["w", "ˈ", "ɑ", "t", "ɚ"]);
    }

    #[test]
    fn ipa_tokens_clustering() {
        assert_eq!(ipa_tokens("həˈloʊ"), ["h", "ə", "ˈ", "l", "oʊ"]);
        assert_eq!(ipa_tokens("t͡ʃɛs"), ["t͡ʃ", "ɛ", "s"]);
        assert_eq!(ipa_tokens("biː"), ["b", "iː"]);
        assert_eq!(ipa_tokens("ɔɪ"), ["ɔɪ"]);
        assert_eq!(ipa_tokens("ˈɑː.tɚ"), ["ˈ", "ɑː", ".", "t", "ɚ"]);
        assert!(ipa_tokens(" , — !").is_empty());
    }

    #[test]
    fn ipa_tsv_parse() {
        let ing = parse_ipa_tsv("hello\thəˈloʊ\nworld\twɜːld\n# comment\nbadline\n");
        assert_eq!(ing.skipped, 1);
        assert_eq!(ing.rows[0], ("hello".into(), "h ə ˈ l oʊ".into()));
        assert_eq!(ing.rows[1], ("world".into(), "w ɜː l d".into()));
    }

    #[test]
    fn tsv_parse_counts_bad_lines() {
        let ing = parse_tsv("hello\th ə l\n# c\nno-tab-line\n\nx\t\n");
        assert_eq!(ing.rows, vec![("hello".into(), "h ə l".into())]);
        assert_eq!(ing.skipped, 2);
    }

    #[test]
    fn detect_formats() {
        assert_eq!(SourceFormat::detect(CMUDICT_SAMPLE), SourceFormat::CmuDict);
        assert_eq!(
            SourceFormat::detect("hello\thəˈloʊ\nworld\twɜːld\n"),
            SourceFormat::IpaTsv
        );
        assert_eq!(
            SourceFormat::detect("hello\th ə l o u\nworld\tw ɜː l d\n"),
            SourceFormat::Tsv
        );
    }

    #[test]
    fn parse_dispatch() {
        let via_dispatch = parse(CMUDICT_SAMPLE, SourceFormat::CmuDict);
        assert_eq!(via_dispatch.rows.len(), 5);
    }
}
