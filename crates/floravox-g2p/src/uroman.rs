//! Romanization of any script to Latin, ported from
//! [`uroman`](https://github.com/isi-nlp/uroman) (Apache-2.0, USC/ISI,
//! Ulf Hermjakob). Vendored data files live in `data/uroman/`.
//!
//! This is the table-driven core: longest-match over the manual
//! `romanization-table.txt` rules (language-specific rules preferred),
//! then per-codepoint mappings from `romanization-auto-table.txt`
//! (generated from `UnicodeData`), then passthrough for Latin, digits,
//! and punctuation. This reproduces the classic `uroman` behavior used
//! by MMS; the 2024 rewrite's lattice scoring (mainly benefiting CJK
//! quality via bigram frequencies and reading tables) is not ported —
//! CJK comes out through the auto table without pinyin readings.
//!
//! ```
//! use floravox_g2p::uroman::romanize;
//!
//! assert_eq!(romanize("привет", None), "privet");
//! assert_eq!(romanize("नमस्ते", None), "namaste");
//! assert_eq!(romanize("hello", None), "hello");
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

/// One manual-table rule: source string, target, optional language.
struct Rule {
    target: String,
    /// `::lcode`-restricted rule (e.g. `tur`); universal rules have none.
    lcode: Option<String>,
}

/// Consonant letters of Brahmi-derived (abugida) scripts whose bare
/// form carries an inherent "a" (Devanagari 0915-0939, Bengali,
/// Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam,
/// Sinhala, Myanmar, Khmer).
const ABUGIDA_CONSONANTS: &[(u32, u32)] = &[
    (0x0915, 0x0939),
    (0x0958, 0x095F),
    (0x0979, 0x097F),
    (0x0995, 0x09B9),
    (0x09DC, 0x09DF),
    (0x0A15, 0x0A39),
    (0x0A95, 0x0AB9),
    (0x0B15, 0x0B39),
    (0x0B5C, 0x0B5F),
    (0x0B95, 0x0BB9),
    (0x0C15, 0x0C39),
    (0x0C95, 0x0CB9),
    (0x0D15, 0x0D39),
    (0x0D9A, 0x0DC6),
    (0x1000, 0x1021),
    (0x1780, 0x17A2),
];

/// Vowel signs (matras) and viramas: when one follows an abugida
/// consonant, the inherent "a" is suppressed (the matra supplies the
/// vowel, or the virama devocalizes the consonant).
const ABUGIDA_VOWEL_SIGNS: &[(u32, u32)] = &[
    (0x093A, 0x094D),
    (0x0962, 0x0963),
    (0x09BE, 0x09CD),
    (0x09E2, 0x09E3),
    (0x0A3E, 0x0A4D),
    (0x0ABE, 0x0ACD),
    (0x0B3E, 0x0B4D),
    (0x0BBE, 0x0BCC),
    (0x0BCD, 0x0BCD),
    (0x0C3E, 0x0C4D),
    (0x0CBE, 0x0CCD),
    (0x0D3E, 0x0D4D),
    (0x0DCA, 0x0DDF),
    (0x102B, 0x1032),
    (0x1039, 0x103E),
    (0x17B6, 0x17C5),
    (0x17D2, 0x17D2),
];

fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|&(a, b)| a <= cp && cp <= b)
}

/// Unmapped combining marks (Mn), absent from the uroman tables;
/// the reference implementation drops them (verified: decomposed
/// e+U+0301 -> "e", lone virama -> "").
const MARK_DROP_RANGES: &[(u32, u32)] = &[
    (0x0300, 0x0310),
    (0x0313, 0x0313),
    (0x0315, 0x031B),
    (0x031D, 0x0326),
    (0x0328, 0x032A),
    (0x032C, 0x032E),
    (0x0330, 0x0331),
    (0x0334, 0x0338),
    (0x033B, 0x033E),
    (0x0340, 0x0342),
    (0x0344, 0x0344),
    (0x0346, 0x0350),
    (0x0352, 0x0356),
    (0x0358, 0x0359),
    (0x035B, 0x035B),
    (0x0362, 0x0362),
    (0x0484, 0x0486),
    (0x0591, 0x05AF),
    (0x05BA, 0x05BA),
    (0x05BC, 0x05BD),
    (0x05BF, 0x05BF),
    (0x05C1, 0x05C2),
    (0x05C4, 0x05C5),
    (0x05C7, 0x05C7),
    (0x0610, 0x0610),
    (0x0612, 0x0614),
    (0x0618, 0x061A),
    (0x0658, 0x0658),
    (0x065D, 0x065E),
    (0x06DB, 0x06DB),
    (0x06E0, 0x06E0),
    (0x0711, 0x0711),
    (0x0731, 0x0731),
    (0x073B, 0x073B),
    (0x0740, 0x0746),
    (0x0748, 0x074A),
    (0x07ED, 0x07EE),
    (0x07F1, 0x07F2),
    (0x0816, 0x0819),
    (0x081B, 0x081B),
    (0x082D, 0x082D),
    (0x0859, 0x085B),
    (0x0897, 0x089F),
    (0x08CA, 0x08D2),
    (0x08E3, 0x08EB),
    (0x08ED, 0x08EE),
    (0x08F0, 0x08F2),
    (0x08F4, 0x08FE),
    (0x0900, 0x0901),
    (0x093C, 0x093C),
    (0x094D, 0x094D),
    (0x0951, 0x0954),
    (0x0981, 0x0981),
    (0x09BC, 0x09BC),
    (0x09CD, 0x09CD),
    (0x0A01, 0x0A01),
    (0x0A3C, 0x0A3C),
    (0x0A4D, 0x0A4D),
    (0x0A51, 0x0A51),
    (0x0A75, 0x0A75),
    (0x0A81, 0x0A81),
    (0x0ABC, 0x0ABC),
    (0x0ACD, 0x0ACD),
    (0x0AFA, 0x0AFF),
    (0x0B01, 0x0B01),
    (0x0B3C, 0x0B3C),
    (0x0B4D, 0x0B4D),
    (0x0B55, 0x0B56),
    (0x0BCD, 0x0BCD),
    (0x0C00, 0x0C00),
    (0x0C3C, 0x0C3C),
    (0x0C4D, 0x0C4D),
    (0x0C55, 0x0C56),
    (0x0C81, 0x0C81),
    (0x0CBC, 0x0CBC),
    (0x0CCD, 0x0CCD),
    (0x0D01, 0x0D01),
    (0x0D3B, 0x0D3C),
    (0x0D4D, 0x0D4D),
    (0x0D81, 0x0D81),
    (0x0DCA, 0x0DCA),
    (0x0DD2, 0x0DD4),
    (0x0DD6, 0x0DD6),
    (0x0EC8, 0x0ECC),
    (0x0ECE, 0x0ECE),
    (0x0F18, 0x0F19),
    (0x0F35, 0x0F35),
    (0x0F37, 0x0F37),
    (0x0F39, 0x0F39),
    (0x0F7E, 0x0F7E),
    (0x0F82, 0x0F84),
    (0x0F86, 0x0F87),
    (0x0F8D, 0x0F8E),
    (0x0FBA, 0x0FBC),
    (0x0FC6, 0x0FC6),
    (0x1037, 0x1037),
    (0x1039, 0x103A),
    (0x1071, 0x1071),
    (0x108D, 0x108D),
    (0x135D, 0x135F),
    (0x1714, 0x1714),
    (0x17B4, 0x17B5),
    (0x17C9, 0x17CB),
    (0x17CE, 0x17CE),
    (0x17D1, 0x17D3),
    (0x17DD, 0x17DD),
    (0x180B, 0x180D),
    (0x180F, 0x180F),
    (0x1939, 0x193B),
    (0x1A58, 0x1A58),
    (0x1A60, 0x1A60),
    (0x1A74, 0x1A7C),
    (0x1AB0, 0x1ABD),
    (0x1AC1, 0x1ACE),
    (0x1B00, 0x1B03),
    (0x1B34, 0x1B34),
    (0x1B6B, 0x1B73),
    (0x1B80, 0x1B81),
    (0x1BA9, 0x1BA9),
    (0x1BAB, 0x1BAD),
    (0x1BE6, 0x1BE6),
    (0x1C36, 0x1C37),
    (0x1CD0, 0x1CD2),
    (0x1CD4, 0x1CD9),
    (0x1CDB, 0x1CE0),
    (0x1CE2, 0x1CE8),
    (0x1CED, 0x1CED),
    (0x1CF4, 0x1CF4),
    (0x1CF8, 0x1CF8),
    (0x1DC0, 0x1DC9),
    (0x1DCB, 0x1DD2),
    (0x1DF5, 0x1DF5),
    (0x1DF8, 0x1DFB),
    (0x1DFD, 0x1DFF),
    (0x20D0, 0x20DC),
    (0x20E1, 0x20E1),
    (0x20E5, 0x20E8),
    (0x20EA, 0x20F0),
    (0x2CEF, 0x2CF1),
    (0x2DF9, 0x2DF9),
    (0x302A, 0x302D),
    (0x3099, 0x309A),
    (0xA678, 0xA678),
    (0xA6F0, 0xA6F1),
    (0xA802, 0xA802),
    (0xA806, 0xA806),
    (0xA8C4, 0xA8C5),
    (0xA8E0, 0xA8E9),
    (0xA8F1, 0xA8F1),
    (0xA92B, 0xA92D),
    (0xA980, 0xA982),
    (0xA9B3, 0xA9B3),
    (0xA9B7, 0xA9B7),
    (0xA9B9, 0xA9B9),
    (0xAAB0, 0xAAB0),
    (0xAAB7, 0xAAB7),
    (0xAABF, 0xAABF),
    (0xAAC1, 0xAAC1),
    (0xAAF6, 0xAAF6),
    (0xABED, 0xABED),
    (0xFB1E, 0xFB1E),
    (0xFE00, 0xFE0F),
    (0xFE21, 0xFE26),
    (0xFE28, 0xFE2F),
    (0x101FD, 0x101FD),
    (0x102E0, 0x102E0),
    (0x10A0C, 0x10A0C),
    (0x10A0F, 0x10A0F),
    (0x10A38, 0x10A3A),
    (0x10A3F, 0x10A3F),
    (0x10AE5, 0x10AE6),
    (0x10D24, 0x10D27),
    (0x10D69, 0x10D6D),
    (0x10EAB, 0x10EAC),
    (0x10EFC, 0x10EFF),
    (0x10F46, 0x10F4D),
    (0x10F4F, 0x10F50),
    (0x10F82, 0x10F85),
    (0x11046, 0x11046),
    (0x11070, 0x11070),
    (0x11073, 0x11074),
    (0x1107F, 0x11080),
    (0x110B9, 0x110BA),
    (0x110C2, 0x110C2),
    (0x11100, 0x11100),
    (0x11102, 0x11102),
    (0x11131, 0x11134),
    (0x11173, 0x11173),
    (0x11180, 0x11180),
    (0x111C9, 0x111CC),
    (0x111CF, 0x111CF),
    (0x11236, 0x11237),
    (0x1123E, 0x1123E),
    (0x11241, 0x11241),
    (0x112E9, 0x112EA),
    (0x11301, 0x11301),
    (0x1133B, 0x1133C),
    (0x11366, 0x1136C),
    (0x113BB, 0x113C0),
    (0x113CE, 0x113CE),
    (0x113D0, 0x113D0),
    (0x113D2, 0x113D2),
    (0x113E1, 0x113E2),
    (0x11442, 0x11443),
    (0x11446, 0x11446),
    (0x1145E, 0x1145E),
    (0x114BF, 0x114BF),
    (0x114C2, 0x114C3),
    (0x115BC, 0x115BC),
    (0x115BF, 0x115C0),
    (0x1163F, 0x11640),
    (0x116B7, 0x116B7),
    (0x1171F, 0x1171F),
    (0x1172B, 0x1172B),
    (0x11839, 0x1183A),
    (0x1193C, 0x1193C),
    (0x1193E, 0x1193E),
    (0x11943, 0x11943),
    (0x119E0, 0x119E0),
    (0x11A0A, 0x11A0A),
    (0x11A33, 0x11A37),
    (0x11A47, 0x11A47),
    (0x11A5B, 0x11A5B),
    (0x11A98, 0x11A99),
    (0x11C3C, 0x11C3C),
    (0x11C3F, 0x11C3F),
    (0x11CB6, 0x11CB6),
    (0x11D41, 0x11D45),
    (0x11D47, 0x11D47),
    (0x11D97, 0x11D97),
    (0x11F00, 0x11F01),
    (0x11F36, 0x11F3A),
    (0x11F40, 0x11F40),
    (0x11F42, 0x11F42),
    (0x11F5A, 0x11F5A),
    (0x13440, 0x13440),
    (0x13447, 0x13455),
    (0x1611E, 0x16129),
    (0x1612D, 0x1612F),
    (0x16AF2, 0x16AF3),
    (0x16B30, 0x16B36),
    (0x16F8F, 0x16F8F),
    (0x1BC9E, 0x1BC9E),
    (0x1CF00, 0x1CF2D),
    (0x1CF30, 0x1CF46),
    (0x1D167, 0x1D169),
    (0x1D17B, 0x1D182),
    (0x1D185, 0x1D189),
    (0x1D18B, 0x1D18B),
    (0x1D1AA, 0x1D1AD),
    (0x1D242, 0x1D244),
    (0x1DA00, 0x1DA36),
    (0x1DA3B, 0x1DA6C),
    (0x1DA75, 0x1DA75),
    (0x1DA84, 0x1DA84),
    (0x1DA9B, 0x1DA9F),
    (0x1DAA1, 0x1DAAF),
    (0x1E08F, 0x1E08F),
    (0x1E130, 0x1E136),
    (0x1E2AE, 0x1E2AE),
    (0x1E2EC, 0x1E2EF),
    (0x1E4EC, 0x1E4EF),
    (0x1E5EE, 0x1E5EF),
    (0x1E8D0, 0x1E8D6),
    (0x1E944, 0x1E947),
    (0x1E94A, 0x1E94A),
    (0xE0100, 0xE01EF),
];

fn parse_slots(line: &str) -> HashMap<&str, String> {
    let mut slots = HashMap::new();
    let mut rest = line;
    while let Some(pos) = rest.find("::") {
        rest = &rest[pos + 2..];
        let end = rest.find("::").unwrap_or(rest.len());
        let (slot, _) = rest.split_at(end);
        if let Some(sp) = slot.find(' ') {
            let (k, v) = slot.split_at(sp);
            slots.insert(k.trim(), dequote(v.trim()));
        }
        rest = &rest[end.min(rest.len())..];
    }
    slots
}

/// Parsed tables, built once on first use.
struct Tables {
    /// Manual rules keyed by source string; a source can have a
    /// universal and several language-specific targets.
    rules: HashMap<String, Vec<Rule>>,
    /// Longest manual source length (bounds longest-match).
    max_source_len: usize,
    /// Auto-table per-codepoint mapping.
    auto: HashMap<char, String>,
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut rules: HashMap<String, Vec<Rule>> = HashMap::new();
        let mut max_source_len = 1usize;
        let mut auto: HashMap<char, String> = HashMap::new();

        for src in [
            include_str!("../data/uroman/romanization-table.txt"),
            include_str!("../data/uroman/romanization-table-arabic-block.txt"),
        ] {
            for line in src.lines() {
                let line = line.trim_end();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let slots = parse_slots(line);
                let (Some(s), Some(t)) = (slots.get("s"), slots.get("t")) else {
                    continue;
                };
                if s.is_empty() {
                    continue;
                }
                let lcode = slots.get("lcode").cloned();
                max_source_len = max_source_len.max(s.chars().count());
                rules.entry(s.clone()).or_default().push(Rule {
                    target: t.clone(),
                    lcode,
                });
            }
        }

        // UnicodeDataOverwrite.txt: curated corrections to the auto
        // table (e.g. Bengali ছ -> chh, Tamil ண -> n instead of the
        // auto-generated geminated forms). Applied after the auto table.
        for line in include_str!("../data/uroman/UnicodeDataOverwrite.txt").lines() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let slots = parse_slots(line);
            if let (Some(u), Some(r)) = (slots.get("u"), slots.get("r")) {
                if let Some(ch) = u32::from_str_radix(u, 16).ok().and_then(char::from_u32) {
                    auto.insert(ch, clean_target(r));
                }
            }
        }

        for line in include_str!("../data/uroman/romanization-auto-table.txt").lines() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let slots = parse_slots(line);
            let (Some(cp), Some(t)) = (slots.get("cp"), slots.get("t")) else {
                continue;
            };
            let Ok(cp) = u32::from_str_radix(cp, 16) else {
                continue;
            };
            let Some(ch) = char::from_u32(cp) else {
                continue;
            };
            // Auto fills only where no *universal* manual rule covers
            // the source: language-restricted rules (e.g. `и ::t y ::
            // lcode ukr`) must not suppress the universal auto mapping
            // (`и -> i`) for everyone else.
            let covered = rules
                .get(ch.encode_utf8(&mut [0u8; 4]))
                .is_some_and(|cands| cands.iter().any(|r| r.lcode.is_none()));
            if !covered {
                auto.entry(ch).or_insert_with(|| clean_target(t));
            }
        }

        Tables {
            rules,
            max_source_len,
            auto,
        }
    })
}

/// Strip one layer of double quotes (`::t "ae"` style entries).
fn dequote(s: &str) -> String {
    s.trim_matches('"').to_string()
}

/// Clean an auto/overwrite target: drop the leading "+" marker the
/// reference's lattice consumes (e.g. Malayalam anusvara `+m` -> `m`).
fn clean_target(t: &str) -> String {
    t.strip_prefix('+').unwrap_or(t).to_string()
}

/// Algorithmic Hangul romanization (Revised Romanization), matching the
/// reference implementation exactly.
fn hangul(ch: char) -> Option<String> {
    const LEADS: [&str; 19] = [
        "g", "gg", "n", "d", "dd", "r", "m", "b", "bb", "s", "ss", "", "j", "jj", "c", "k", "t",
        "p", "h",
    ];
    const VOWELS: [&str; 21] = [
        "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wai", "oe", "yo", "u", "weo",
        "we", "wi", "yu", "eu", "yi", "i",
    ];
    const TAILS: [&str; 28] = [
        "", "g", "gg", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh",
        "m", "b", "bs", "s", "ss", "ng", "j", "c", "k", "t", "p", "h",
    ];
    let cp = u32::from(ch);
    if (0xAC00..=0xD7A3).contains(&cp) {
        let code = cp - 0xAC00;
        let rom = format!(
            "{}{}{}",
            LEADS[(code / (28 * 21)) as usize],
            VOWELS[((code / 28) % 21) as usize],
            TAILS[(code % 28) as usize]
        );
        Some(rom)
    } else {
        None
    }
}

/// Longest manual-table match at the start of `chars`.
fn manual_match(tabs: &Tables, chars: &[char], lang: Option<&str>) -> Option<(String, usize)> {
    let max = tabs.max_source_len.min(chars.len());
    for len in (1..=max).rev() {
        let s: String = chars[..len].iter().collect();
        let Some(cands) = tabs.rules.get(&s) else {
            continue;
        };
        let hit = cands
            .iter()
            .find(|r| lang.is_some_and(|l| r.lcode.as_deref() == Some(l)))
            .or_else(|| cands.iter().find(|r| r.lcode.is_none()));
        if let Some(rule) = hit {
            return Some((rule.target.clone(), len));
        }
    }
    None
}

/// Auto-table single-char mapping; `None` means drop (unmapped mark).
/// Hangul syllables are romanized algorithmically first.
fn auto_match(tabs: &Tables, ch: char) -> Option<(String, usize)> {
    if let Some(rom) = hangul(ch) {
        return Some((rom, 1));
    }
    match tabs.auto.get(&ch) {
        Some(t) => Some((t.clone(), 1)),
        None => {
            if in_ranges(u32::from(ch), MARK_DROP_RANGES) {
                None
            } else {
                Some((ch.to_string(), 1))
            }
        }
    }
}

/// Romanize `text` to Latin. `lang` is an ISO 639-3 code (`"tur"`,
/// `"zho"`, ...) selecting language-specific rules when present.
#[must_use]
pub fn romanize(text: &str, lang: Option<&str>) -> String {
    let tabs = tables();
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < chars.len() {
        let first = chars[i];
        let Some((target, len)) =
            manual_match(tabs, &chars[i..], lang).or_else(|| auto_match(tabs, first))
        else {
            i += 1;
            continue; // dropped (unmapped combining mark)
        };
        out.push_str(&target);
        // Abugida consonants carry an inherent "a" unless a vowel sign
        // or virama follows the consumed source. Word-finally, the
        // schwa is dropped once the word already has a vowel (matching
        // the reference: मकान -> "makaan" but a lone क -> "ka").
        if in_ranges(u32::from(first), ABUGIDA_CONSONANTS) {
            let next = chars.get(i + len);
            let suppressed = next.is_some_and(|&n| in_ranges(u32::from(n), ABUGIDA_VOWEL_SIGNS));
            if !suppressed {
                // Word-final schwa deletion is Devanagari-specific in the
                // reference (Hindi orthography); Bengali and Kannada keep
                // their final inherent vowels.
                let devanagari = (0x0900..=0x097F).contains(&u32::from(first));
                let word_final = i + len == chars.len();
                let has_vowel = out.contains(['a', 'e', 'i', 'o', 'u']);
                if !(devanagari && word_final && has_vowel) {
                    out.push('a');
                }
            }
        }
        i += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_passthrough() {
        assert_eq!(romanize("hello world", None), "hello world");
        assert_eq!(romanize("Hello, World! 123", None), "Hello, World! 123");
    }

    #[test]
    fn cyrillic() {
        assert_eq!(romanize("привет", None), "privet");
        assert_eq!(romanize("НАДЕЖДА", None), "NADEZhDA");
    }

    #[test]
    fn indic_and_semitic_and_ethiopic() {
        assert_eq!(romanize("नमस्ते", None), "namaste");
        assert_eq!(romanize("γεια", None), "geia");
        assert_eq!(romanize("ሰላም", None), "salaame");
    }

    #[test]
    fn latin_extensions_use_manual_rules() {
        assert_eq!(romanize("köln", None), "koeln");
        assert_eq!(romanize("Ångström", None), "Aangstroem");
    }

    #[test]
    fn language_specific_rules_win() {
        // Turkish ç -> ch only for ::lcode tur; universal is s.
        assert_eq!(romanize("çile", Some("tur")), "chile");
        assert_eq!(romanize("çile", None), "sile");
    }

    /// Pairs frozen from the reference implementation (python uroman
    /// 2024, default language) during development; the full battery
    /// agreed 58/60, with Han (needs the pinyin reading table) and one
    /// Myanmar medial nuance as the known gaps.
    #[test]
    fn matches_reference_battery() {
        let cases = [
            ("привет", "privet"),
            ("НАДЕЖДА", "NADEZhDA"),
            ("спасибо", "spasibo"),
            ("γεια", "geia"),
            ("ευχαριστώ", "eucharisto"),
            ("שלום", "shlvm"),
            ("נאָך", "nak"),
            ("नमस्ते", "namaste"),
            ("धन्यवाद", "dhanyavaad"),
            ("मकान", "makaan"),
            ("आप", "aap"),
            ("ক", "ka"),
            ("আছি", "aachhi"),
            ("ሰላም", "salaame"),
            ("வணக்கம்", "vanakkam"),
            ("నమస్కారం", "namaskaaram"),
            ("വണക്കം", "vannakkam"),
            ("안녕하세요", "annyeonghaseyo"),
            ("감사합니다", "gamsahabnida"),
            ("مرحبا", "mrhba"),
            ("ک", "k"),
            ("شكرا", "shkra"),
        ];
        for (input, expect) in cases {
            assert_eq!(romanize(input, None), expect, "input {input:?}");
        }
    }

    #[test]
    fn multi_char_sources_match_longest_first() {
        // Devanagari nukta clusters are two-char manual rules; the
        // word-final consonant keeps its inherent a (reference: "za").
        assert_eq!(romanize("ज़", None), "za");
        assert_eq!(romanize("क", None), "ka");
    }
}
