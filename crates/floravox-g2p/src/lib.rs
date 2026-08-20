//! # floravox-g2p
//!
//! Hybrid grapheme-to-phoneme conversion:
//!
//! * **Tier 1** — [`FstLexicon`]: static lexicons compiled with the `fst`
//!   crate (`CMUDict`, `WikiPron`, gruut extractions). On-disk lexicons are
//!   memory-mapped (~0 resident RAM), lookups are sub-100 µs, footprints are
//!   ~5–15 MB per language.
//! * **Tier 2** — [`OovFallback`]: pluggable out-of-vocabulary strategy —
//!   rule-based spelling fallback here, a [`ByT5`](byt5::Byt5G2p) ONNX
//!   engine behind the `onnx` feature, or a Phonetisaurus WFST behind the
//!   same trait.
//!
//! Wrap any phonemizer in a bounded [`CachedPhonemizer`] so repeated words
//! (the common case in AAC and screen-reader workloads) cost one hash lookup.
//!
//! The [`ingest`] module converts third-party pronunciation data (CMUDICT,
//! WikiPron / gruut extractions) into rows for [`LexiconWriter`].
//!
//! On-disk format: a lexicon "stem" is two files — `stem.fst` (word →
//! packed `offset/length` u64) and `stem.pho` (a flat blob of
//! space-separated phoneme strings). Both are mmap'd at open time.
//!
//! ```
//! use floravox_g2p::{FstLexicon, LexiconPhonemizer, RuleFallback, TokenPhonemizer};
//!
//! let lex = FstLexicon::from_rows(vec![
//!     ("hello".into(), "h ə l o u".into()),
//!     ("world".into(), "w ɜː l d".into()),
//! ])
//! .unwrap();
//! let mut g2p = LexiconPhonemizer::new(lex, RuleFallback::default());
//! // "hello," → h ə l o u + trailing pause symbol
//! assert_eq!(g2p.phonemize_token("hello,").len(), 6);
//! assert!(!g2p.phonemize_token("zzzq").is_empty()); // OOV: spelled out
//! ```

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod ingest;

#[cfg(feature = "onnx")]
pub mod byt5;

pub use ingest::{Ingested, SourceFormat};
#[cfg(feature = "onnx")]
pub use byt5::Byt5G2p;

/// One pronunciation symbol (IPA-ish, model-specific alphabet).
pub type Phoneme = String;

/// Errors produced by lexicon operations.
#[derive(Debug)]
pub enum G2pError {
    /// The lexicon files could not be opened or parsed.
    Open(std::io::Error),
    /// Input data could not be compiled (unsorted keys, bad TSV, ...).
    Compile(String),
    /// A neural OOV engine failed (model load or inference).
    Inference(String),
}

impl fmt::Display for G2pError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(e) => write!(f, "lexicon open failed: {e}"),
            Self::Compile(e) => write!(f, "lexicon compile failed: {e}"),
            Self::Inference(e) => write!(f, "g2p inference failed: {e}"),
        }
    }
}

impl std::error::Error for G2pError {}

/// Pack a blob (offset, length) into an FST u64 value.
/// Offsets up to 2^43 bytes, entries up to 1 MiB.
const LEN_BITS: u32 = 20;
const LEN_MASK: u64 = (1 << LEN_BITS) - 1;

fn pack_value(offset: usize, len: usize) -> u64 {
    debug_assert!(len as u64 <= LEN_MASK, "phoneme entry exceeds 1 MiB");
    ((offset as u64) << LEN_BITS) | (len as u64 & LEN_MASK)
}

fn unpack_value(v: u64) -> (usize, usize) {
    (((v >> LEN_BITS) as usize), (v & LEN_MASK) as usize)
}

/// A lexicon: an FST mapping lowercased words to phoneme strings stored in a
/// companion blob. `D` is the backing storage (`Vec<u8>` for in-memory
/// builds, `memmap2::Mmap` for files).
pub struct FstLexicon<D: AsRef<[u8]> = Vec<u8>> {
    map: fst::Map<D>,
    blob: D,
}

/// In-memory lexicon (tests, small builds).
pub type MemLexicon = FstLexicon<Vec<u8>>;
/// Memory-mapped lexicon loaded from disk.
pub type MmapLexicon = FstLexicon<memmap2::Mmap>;

impl<D: AsRef<[u8]>> fmt::Debug for FstLexicon<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FstLexicon")
            .field("entries", &self.map.len())
            .finish()
    }
}

impl FstLexicon<memmap2::Mmap> {
    /// Open a compiled lexicon from its stem: `stem.fst` + `stem.pho`.
    /// Both halves are memory-mapped; resident RAM stays near zero.
    /// # Errors
    ///
    /// [`G2pError::Open`] when either file is missing or unparseable.
    pub fn open(stem: impl AsRef<Path>) -> Result<Self, G2pError> {
        let fst_path = append_ext(stem.as_ref(), "fst");
        let pho_path = append_ext(stem.as_ref(), "pho");
        let map = {
            let file = File::open(&fst_path).map_err(G2pError::Open)?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(G2pError::Open)?;
            fst::Map::new(mmap).map_err(|e| G2pError::Open(std::io::Error::other(e)))?
        };
        let blob = {
            let file = File::open(&pho_path).map_err(G2pError::Open)?;
            unsafe { memmap2::Mmap::map(&file) }.map_err(G2pError::Open)?
        };
        Ok(Self { map, blob })
    }
}

impl FstLexicon<Vec<u8>> {
    /// Compile from inline TSV (`word\tph1 ph2 ph3` per line; `#` comments).
    /// Duplicate keys: last wins.
    /// # Errors
    ///
    /// [`G2pError::Compile`] on malformed TSV lines.
    pub fn from_tsv(tsv: &str) -> Result<Self, G2pError> {
        let mut rows: Vec<(String, String)> = Vec::new();
        for line in tsv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((word, phones)) = line.split_once('\t') else {
                return Err(G2pError::Compile(format!("line is not TSV: {line:?}")));
            };
            rows.push((word.trim().to_owned(), phones.trim().to_owned()));
        }
        Self::from_rows(rows)
    }

    /// Compile an in-memory lexicon from (word, phonemes) rows.
    /// # Errors
    ///
    /// [`G2pError::Compile`] when the FST cannot be built.
    pub fn from_rows(rows: Vec<(String, String)>) -> Result<Self, G2pError> {
        let mut unique: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        for (w, p) in rows {
            unique.insert(w.to_lowercase(), p);
        }
        let mut blob = Vec::new();
        let mut entries: Vec<(String, u64)> = Vec::with_capacity(unique.len());
        for (word, phones) in unique {
            let offset = blob.len();
            blob.extend_from_slice(phones.as_bytes());
            let len = blob.len() - offset;
            entries.push((word, pack_value(offset, len)));
        }
        let mut builder = fst::MapBuilder::memory();
        for (word, value) in entries {
            builder
                .insert(word.as_bytes(), value)
                .map_err(|e| G2pError::Compile(e.to_string()))?;
        }
        let fst_bytes = builder
            .into_inner()
            .map_err(|e| G2pError::Compile(e.to_string()))?;
        let map = fst::Map::new(fst_bytes).map_err(|e| G2pError::Compile(e.to_string()))?;
        Ok(Self { map, blob })
    }
}

impl<D: AsRef<[u8]>> FstLexicon<D> {
    /// Number of entries in the lexicon.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True when the lexicon has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Look up a word (case-insensitive); `None` when out of vocabulary.
    #[must_use]
    pub fn lookup(&self, word: &str) -> Option<Vec<Phoneme>> {
        let key = word.to_lowercase();
        let value = self.map.get(key.as_bytes())?;
        let (off, len) = unpack_value(value);
        let blob = self.blob.as_ref();
        let raw = blob.get(off..off + len)?;
        std::str::from_utf8(raw)
            .ok()
            .map(|s| s.split_whitespace().map(str::to_owned).collect())
    }
}

/// Builder that writes a lexicon pair (`stem.fst` / `stem.pho`) to disk.
pub struct LexiconWriter {
    stem: PathBuf,
}

impl LexiconWriter {
    /// Create a writer targeting the given stem path.
    #[must_use]
    pub fn new(stem: impl Into<PathBuf>) -> Self {
        Self { stem: stem.into() }
    }

    /// Compile `rows` and write both halves. Returns the entry count.
    /// # Errors
    ///
    /// [`G2pError::Open`] on filesystem failures; [`G2pError::Compile`] on bad rows.
    pub fn write(self, rows: Vec<(String, String)>) -> Result<usize, G2pError> {
        let lex = FstLexicon::from_rows(rows)?;
        let count = lex.len();
        let fst_bytes = lex.map.as_fst().as_bytes().to_vec();
        let mut f = File::create(append_ext(&self.stem, "fst")).map_err(G2pError::Open)?;
        f.write_all(&fst_bytes).map_err(G2pError::Open)?;
        let mut p = File::create(append_ext(&self.stem, "pho")).map_err(G2pError::Open)?;
        p.write_all(&lex.blob).map_err(G2pError::Open)?;
        Ok(count)
    }
}

/// Replace the extension of `path` (or append one if it has none).
fn append_ext(path: &Path, ext: &str) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    if path.extension().is_some() {
        os = path.with_extension("").into_os_string();
    }
    let mut s = os.into_string().unwrap_or_default();
    s.push('.');
    s.push_str(ext);
    PathBuf::from(s)
}

/// Strategy for words missing from the lexicon.
pub trait OovFallback {
    /// Pronounce an out-of-vocabulary word.
    fn fallback(&mut self, word: &str) -> Vec<Phoneme>;
}

impl<F: OovFallback + ?Sized> OovFallback for Box<F> {
    fn fallback(&mut self, word: &str) -> Vec<Phoneme> {
        (**self).fallback(word)
    }
}

/// Try `A` first; when it produces nothing (unknown input, failed engine),
/// fall back to `B`. Typical chain: neural engine → letter spelling.
///
/// ```
/// use floravox_g2p::{ChainedFallback, OovFallback, RuleFallback};
///
/// struct Empty;
/// impl OovFallback for Empty {
///     fn fallback(&mut self, _word: &str) -> Vec<String> { Vec::new() }
/// }
///
/// let mut chain = ChainedFallback(Empty, RuleFallback::default());
/// assert!(!chain.fallback("zzzq").is_empty()); // spelled out by B
/// ```
pub struct ChainedFallback<A, B>(pub A, pub B);

impl<A: OovFallback, B: OovFallback> OovFallback for ChainedFallback<A, B> {
    fn fallback(&mut self, word: &str) -> Vec<Phoneme> {
        let first = self.0.fallback(word);
        if first.is_empty() {
            self.1.fallback(word)
        } else {
            first
        }
    }
}

/// Letter-name spelling fallback: pronounces unknown words by "spelling"
/// them. Crude but deterministic — an edge device keeps speaking instead of
/// going silent. A Phonetisaurus WFST or `ByT5` model replaces this behind the
/// same trait.
#[derive(Debug, Default, Clone)]
pub struct RuleFallback {
    /// Drop unknown characters instead of stopping at them (default true).
    pub drop_unknown: bool,
}

impl OovFallback for RuleFallback {
    fn fallback(&mut self, word: &str) -> Vec<Phoneme> {
        let mut out = Vec::new();
        for ch in word.chars().filter(|c| c.is_alphanumeric()) {
            let lower = ch.to_lowercase().next().unwrap_or(ch);
            if let Some(p) = letter_phoneme(lower) {
                out.push(p.to_string());
            } else if !self.drop_unknown {
                break;
            }
        }
        out
    }
}

/// English letter names for spelling out (per-language tables ship with real
/// lexicon data).
fn letter_phoneme(c: char) -> Option<&'static str> {
    Some(match c {
        'a' => "eɪ",
        'b' => "biː",
        'c' => "siː",
        'd' => "diː",
        'e' => "iː",
        'f' => "ɛf",
        'g' => "dʒiː",
        'h' => "eɪtʃ",
        'i' => "aɪ",
        'j' => "dʒeɪ",
        'k' => "keɪ",
        'l' => "ɛl",
        'm' => "ɛm",
        'n' => "ɛn",
        'o' => "oʊ",
        'p' => "piː",
        'q' => "kjuː",
        'r' => "ɑːr",
        's' => "ɛs",
        't' => "tiː",
        'u' => "juː",
        'v' => "viː",
        'w' => "dʌbljuː",
        'x' => "ɛks",
        'y' => "waɪ",
        'z' => "ziː",
        '0' => "zɪəroʊ",
        '1' => "wʌn",
        '2' => "tuː",
        '3' => "θriː",
        '4' => "fɔːr",
        '5' => "faɪv",
        '6' => "sɪks",
        '7' => "sɛvən",
        '8' => "eɪt",
        '9' => "naɪn",
        _ => return None,
    })
}

/// A full phonemizer: lexicon first, OOV fallback second.
pub struct LexiconPhonemizer<D: AsRef<[u8]>, F: OovFallback> {
    lexicon: FstLexicon<D>,
    fallback: F,
}

impl<D: AsRef<[u8]>, F: OovFallback> LexiconPhonemizer<D, F> {
    /// Combine a lexicon with an OOV strategy.
    pub fn new(lexicon: FstLexicon<D>, fallback: F) -> Self {
        Self { lexicon, fallback }
    }

    /// Phonemize a single bare word (no punctuation splitting).
    pub fn phonemize_word(&mut self, word: &str) -> Vec<Phoneme> {
        self.lexicon
            .lookup(word)
            .unwrap_or_else(|| self.fallback.fallback(word))
    }

    /// Number of lexicon entries (diagnostics).
    #[must_use]
    pub fn lexicon_len(&self) -> usize {
        self.lexicon.len()
    }
}

/// Pause symbols emitted for punctuation, matching the piper phoneme
/// alphabet convention (`,` = short pause, `.` = long pause, `-` = break).
fn punct_phonemes(token: &str) -> Vec<Phoneme> {
    let mut out = Vec::new();
    for ch in token.chars() {
        match ch {
            ',' | ';' | ':' => out.push(",".into()),
            '.' | '!' | '?' => out.push(".".into()),
            '-' | '—' | '–' => out.push("-".into()),
            _ => {}
        }
    }
    out
}

/// Strip punctuation from a token, returning the alphanumeric core.
fn word_core(token: &str) -> Option<&str> {
    let core: &str = token.trim_matches(|c: char| !c.is_alphanumeric());
    (!core.is_empty()).then_some(core)
}

/// Extension trait: phonemize a whitespace-delimited token (word core +
/// attached punctuation → phonemes + pause symbols).
pub trait TokenPhonemizer {
    /// Phonemize one whitespace token.
    fn phonemize_token(&mut self, token: &str) -> Vec<Phoneme>;
}

impl<P: TokenPhonemizer + ?Sized> TokenPhonemizer for Box<P> {
    fn phonemize_token(&mut self, token: &str) -> Vec<Phoneme> {
        (**self).phonemize_token(token)
    }
}

impl<D: AsRef<[u8]>, F: OovFallback> TokenPhonemizer for LexiconPhonemizer<D, F> {
    fn phonemize_token(&mut self, token: &str) -> Vec<Phoneme> {
        let mut out = Vec::new();
        if let Some(core) = word_core(token) {
            out.extend(self.phonemize_word(core));
        }
        // Pause symbols follow the word ("hello," → h ə l o u ,).
        out.extend(punct_phonemes(token));
        out
    }
}

/// Bounded LRU cache wrapper. Words repeat constantly in real workloads
/// (AAC boards, screen readers); a small cache turns them into hash lookups.
pub struct CachedPhonemizer<P: TokenPhonemizer> {
    inner: P,
    cache: LruCache,
}

impl<P: TokenPhonemizer> CachedPhonemizer<P> {
    /// Wrap `inner` with a cache of `capacity` entries.
    #[must_use]
    pub fn new(inner: P, capacity: usize) -> Self {
        Self {
            inner,
            cache: LruCache::new(capacity),
        }
    }

    /// Mutable access to the wrapped phonemizer.
    pub fn inner(&mut self) -> &mut P {
        &mut self.inner
    }
}

impl<P: TokenPhonemizer> TokenPhonemizer for CachedPhonemizer<P> {
    fn phonemize_token(&mut self, token: &str) -> Vec<Phoneme> {
        if let Some(hit) = self.cache.get(token) {
            return (*hit).clone();
        }
        let result = self.inner.phonemize_token(token);
        self.cache
            .insert(token.to_string(), Arc::new(result.clone()));
        result
    }
}

/// Minimal LRU: `HashMap` plus age counters. No unsafe code, no dependencies.
struct LruCache {
    map: HashMap<String, (Arc<Vec<Phoneme>>, u64)>,
    capacity: usize,
    tick: u64,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            capacity: capacity.max(1),
            tick: 0,
        }
    }

    fn get(&mut self, key: &str) -> Option<Arc<Vec<Phoneme>>> {
        self.tick += 1;
        let t = self.tick;
        self.map.get_mut(key).map(|(v, age)| {
            *age = t;
            Arc::clone(v)
        })
    }

    fn insert(&mut self, key: String, value: Arc<Vec<Phoneme>>) {
        self.tick += 1;
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            if let Some(oldest) = self
                .map
                .iter()
                .min_by_key(|(_, (_, age))| *age)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&oldest);
            }
        }
        self.map.insert(key, (value, self.tick));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lexicon() -> MemLexicon {
        FstLexicon::from_tsv("hello\th ə l o u\nworld\tw ɜː l d\nWorld\tw ɜː l d\n")
            .expect("compile")
    }

    #[test]
    fn lookup_hit_and_case() {
        let lex = test_lexicon();
        assert_eq!(
            lex.lookup("hello"),
            Some(
                vec!["h", "ə", "l", "o", "u"]
                    .into_iter()
                    .map(String::from)
                    .collect()
            )
        );
        assert_eq!(lex.lookup("WORLD").map(|v| v.len()), Some(4)); // w ɜː l d
        assert!(lex.lookup("absent").is_none());
    }

    #[test]
    fn duplicate_keys_last_wins() {
        let lex = FstLexicon::from_tsv("word\ta\nword\tb c\n").unwrap();
        assert_eq!(lex.len(), 1);
        assert_eq!(lex.lookup("word").unwrap(), vec!["b", "c"]);
    }

    #[test]
    fn long_pronunciations_survive() {
        let phones = "ɛ k s t ɹ ə ɔː ɹ d ə n ɛ ɹ iː"; // 14 symbols
        let lex = FstLexicon::from_rows(vec![("extraordinary".into(), phones.into())]).unwrap();
        assert_eq!(lex.lookup("extraordinary").unwrap().len(), 14);
    }

    #[test]
    fn file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let stem = dir.path().join("en");
        let n = LexiconWriter::new(&stem)
            .write(vec![
                ("hello".into(), "h ə l o u".into()),
                ("world".into(), "w ɜː l d".into()),
            ])
            .unwrap();
        assert_eq!(n, 2);
        let lex = MmapLexicon::open(&stem).unwrap();
        assert_eq!(lex.len(), 2);
        assert_eq!(lex.lookup("world").unwrap()[1], "ɜː");
        assert!(lex.lookup("nope").is_none());
    }

    #[test]
    fn fallback_spells() {
        let lex = test_lexicon();
        let mut g2p = LexiconPhonemizer::new(lex, RuleFallback::default());
        assert!(!g2p.phonemize_word("qqq").is_empty());
        assert_eq!(g2p.phonemize_word("hello")[0], "h");
    }

    #[test]
    fn token_with_punctuation() {
        let lex = test_lexicon();
        let mut g2p = LexiconPhonemizer::new(lex, RuleFallback::default());
        let ph = g2p.phonemize_token("hello,");
        assert_eq!(ph.first().map(String::as_str), Some("h"));
        assert_eq!(ph.last().map(String::as_str), Some(","));
    }

    #[test]
    fn cache_hits_and_eviction() {
        let lex = test_lexicon();
        let mut cached =
            CachedPhonemizer::new(LexiconPhonemizer::new(lex, RuleFallback::default()), 4);
        let a = cached.phonemize_token("hello");
        let b = cached.phonemize_token("hello");
        assert_eq!(a, b);
        for i in 0..10 {
            cached.phonemize_token(&format!("w{i}"));
        }
        assert_eq!(cached.phonemize_token("hello").len(), 5);
    }
}
