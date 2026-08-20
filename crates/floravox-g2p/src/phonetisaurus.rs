//! Phonetisaurus-style WFST grapheme-to-phoneme as an [`OovFallback`].
//!
//! A Phonetisaurus model is an `OpenFst` binary vector FST (tropical
//! weights) transducing grapheme symbols to phoneme symbols, trained by
//! aligning a lexicon and building an n-gram transducer over the aligned
//! symbol pairs. Decoding a word is a shortest-path search: compose the
//! word's grapheme segmentation automaton with the model and take the
//! minimum-weight path.
//!
//! ## Model files
//!
//! Two layouts are understood (detected automatically):
//!
//! * **Embedded tables** (e.g. the `cmudict-*.fst` models from the
//!   phonetisaurus-downloads repo): a single `model.fst` whose header
//!   flags advertise input/output symbol tables written after the header.
//! * **External tables**: `model.fst` plus `model.grapheme.table` and
//!   `model.phoneme.table` (`OpenFst` text symbol tables, `symbol id` or
//!   `id symbol` per line).
//!
//! [`PhonetisaurusG2p::open`] takes the `model` stem either way.
//!
//! ## Format details (validated against a real 1M-state model)
//!
//! ```text
//! header:  magic i32, "vector", "standard", version=2, flags i32,
//!          properties u64, start i64, numstates i64, numarcs i64
//! table:   4-byte marker, name string, i64, i64 count,
//!          count × (string symbol, i64 key)      — when flags advertise it
//! state:   f32 final weight (inf = non-final), i64 arc count
//! arc:     i32 ilabel, i32 olabel, f32 weight, i32 nextstate   (16 bytes)
//! ```
//!
//! Two arc encodings exist in the wild: phonetisaurus files write 32-bit
//! nextstates (16-byte arcs); stock `OpenFst` writes 64-bit nextstates
//! (20-byte arcs). Both are parsed — the variant whose states consume the
//! file exactly and keep every nextstate in range wins.
//!
//! Symbols follow the phonetisaurus joint conventions: `a|c` on the input
//! side is the two-character grapheme `"ac"`, `AH0|N` on the output side
//! emits two phonemes, and `_` aligns to nothing (an input `_` consumes
//! no characters). Input casing is auto-detected from the grapheme table.
//!
//! The decoder is a clean-room implementation of the container format and
//! the path search — no GPL code enters the tree, no `ort` dependency,
//! and it works in frontend-only (`--no-default-features`) builds. The
//! search itself is a Dijkstra-style label-correcting pass with hard caps
//! against pathological epsilon cycles.

// Numeric casts below narrow values that were range-checked against the
// file header or are bounded by word length; the sizes involved cannot
// realistically exceed the narrower types on supported targets.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::{G2pError, OovFallback, Phoneme};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::{Path, PathBuf};

/// `OpenFst` binary magic number (`kFstMagicNumber`).
const FST_MAGIC: i32 = 2_125_659_606;
/// Only unaligned vector-FST file version 2 is supported.
const FST_VERSION: i32 = 2;
/// Header flag: input symbol table embedded after the header.
const FLAG_INPUT_SYMBOLS: i32 = 0x1;
/// Header flag: output symbol table embedded after the header.
const FLAG_OUTPUT_SYMBOLS: i32 = 0x2;
/// Safety caps for the decode search (pathological epsilon cycles).
const MAX_POPS: usize = 65_536;
const MAX_BACKTRACK: usize = 4096;

/// One weighted, labeled transition (`ilabel`/`olabel` 0 = epsilon).
#[derive(Debug, Clone, Copy)]
struct Arc {
    ilabel: i32,
    olabel: i32,
    weight: f32,
    nextstate: u32,
}

/// One arc as written by [`write_model`]: `(ilabel, olabel, weight,
/// nextstate)` with a 32-bit nextstate (the 16-byte encoding).
pub type WrittenArc = (i32, i32, f32, i32);

/// Serialize a model in the `OpenFst` v2 binary layout this module reads:
/// embedded symbol tables (marker + name + counts + entries), then
/// states with 16-byte arcs. Tables are `(symbol, id)` lists written in
/// ascending id order; ids 0..2 are the `<eps>`/`|`/`_` convention.
///
/// # Errors
///
/// [`G2pError::Compile`] when table ids are not exactly 0..n.
#[allow(clippy::similar_names)] // put_i32/put_i64/put_f32/put_str family
pub fn write_model(
    start: u32,
    states: &[(Option<f32>, Vec<WrittenArc>)],
    isyms: &[(String, i32)],
    osyms: &[(String, i32)],
    out: &mut Vec<u8>,
) -> Result<(), G2pError> {
    let bad_table = |t: &[(String, i32)]| {
        t.iter()
            .enumerate()
            .any(|(i, (_, id))| i32::try_from(i).unwrap_or(-1) != *id)
    };
    if bad_table(isyms) || bad_table(osyms) {
        return Err(G2pError::Compile(
            "symbol tables must be dense, ascending, starting at 0".into(),
        ));
    }
    let numarcs: u64 = states.iter().map(|(_, a)| a.len() as u64).sum();
    let put_i32 = |v: &mut Vec<u8>, x: i32| v.extend_from_slice(&x.to_le_bytes());
    let put_i64 = |v: &mut Vec<u8>, x: i64| v.extend_from_slice(&x.to_le_bytes());
    let put_f32 = |v: &mut Vec<u8>, x: f32| v.extend_from_slice(&x.to_le_bytes());
    let put_str = |v: &mut Vec<u8>, s: &str| {
        put_i32(v, s.len() as i32);
        v.extend_from_slice(s.as_bytes());
    };
    let put_table = |v: &mut Vec<u8>, name: &str, t: &[(String, i32)]| {
        put_i32(v, 0x7EB2_FB74_i32); // table marker (observed on real models)
        put_str(v, name);
        let n = t.len() as i64;
        put_i64(v, n);
        put_i64(v, n);
        for (sym, id) in t {
            put_str(v, sym);
            put_i64(v, i64::from(*id));
        }
    };

    put_i32(out, FST_MAGIC);
    put_str(out, "vector");
    put_str(out, "standard");
    put_i32(out, FST_VERSION);
    put_i32(out, FLAG_INPUT_SYMBOLS | FLAG_OUTPUT_SYMBOLS);
    out.extend_from_slice(&0x0000_0081_A542_0003_u64.to_le_bytes()); // properties
    put_i64(out, i64::from(start));
    put_i64(out, states.len() as i64);
    put_i64(out, numarcs as i64);
    put_table(out, "isyms", isyms);
    put_table(out, "osyms", osyms);
    for (final_weight, arcs) in states {
        put_f32(out, final_weight.unwrap_or(f32::INFINITY));
        put_i64(out, arcs.len() as i64);
        for (il, ol, w, ns) in arcs {
            put_i32(out, *il);
            put_i32(out, *ol);
            put_f32(out, *w);
            put_i32(out, *ns);
        }
    }
    Ok(())
}

/// Symbol tables embedded in (or loaded beside) a model.
#[derive(Debug, Default, Clone)]
pub struct SymbolTables {
    /// Grapheme string → id (compound `a|c` keys kept verbatim).
    pub graphemes: Vec<(String, i32)>,
    /// Phoneme id → symbol.
    pub phonemes: Vec<(String, i32)>,
}

/// A parsed `OpenFst` binary vector FST with tropical (f32) weights.
#[derive(Debug)]
pub struct VectorFst {
    start: Option<u32>,
    /// `None` = non-final state.
    final_weights: Vec<Option<f32>>,
    arcs: Vec<Arc>,
    /// CSR offsets, `len == states + 1`.
    offsets: Vec<u32>,
}

/// Parse result: the machine plus any symbol tables found beside it.
struct ParsedModel {
    fst: VectorFst,
    tables: SymbolTables,
    /// Which tables came from the file itself (vs. yet to be loaded).
    embedded_input: bool,
    embedded_output: bool,
}

impl VectorFst {
    /// Number of states (diagnostics).
    #[must_use]
    pub fn num_states(&self) -> usize {
        self.final_weights.len()
    }

    /// Number of arcs (diagnostics).
    #[must_use]
    pub fn num_arcs(&self) -> usize {
        self.arcs.len()
    }

    fn arcs(&self, state: u32) -> &[Arc] {
        let s = state as usize;
        &self.arcs[self.offsets[s] as usize..self.offsets[s + 1] as usize]
    }

    fn final_weight(&self, state: u32) -> Option<f32> {
        self.final_weights[state as usize]
    }
}

/// Parse an `OpenFst` binary vector FST, including embedded symbol tables.
///
/// Both the 16-byte (i32 nextstate, `phonetisaurus`) and 20-byte (i64
/// nextstate, stock `OpenFst`) arc encodings are tried; the variant that
/// consumes the file exactly and keeps all state ids in range wins.
///
/// # Errors
///
/// [`G2pError::Compile`] for wrong magic/type/version, truncation,
/// out-of-range state ids, or a body that fits neither arc encoding.
fn parse_model(bytes: &[u8]) -> Result<ParsedModel, G2pError> {
    let compile = |m: String| G2pError::Compile(m);
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.i32()? != FST_MAGIC {
        return Err(compile("not an `OpenFst` binary file".into()));
    }
    if r.string()? != "vector" {
        return Err(compile("only 'vector' FST types are supported".into()));
    }
    if r.string()? != "standard" {
        return Err(compile(
            "only 'standard' arcs (f32 weights) are supported".into(),
        ));
    }
    if r.i32()? != FST_VERSION {
        return Err(compile("only `OpenFst` file version 2 is supported".into()));
    }
    let flags = r.i32()?;
    if flags & !(FLAG_INPUT_SYMBOLS | FLAG_OUTPUT_SYMBOLS) != 0 {
        return Err(compile(format!("unsupported header flags {flags:#x}")));
    }
    let _properties = r.u64()?;
    let start = r.i64()?;
    let numstates = r.i64()?;
    let _numarcs = r.i64()?;
    if !(0..=100_000_000).contains(&numstates) {
        return Err(compile("implausible state count".into()));
    }

    let mut tables = SymbolTables::default();
    if flags & FLAG_INPUT_SYMBOLS != 0 {
        tables.graphemes = read_symbol_table(&mut r)?;
    }
    if flags & FLAG_OUTPUT_SYMBOLS != 0 {
        tables.phonemes = read_symbol_table(&mut r)?;
    }

    let body = &bytes[r.pos..];
    let fst = parse_states(body, start, numstates, 4)
        .or_else(|_| parse_states(body, start, numstates, 8))?;
    Ok(ParsedModel {
        fst,
        tables,
        embedded_input: flags & FLAG_INPUT_SYMBOLS != 0,
        embedded_output: flags & FLAG_OUTPUT_SYMBOLS != 0,
    })
}

/// Read one embedded symbol table: marker, name, two i64s (count second),
/// then `count` × (string, i64 key).
fn read_symbol_table(r: &mut Reader<'_>) -> Result<Vec<(String, i32)>, G2pError> {
    let _marker = r.i32()?;
    let _name = r.string()?;
    let _a = r.i64()?;
    let count = r.i64()?;
    if !(0..=100_000_000).contains(&count) {
        return Err(G2pError::Compile("implausible symbol count".into()));
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let sym = r.string()?;
        let key = r.i64()?;
        if !(-1..=i64::from(u32::MAX)).contains(&key) {
            return Err(G2pError::Compile(format!("symbol key {key} out of range")));
        }
        out.push((sym, key as i32));
    }
    Ok(out)
}

/// Parse the state section. `nextsize` is 4 (phonetisaurus i32 arcs) or 8
/// (stock `OpenFst` i64 arcs). Fails unless every state fits and the last
/// byte of the file is consumed exactly.
///
fn parse_states(
    body: &[u8],
    start: i64,
    numstates: i64,
    nextsize: usize,
) -> Result<VectorFst, G2pError> {
    let compile = |m: String| G2pError::Compile(m);
    let mut r = Reader { buf: body, pos: 0 };
    let mut final_weights: Vec<Option<f32>> = Vec::with_capacity(numstates as usize);
    let mut arcs: Vec<Arc> = Vec::new();
    let mut offsets: Vec<u32> = Vec::with_capacity(numstates as usize + 1);
    offsets.push(0);
    for _ in 0..numstates {
        let fw = r.f32()?;
        let narcs = r.i64()?;
        if !(0..=100_000_000).contains(&narcs) {
            return Err(compile("implausible arc count".into()));
        }
        final_weights.push(fw.is_finite().then_some(fw));
        for _ in 0..narcs {
            let ilabel = r.i32()?;
            let olabel = r.i32()?;
            let weight = r.f32()?;
            let ns = match nextsize {
                4 => i64::from(r.i32()?),
                _ => r.i64()?,
            };
            if !(0..numstates).contains(&ns) {
                return Err(compile(format!(
                    "arc points outside the state space ({ns})"
                )));
            }
            arcs.push(Arc {
                ilabel,
                olabel,
                weight,
                nextstate: ns as u32,
            });
        }
        offsets.push(arcs.len() as u32);
    }
    if r.pos != body.len() {
        return Err(compile(format!(
            "{} trailing bytes after the last state",
            body.len() - r.pos
        )));
    }
    let start = (0..numstates).contains(&start).then_some(start as u32);
    Ok(VectorFst {
        start,
        final_weights,
        arcs,
        offsets,
    })
}

/// Cursor over little-endian bytes; EOF becomes a `Compile` error.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], G2pError> {
        let end = self.pos.checked_add(n).ok_or_else(eof)?;
        let s = self.buf.get(self.pos..end).ok_or_else(eof)?;
        self.pos = end;
        Ok(s)
    }

    fn i32(&mut self) -> Result<i32, G2pError> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn i64(&mut self) -> Result<i64, G2pError> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, G2pError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn f32(&mut self) -> Result<f32, G2pError> {
        Ok(f32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn string(&mut self) -> Result<String, G2pError> {
        let len = self.i32()?;
        let bytes = self.take(usize::try_from(len).map_err(|_| eof())?)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| G2pError::Compile("non-UTF-8 string in FST".into()))
    }
}

fn eof() -> G2pError {
    G2pError::Compile("truncated FST file".into())
}

/// Parse an `OpenFst` text symbol table (`symbol id` per line; the
/// `id symbol` order is tolerated). Returns `(symbol, id)` pairs.
fn parse_symbols(text: &str) -> Vec<(String, i32)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let (Some(a), Some(b)) = (fields.next(), fields.next()) else {
            continue;
        };
        if let Ok(id) = b.parse::<i32>() {
            out.push((a.to_owned(), id));
        } else if let Ok(id) = a.parse::<i32>() {
            out.push((b.to_owned(), id));
        }
    }
    out
}

/// Phonetisaurus WFST G2P engine: lexicon-free grapheme→phoneme
/// transcription via shortest-path over the model transducer.
pub struct PhonetisaurusG2p {
    fst: VectorFst,
    /// Grapheme string (compounds normalized: `a|c` → `ac`) → symbol id.
    /// Epsilon (0), the `|` separator and the `_` null marker are excluded.
    graphemes: HashMap<String, i32>,
    /// Phoneme symbol id → string (unknown ids stay empty).
    phonemes: Vec<String>,
    /// Longest normalized grapheme symbol (bounds segmentation).
    max_grapheme_len: usize,
    /// Uppercase input before decoding when the table is upper-cased
    /// (auto-detected; override for exotic models).
    pub uppercase: bool,
}

impl PhonetisaurusG2p {
    /// Load a model. With embedded symbol tables a bare `model.fst` (or
    /// its stem) is enough; otherwise `model.grapheme.table` and
    /// `model.phoneme.table` must sit beside it.
    ///
    /// # Errors
    ///
    /// [`G2pError::Open`] for filesystem failures; [`G2pError::Compile`]
    /// for malformed FST or symbol-table data.
    pub fn open(stem: impl AsRef<Path>) -> Result<Self, G2pError> {
        let stem = stem.as_ref();
        let fst_path = with_ext(stem, "fst");
        let fst_bytes = std::fs::read(&fst_path).map_err(G2pError::Open)?;
        let mut parsed = parse_model(&fst_bytes)?;
        if !parsed.embedded_input {
            let text = std::fs::read_to_string(with_ext(&stem_without_ext(stem), "grapheme.table"))
                .map_err(|e| {
                    G2pError::Open(std::io::Error::other(format!(
                        "no embedded input table and no {}: {e}",
                        with_ext(&stem_without_ext(stem), "grapheme.table").display()
                    )))
                })?;
            parsed.tables.graphemes = parse_symbols(&text);
        }
        if !parsed.embedded_output {
            let text = std::fs::read_to_string(with_ext(&stem_without_ext(stem), "phoneme.table"))
                .map_err(|e| {
                    G2pError::Open(std::io::Error::other(format!(
                        "no embedded output table and no {}: {e}",
                        with_ext(&stem_without_ext(stem), "phoneme.table").display()
                    )))
                })?;
            parsed.tables.phonemes = parse_symbols(&text);
        }
        Self::build(parsed.fst, &parsed.tables)
    }

    /// Build from raw parts: FST bytes plus external text symbol tables
    /// (used when the FST carries no embedded tables).
    ///
    /// # Errors
    ///
    /// [`G2pError::Compile`] for malformed FST or symbol-table data.
    pub fn from_parts(
        fst_bytes: &[u8],
        grapheme_table: &str,
        phoneme_table: &str,
    ) -> Result<Self, G2pError> {
        let mut parsed = parse_model(fst_bytes)?;
        if parsed.embedded_input {
            return Err(G2pError::Compile(
                "FST embeds its input table; pass empty external tables".into(),
            ));
        }
        if parsed.embedded_output {
            return Err(G2pError::Compile(
                "FST embeds its output table; pass empty external tables".into(),
            ));
        }
        parsed.tables.graphemes = parse_symbols(grapheme_table);
        parsed.tables.phonemes = parse_symbols(phoneme_table);
        Self::build(parsed.fst, &parsed.tables)
    }

    fn build(fst: VectorFst, tables: &SymbolTables) -> Result<Self, G2pError> {
        let mut graphemes = HashMap::new();
        let mut max_grapheme_len = 1;
        let mut has_upper = false;
        let mut has_lower = false;
        for (sym, id) in &tables.graphemes {
            if *id <= 2 {
                continue; // epsilon, '|' separator, '_' null marker
            }
            let normalized = sym.replace('|', "");
            if normalized.is_empty() {
                continue;
            }
            has_upper |= normalized.chars().any(|c| c.is_ascii_uppercase());
            has_lower |= normalized.chars().any(|c| c.is_ascii_lowercase());
            max_grapheme_len = max_grapheme_len.max(normalized.chars().count());
            graphemes.insert(normalized, *id);
        }
        let mut phonemes: Vec<String> = Vec::new();
        for (sym, id) in &tables.phonemes {
            let Some(idx) = usize::try_from(*id).ok() else {
                continue;
            };
            if phonemes.len() <= idx {
                phonemes.resize(idx + 1, String::new());
            }
            phonemes[idx].clone_from(sym);
        }
        if graphemes.is_empty() {
            return Err(G2pError::Compile("empty grapheme table".into()));
        }
        Ok(Self {
            fst,
            graphemes,
            phonemes,
            max_grapheme_len,
            uppercase: has_upper && !has_lower,
        })
    }

    /// Number of model states (diagnostics).
    #[must_use]
    pub fn num_states(&self) -> usize {
        self.fst.num_states()
    }

    /// Number of model arcs (diagnostics).
    #[must_use]
    pub fn num_arcs(&self) -> usize {
        self.fst.num_arcs()
    }

    /// Transcribe `word` (`None` when no path exists — unknown graphemes,
    /// search-cap exhaustion). Multi-character graphemes (`a|c` compounds)
    /// are considered at every input position; `_` and epsilon input arcs
    /// emit without consuming. Words that fail on casing (a capital in a
    /// lower-cased table, or vice versa) are retried with the other case.
    #[must_use]
    /// Casing cascade: the table's own case first (uppercase models get
    /// uppercased input), then lowercase, then uppercase. Keeps uppercase
    /// CMUDict-style models working while letting models trained on
    /// lowercase lexicons (gruut) accept `Haus` via `haus`.
    pub fn phonemize(&self, word: &str) -> Option<Vec<Phoneme>> {
        let mut candidates: Vec<String> = Vec::with_capacity(3);
        let primary = if self.uppercase {
            word.to_uppercase()
        } else {
            word.to_string()
        };
        candidates.push(primary.clone());
        let lower = word.to_lowercase();
        if lower != primary {
            candidates.push(lower);
        }
        let upper = word.to_uppercase();
        if upper != primary && !candidates.contains(&upper) {
            candidates.push(upper);
        }
        candidates
            .into_iter()
            .find_map(|c| self.phonemize_cased(&c))
    }

    /// Transcribe an already-cased word through the search.
    fn phonemize_cased(&self, word: &str) -> Option<Vec<Phoneme>> {
        let chars: Vec<char> = word.chars().collect();
        if chars.is_empty() {
            return Some(Vec::new());
        }
        let (dist, best) = self.search(&chars)?;
        let mut node = best;
        let mut labels: Vec<i32> = Vec::new();
        for _ in 0..MAX_BACKTRACK {
            let data = dist[&node];
            let Some((pred, olabel)) = data.pred else {
                break;
            };
            if olabel > 2 {
                labels.push(olabel);
            }
            node = pred;
        }
        labels.reverse();
        Some(
            labels
                .iter()
                .filter_map(|&id| usize::try_from(id).ok().and_then(|i| self.phonemes.get(i)))
                .filter(|s| !s.is_empty() && **s != "_")
                .flat_map(|s| s.split('|').map(str::to_owned))
                .filter(|p| !p.is_empty())
                .collect(),
        )
    }

    /// Dijkstra-style label-correcting search over `(input position, fst
    /// state)` nodes. Returns the predecessor map and the best complete
    /// (all input consumed, final state) node.
    fn search(&self, chars: &[char]) -> Option<(HashMap<u64, NodeData>, u64)> {
        let start = self.fst.start?;
        let end = chars.len();
        let mut dist: HashMap<u64, NodeData> = HashMap::new();
        let mut heap: BinaryHeap<Reverse<(Cost, u64)>> = BinaryHeap::new();
        dist.insert(
            pack(0, start),
            NodeData {
                cost: 0.0,
                pred: None,
            },
        );
        heap.push(Reverse((Cost(0.0), pack(0, start))));

        let mut best_complete: Option<(f32, u64)> = None;
        let mut pops = 0usize;
        while let Some(Reverse((c, node))) = heap.pop() {
            let data = dist[&node];
            if c.0 > data.cost {
                continue; // stale entry
            }
            pops += 1;
            if pops > MAX_POPS {
                break;
            }
            let (in_pos, state) = unpack(node);
            if in_pos == end {
                if let Some(fw) = self.fst.final_weight(state) {
                    let total = data.cost + fw;
                    if best_complete.is_none_or(|(b, _)| total < b) {
                        best_complete = Some((total, node));
                    }
                }
            }

            // Grapheme segments available at this position: epsilon and
            // the '_' null marker (both consume nothing), plus every
            // normalized table prefix of the remaining input.
            let mut segs: Vec<(usize, i32)> = vec![(in_pos, 0), (in_pos, 2)];
            for k in 1..=self.max_grapheme_len {
                if in_pos + k > end {
                    break;
                }
                let s: String = chars[in_pos..in_pos + k].iter().collect();
                if let Some(&id) = self.graphemes.get(&s) {
                    segs.push((in_pos + k, id));
                }
            }

            for arc in self.fst.arcs(state) {
                for &(new_pos, id) in &segs {
                    if arc.ilabel != id {
                        continue;
                    }
                    let nk = pack(new_pos, arc.nextstate);
                    let nc = data.cost + arc.weight;
                    if dist.get(&nk).is_none_or(|d| nc < d.cost) {
                        dist.insert(
                            nk,
                            NodeData {
                                cost: nc,
                                pred: Some((node, arc.olabel)),
                            },
                        );
                        heap.push(Reverse((Cost(nc), nk)));
                    }
                }
            }
        }
        best_complete.map(|(_, node)| (dist, node))
    }
}

/// Search node: best known cost and how we got there.
#[derive(Clone, Copy)]
struct NodeData {
    cost: f32,
    /// Predecessor node and the output label emitted entering it.
    pred: Option<(u64, i32)>,
}

/// Pack `(position, state)` into a node key. Positions are word-length
/// bounded (well under 2^32) and states are table-bounded u32s.
fn pack(pos: usize, st: u32) -> u64 {
    (u64::from(u32::try_from(pos).unwrap_or(u32::MAX)) << 32) | u64::from(st)
}

/// Unpack a node key into `(position, state)`.
fn unpack(node: u64) -> (usize, u32) {
    (
        usize::try_from(node >> 32).unwrap_or(usize::MAX),
        node as u32,
    )
}

impl OovFallback for PhonetisaurusG2p {
    fn fallback(&mut self, word: &str) -> Vec<Phoneme> {
        self.phonemize(word).unwrap_or_default()
    }
}

/// f32 ordered by IEEE 754 total ordering (`BinaryHeap` payload).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cost(f32);

impl Eq for Cost {}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn stem_without_ext(path: &Path) -> PathBuf {
    if path.extension().is_some() {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn with_ext(path: &Path, ext: &str) -> PathBuf {
    let base = stem_without_ext(path);
    let mut s = base.into_os_string();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChainedFallback;

    /// Serialize a tiny model in the phonetisaurus binary format
    /// (16-byte arcs unless `wide_next` asks for the 20-byte variant).
    /// Arcs: `(from, ilabel, olabel, weight, to)`; finals: `(state, weight)`.
    fn write_fst(
        start: i64,
        arcs: &[(u32, i32, i32, f32, u32)],
        finals: &[(u32, f32)],
        wide_next: bool,
    ) -> Vec<u8> {
        let numstates = arcs
            .iter()
            .flat_map(|(f, _, _, _, t)| [*f, *t])
            .chain(finals.iter().map(|(s, _)| *s))
            .chain([start as u32])
            .max()
            .map_or(0, |m| m + 1);
        let mut v = Vec::new();
        v.extend_from_slice(&FST_MAGIC.to_le_bytes());
        let put_str = |v: &mut Vec<u8>, s: &str| {
            v.extend_from_slice(&(s.len() as i32).to_le_bytes());
            v.extend_from_slice(s.as_bytes());
        };
        put_str(&mut v, "vector");
        put_str(&mut v, "standard");
        v.extend_from_slice(&FST_VERSION.to_le_bytes());
        v.extend(&0_i32.to_le_bytes()); // flags
        v.extend_from_slice(&0u64.to_le_bytes()); // properties
        v.extend_from_slice(&start.to_le_bytes());
        v.extend_from_slice(&i64::from(numstates).to_le_bytes());
        v.extend_from_slice(&(arcs.len() as i64).to_le_bytes());
        for s in 0..numstates {
            let fw = finals
                .iter()
                .find(|(fs, _)| *fs == s)
                .map_or(f32::INFINITY, |(_, w)| *w);
            v.extend_from_slice(&fw.to_le_bytes());
            let state_arcs: Vec<_> = arcs.iter().filter(|(f, _, _, _, _)| *f == s).collect();
            v.extend_from_slice(&(state_arcs.len() as i64).to_le_bytes());
            for (_, il, ol, w, ns) in state_arcs {
                v.extend_from_slice(&il.to_le_bytes());
                v.extend_from_slice(&ol.to_le_bytes());
                v.extend_from_slice(&w.to_le_bytes());
                if wide_next {
                    v.extend_from_slice(&i64::from(*ns).to_le_bytes());
                } else {
                    v.extend_from_slice(&(*ns as i32).to_le_bytes());
                }
            }
        }
        v
    }

    const GRAPHEMES: &str = "<eps> 0\n| 1\n_ 2\nH 3\nI 4\nHI 5\n";
    const PHONEMES: &str = "<eps> 0\n| 1\n_ 2\nh 3\naɪ 4\nhaɪ 5\n";

    // States: 0 --H/h--> 1 --eps--> 2 --I/aɪ--> 3(final)
    //         0 --HI/haɪ---------(weight varies)--> 3
    //         1 --_/(compound out)--> ... tests null markers
    fn model(hi_weight: f32) -> PhonetisaurusG2p {
        let bytes = write_fst(
            0,
            &[
                (0, 3, 3, 0.1, 1),       // H -> h
                (1, 0, 0, 0.05, 2),      // pure epsilon hop
                (2, 4, 4, 0.2, 3),       // I -> aɪ
                (0, 5, 5, hi_weight, 3), // HI -> haɪ (multi-char grapheme)
                (1, 0, 0, 0.0, 1),       // zero-weight epsilon self-loop
            ],
            &[(3, 0.0)],
            false,
        );
        PhonetisaurusG2p::from_parts(&bytes, GRAPHEMES, PHONEMES).unwrap()
    }

    #[test]
    fn decodes_via_epsilon_path() {
        let m = model(0.5);
        assert_eq!(m.phonemize("HI").unwrap(), vec!["h", "aɪ"]);
        // Table is uppercase → auto-detected uppercasing.
        assert_eq!(m.phonemize("hi").unwrap(), vec!["h", "aɪ"]);
        assert!(m.uppercase);
    }

    #[test]
    fn prefers_cheaper_multichar_grapheme() {
        let m = model(0.1); // HI path (0.1) now beats H+eps+I (0.35)
        assert_eq!(m.phonemize("HI").unwrap(), vec!["haɪ"]);
    }

    #[test]
    fn unknown_grapheme_is_none() {
        let m = model(0.5);
        assert!(m.phonemize("X").is_none());
        assert!(m.phonemize("HXI").is_none());
    }

    #[test]
    fn wide_nextstate_variant_parses() {
        let bytes = write_fst(
            0,
            &[(0, 3, 3, 0.1, 1), (1, 4, 4, 0.1, 2)],
            &[(2, 0.0)],
            true,
        );
        let m = PhonetisaurusG2p::from_parts(&bytes, GRAPHEMES, PHONEMES).unwrap();
        assert_eq!(m.phonemize("HI").unwrap(), vec!["h", "aɪ"]);
    }

    #[test]
    fn compound_symbols_and_null_markers() {
        // Input "a|c" = grapheme "ac"; output "AH0|N" = two phonemes;
        // input "_" consumes nothing.
        let g = "<eps> 0\n| 1\n_ 2\nac 3\na 4\n";
        let p = "<eps> 0\n| 1\n_ 2\nAH0|N 3\nh 4\n";
        let bytes = write_fst(
            0,
            &[
                (0, 3, 3, 0.1, 1), // "ac" -> "AH0|N"
                (0, 4, 4, 0.1, 2), // "a" -> "h"
                (2, 2, 3, 0.1, 1), // "_" (null in) -> "AH0|N"
                (1, 0, 0, 0.0, 3),
            ],
            &[(3, 0.0)],
            false,
        );
        let m = PhonetisaurusG2p::from_parts(&bytes, g, p).unwrap();
        assert!(!m.uppercase); // lowercase table → no uppercasing
        assert_eq!(m.phonemize("ac").unwrap(), vec!["AH0", "N"]);
        // "a" then the null-input arc ("_") emits without consuming text.
        assert_eq!(m.phonemize("a").unwrap(), vec!["h", "AH0", "N"]);
    }

    #[test]
    fn symbol_table_field_order_tolerated() {
        let bytes = write_fst(
            0,
            &[(0, 3, 3, 0.0, 1), (1, 4, 4, 0.0, 2)],
            &[(2, 0.0)],
            false,
        );
        let m = PhonetisaurusG2p::from_parts(
            &bytes,
            "0 <eps>\n1 |\n2 _\n3 H\n4 I\n", // id-first order
            PHONEMES,
        )
        .unwrap();
        assert_eq!(m.phonemize("HI").unwrap(), vec!["h", "aɪ"]);
    }

    #[test]
    fn corrupt_inputs_rejected() {
        assert!(parse_model(&[0u8; 8]).is_err());
        let mut good = write_fst(0, &[(0, 3, 3, 0.0, 1)], &[(1, 0.0)], false);
        good.truncate(good.len() - 3); // truncated
        assert!(PhonetisaurusG2p::from_parts(&good, GRAPHEMES, PHONEMES).is_err());
    }

    #[test]
    fn file_roundtrip_and_fst_stem() {
        let dir = tempfile::tempdir().unwrap();
        let stem = dir.path().join("model");
        std::fs::write(
            with_ext(&stem, "fst"),
            write_fst(
                0,
                &[(0, 3, 3, 0.1, 1), (2, 4, 4, 0.2, 3), (1, 0, 0, 0.05, 2)],
                &[(3, 0.0)],
                false,
            ),
        )
        .unwrap();
        std::fs::write(with_ext(&stem, "grapheme.table"), GRAPHEMES).unwrap();
        std::fs::write(with_ext(&stem, "phoneme.table"), PHONEMES).unwrap();
        let via_stem = PhonetisaurusG2p::open(&stem).unwrap();
        assert_eq!(via_stem.num_states(), 4);
        assert_eq!(via_stem.phonemize("HI").unwrap(), vec!["h", "aɪ"]);
        // A path ending in .fst works as the stem too.
        let via_fst = PhonetisaurusG2p::open(with_ext(&stem, "fst")).unwrap();
        assert_eq!(via_fst.phonemize("HI").unwrap(), vec!["h", "aɪ"]);
    }

    #[test]
    fn embedded_tables() {
        let dir = tempfile::tempdir().unwrap();
        let stem = dir.path().join("model");
        let mut v = Vec::new();
        v.extend_from_slice(&FST_MAGIC.to_le_bytes());
        let put_str = |v: &mut Vec<u8>, s: &str| {
            v.extend_from_slice(&(s.len() as i32).to_le_bytes());
            v.extend_from_slice(s.as_bytes());
        };
        put_str(&mut v, "vector");
        put_str(&mut v, "standard");
        v.extend_from_slice(&FST_VERSION.to_le_bytes());
        v.extend_from_slice(&(FLAG_INPUT_SYMBOLS | FLAG_OUTPUT_SYMBOLS).to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(&0_i64.to_le_bytes()); // start
        v.extend_from_slice(&4_i64.to_le_bytes()); // numstates
        v.extend_from_slice(&3_i64.to_le_bytes()); // numarcs
        for table in [&GRAPHEMES, &PHONEMES] {
            let pairs: Vec<(String, i32)> = parse_symbols(table);
            v.extend_from_slice(&0x7EB2_FB74_u32.to_le_bytes()); // marker
            put_str(&mut v, "tbl");
            v.extend_from_slice(&(pairs.len() as i64).to_le_bytes());
            v.extend_from_slice(&(pairs.len() as i64).to_le_bytes());
            for (sym, id) in pairs {
                put_str(&mut v, &sym);
                v.extend_from_slice(&i64::from(id).to_le_bytes());
            }
        }
        // states: 0 --(3,3,.1)--> 1 --(0,0,.05)--> 2 --(4,4,.2)--> 3(final)
        let state = |final_w: f32, arcs: &[(i32, i32, f32, i32)], v: &mut Vec<u8>| {
            v.extend_from_slice(&final_w.to_le_bytes());
            v.extend_from_slice(&(arcs.len() as i64).to_le_bytes());
            for (il, ol, w, nx) in arcs {
                v.extend_from_slice(&il.to_le_bytes());
                v.extend_from_slice(&ol.to_le_bytes());
                v.extend_from_slice(&w.to_le_bytes());
                v.extend_from_slice(&nx.to_le_bytes());
            }
        };
        state(f32::INFINITY, &[(3, 3, 0.1, 1)], &mut v);
        state(f32::INFINITY, &[(0, 0, 0.05, 2)], &mut v);
        state(f32::INFINITY, &[(4, 4, 0.2, 3)], &mut v);
        state(0.0, &[], &mut v);
        std::fs::write(with_ext(&stem, "fst"), v).unwrap();
        // No external tables needed:
        let m = PhonetisaurusG2p::open(&stem).unwrap();
        assert_eq!(m.num_states(), 4);
        assert_eq!(m.num_arcs(), 3);
        assert_eq!(m.phonemize("HI").unwrap(), vec!["h", "aɪ"]);
    }

    #[test]
    fn write_model_roundtrips_through_the_reader() {
        // isyms/osyms: ids 0..2 reserved, then segments in id order.
        let isyms: Vec<(String, i32)> = vec![
            ("<eps>".into(), 0),
            ("|".into(), 1),
            ("_".into(), 2),
            ("h|a".into(), 3),
            ("l".into(), 4),
            ("o".into(), 5),
        ];
        let osyms: Vec<(String, i32)> = vec![
            ("<eps>".into(), 0),
            ("|".into(), 1),
            ("_".into(), 2),
            ("h|a".into(), 3),
            ("l".into(), 4),
            ("o|ʊ".into(), 5),
        ];
        // start=0; 0 -(3,3,.1)-> 1 -(4,4,.1)-> 2 -(5,5,.1)-> 3(final)
        let states = vec![
            (None, vec![(3, 3, 0.1, 1)]),
            (None, vec![(4, 4, 0.1, 2)]),
            (None, vec![(5, 5, 0.1, 3)]),
            (Some(0.0), vec![]),
        ];
        let mut bytes = Vec::new();
        write_model(0, &states, &isyms, &osyms, &mut bytes).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trained.fst");
        std::fs::write(&path, &bytes).unwrap();
        let model = PhonetisaurusG2p::open(&path).unwrap();
        // "ha" via the compound grapheme, then "l", "o"; outputs split
        // compounds back apart.
        assert_eq!(
            model.phonemize("halo").unwrap(),
            vec!["h", "a", "l", "o", "ʊ"]
        );
        // lowercase table accepts capitals via the cascade
        assert_eq!(
            model.phonemize("Halo").unwrap(),
            vec!["h", "a", "l", "o", "ʊ"]
        );
    }

    #[test]
    fn oov_fallback_and_chain() {
        let mut chain = ChainedFallback(model(0.5), crate::RuleFallback::default());
        assert_eq!(chain.fallback("hi"), vec!["h", "aɪ"]);
        // Unknown graphemes: engine yields nothing, spelling takes over.
        assert!(!chain.fallback("Xzq").is_empty());
        assert_eq!(chain.fallback("Xzq").len(), 3);
    }

    /// Real-model check: `PHONETISaurus_MODEL=/path/model.fst cargo test
    /// -p floravox-g2p real_model -- --nocapture` decodes sample words.
    #[test]
    fn real_model() {
        let Ok(path) = std::env::var("PHONETISAURUS_MODEL") else {
            return;
        };
        let m = PhonetisaurusG2p::open(&path).expect("open real model");
        eprintln!(
            "states={} arcs={} grapheme-segments={} uppercase={}",
            m.num_states(),
            m.num_arcs(),
            m.graphemes.len(),
            m.uppercase
        );
        for word in ["hello", "Hello", "world", "phonetisaurus"] {
            eprintln!("{word} -> {:?}", m.phonemize(word));
        }
        assert!(
            m.phonemize("hello").is_some_and(|p| !p.is_empty()),
            "real model produced no path for 'hello'"
        );
    }
}
