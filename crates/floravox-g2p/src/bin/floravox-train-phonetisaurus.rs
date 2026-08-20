//! Train a `Phonetisaurus`-style G2P model on a lexicon.
//! Pipeline (all in-process, no external tools):
//!
//! 1. **M2M alignment** — hard-EM over substring pairs (grapheme segment
//!    ↔ phoneme segment, up to `--gmax`/`--pmax` each, deletions
//!    allowed), like `phonetisaurus-align` but in Rust.
//! 2. **`N`-gram model** — counts over the aligned joint tokens (order
//!    `--order`, stupid-backoff α = 0.4).
//! 3. **WFST serialization** — `OpenFst` v2 binary layout with embedded
//!    symbol tables, loadable by `PhonetisaurusG2p::open` and by real
//!    phonetisaurus tools.
//!
//! Because output symbols come only from the training lexicon, the
//! model's alphabet matches it by construction — the reason per-language
//! models are trained on the published voicegarden-lexicons bundles.
//!
//! Evaluation holds out `--holdout` of the lexicon, trains on the rest,
//! and decodes the held-out words through the shipped model file (the
//! same reader consumers use): exact-match rate, phoneme error rate
//! (PER), and decode coverage.
//!
//! Usage:
//!   floravox-train-phonetisaurus LEXICON.tsv MODEL.fst
//!     [--order 7] [--iters 8] [--gmax 2] [--pmax 2]
//!     [--holdout 0.05] [--seed 7] [--metrics metrics.json]

// This is a training CLI: the DP math uses single-letter bindings (g/p),
// count/size casts, and a long orchestrating main by design.
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use floravox_g2p::phonetisaurus::write_model;
use floravox_g2p::PhonetisaurusG2p;
use std::collections::HashMap;

/// Floor probability for unseen pairs during EM.
const FLOOR: f64 = 1e-10;
/// Stupid-backoff factor.
const ALPHA: f64 = 0.4;
/// Reserved "no segment" id in the inventories.
const INVALID: u32 = u32::MAX;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut input = None;
    let mut output = None;
    let mut order = 7usize;
    let mut iters = 8usize;
    let mut gmax = 2usize;
    let mut pmax = 2usize;
    let mut holdout = 0.05f64;
    let mut seed: u64 = 7;
    let mut metrics_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--order" => order = parse(&mut args, &a),
            "--iters" => iters = parse(&mut args, &a),
            "--gmax" => gmax = parse(&mut args, &a),
            "--pmax" => pmax = parse(&mut args, &a),
            "--holdout" => holdout = parse(&mut args, &a),
            "--seed" => seed = parse(&mut args, &a),
            "--metrics" => metrics_path = args.next(),
            other if other.starts_with("--") => {
                eprintln!("unknown flag {other:?}");
                std::process::exit(2);
            }
            other => {
                if input.is_none() {
                    input = Some(other.to_string());
                } else if output.is_none() {
                    output = Some(other.to_string());
                } else {
                    eprintln!("unexpected argument {other:?}");
                    std::process::exit(2);
                }
            }
        }
    }
    let (Some(input), Some(output)) = (input, output) else {
        eprintln!("usage: floravox-train-phonetisaurus LEXICON.tsv MODEL.fst [flags]");
        std::process::exit(2);
    };
    assert!(
        (1..=3).contains(&gmax) && (1..=3).contains(&pmax),
        "gmax/pmax in 1..=3"
    );
    assert!(order >= 2, "order >= 2");

    let t0 = std::time::Instant::now();
    let lex = read_lexicon(&input);
    eprintln!("lexicon: {} entries", lex.len());

    // deterministic shuffle + split
    let mut rng = Rng::new(seed);
    let mut idx: Vec<usize> = (0..lex.len()).collect();
    for i in (1..idx.len()).rev() {
        let j = rng.next() as usize % (i + 1);
        idx.swap(i, j);
    }
    let n_hold = ((lex.len() as f64) * holdout).round() as usize;
    let train: Vec<(String, Vec<String>)> = idx[n_hold..].iter().map(|&i| lex[i].clone()).collect();
    let hold: Vec<(String, Vec<String>)> = idx[..n_hold].iter().map(|&i| lex[i].clone()).collect();
    eprintln!("split: {} train / {} holdout", train.len(), hold.len());

    // ---- inventories over the TRAIN set ----
    let (gmap, glist, pmap, plist) = inventories(&train, gmax, pmax);
    eprintln!(
        "inventories: {} grapheme segments, {} phone segments",
        glist.len(),
        plist.len()
    );

    // ---- EM alignment ----
    let mut probs: HashMap<(u32, u32), f64> = HashMap::new();
    for (w, ph) in &train {
        let chars: Vec<char> = w.chars().collect();
        if chars.is_empty() || ph.is_empty() {
            continue;
        }
        let unit = 1.0 / f64::from(u32::try_from(chars.len() * ph.len()).unwrap_or(1));
        for c in &chars {
            if let Some(&g) = gmap.get(&c.to_string()) {
                for p in ph {
                    if let Some(&pn) = pmap.get(p.as_str()) {
                        *probs.entry((g, pn)).or_insert(0.0) += unit;
                    }
                }
            }
        }
    }
    let mut alignments: Vec<Vec<(u32, u32)>> = Vec::with_capacity(train.len());
    for it in 0..iters {
        let mut counts: HashMap<(u32, u32), f64> = HashMap::new();
        alignments.clear();
        for (w, ph) in &train {
            let path = align(w, ph, &gmap, &pmap, gmax, pmax, &probs);
            for pair in &path {
                *counts.entry(*pair).or_insert(0.0) += 1.0;
            }
            alignments.push(path);
        }
        let total: f64 = counts.values().sum();
        probs = counts
            .iter()
            .map(|(k, v)| (*k, v / total + FLOOR))
            .collect();
        eprintln!("em iter {}: {} distinct pairs", it + 1, probs.len());
    }

    // ---- joint tokens: pair -> token id; 0 = B, 1 = E ----
    let mut tok_of: HashMap<(u32, u32), u32> = HashMap::new();
    let mut tok_pairs: Vec<(u32, u32)> = vec![(u32::MAX, u32::MAX), (u32::MAX, u32::MAX)];
    for al in &alignments {
        for p in al {
            if !tok_of.contains_key(p) {
                tok_of.insert(*p, tok_pairs.len() as u32);
                tok_pairs.push(*p);
            }
        }
    }
    eprintln!("joint tokens: {}", tok_pairs.len());

    // ---- n-gram counts ----
    let mut ngrams: HashMap<(Vec<u32>, u32), u64> = HashMap::new();
    for al in &alignments {
        let mut seq: Vec<u32> = Vec::with_capacity(al.len() + 2);
        seq.push(0); // B
        for p in al {
            seq.push(tok_of[p]);
        }
        seq.push(1); // E
        for o in 1..=order.min(seq.len()) {
            for i in (o - 1)..seq.len() {
                let h = &seq[i + 1 - o..i];
                *ngrams.entry((h.to_vec(), seq[i])).or_insert(0) += 1;
            }
        }
    }
    eprintln!("ngram entries: {}", ngrams.len());
    let mut ctx_total: HashMap<Vec<u32>, u64> = HashMap::new();
    for ((h, _), c) in &ngrams {
        *ctx_total.entry(h.clone()).or_insert(0) += c;
    }

    // ---- states ----
    let mut state_of: HashMap<Vec<u32>, u32> = HashMap::new();
    let ensure = |h: &[u32], state_of: &mut HashMap<Vec<u32>, u32>| -> u32 {
        if let Some(&s) = state_of.get(h) {
            s
        } else {
            let s = state_of.len() as u32;
            state_of.insert(h.to_vec(), s);
            s
        }
    };
    ensure(&[], &mut state_of);
    // Start = the begin-token-conditioned context. Starting here (rather
    // than the unigram hub, whose out-degree spans every token) keeps the
    // decoder's search frontier tight, matching how published
    // phonetisaurus models are structured.
    let start_state = ensure(&[0], &mut state_of);

    let mut raw_arcs: Vec<(u32, u32, f32, u32)> = Vec::new(); // (from, tok, w, dest)
    let mut finals: HashMap<u32, f32> = HashMap::new();
    for ((h, w), c) in &ngrams {
        let from = ensure(h, &mut state_of);
        let mut full = h.clone();
        full.push(*w);
        while full.len() > order - 1 {
            full.remove(0);
        }
        let dest = ensure(&full, &mut state_of);
        let p = *c as f64 / ctx_total[h] as f64;
        raw_arcs.push((from, *w, -p.log10() as f32, dest));
        if *w == 1 {
            finals.insert(dest, 0.0);
        }
    }
    let contexts: Vec<Vec<u32>> = state_of.keys().cloned().collect();
    let mut raw_backoff: Vec<(u32, u32)> = Vec::new();
    for h in contexts {
        if h.is_empty() {
            continue;
        }
        let mut suffix = h.clone();
        suffix.remove(0);
        let from = ensure(&h, &mut state_of);
        let to = ensure(&suffix, &mut state_of);
        raw_backoff.push((from, to));
    }
    let n_states = state_of.len();
    eprintln!(
        "states: {}, arcs: {} (+{} backoff)",
        n_states,
        raw_arcs.len(),
        raw_backoff.len()
    );

    // ---- symbol tables ----
    let mut isyms: Vec<(String, i32)> = vec![("<eps>".into(), 0), ("|".into(), 1), ("_".into(), 2)];
    let mut osyms = isyms.clone();
    let tok_labels: Vec<(i32, i32)> = tok_pairs
        .iter()
        .map(|&(g, p)| {
            let il = label(g, &glist, &mut isyms, true);
            let ol = label(p, &plist, &mut osyms, false);
            (il, ol)
        })
        .collect();

    // ---- assemble ----
    let mut out_states: Vec<(Option<f32>, Vec<(i32, i32, f32, i32)>)> =
        vec![(None, Vec::new()); n_states];
    for (s, state) in out_states.iter_mut().enumerate() {
        state.0 = finals.get(&(s as u32)).copied();
    }
    let arc_count = raw_arcs.len() + raw_backoff.len();
    for (from, tok, w, dest) in raw_arcs {
        let (il, ol) = tok_labels[tok as usize];
        out_states[from as usize].1.push((il, ol, w, dest as i32));
    }
    let backoff_w = -ALPHA.log10() as f32;
    for (from, to) in raw_backoff {
        out_states[from as usize]
            .1
            .push((0, 0, backoff_w, to as i32));
    }
    for (_, arcs) in &mut out_states {
        arcs.sort_unstable_by_key(|a| a.3);
    }

    let mut bytes = Vec::new();
    write_model(start_state, &out_states, &isyms, &osyms, &mut bytes).expect("serialize model");
    std::fs::write(&output, &bytes).expect("write model");
    eprintln!(
        "wrote {} ({} bytes) in {:.1}s",
        output,
        bytes.len(),
        t0.elapsed().as_secs_f64()
    );

    // ---- evaluation through the shipped file ----
    let model = PhonetisaurusG2p::open(&output).expect("reload model for eval");
    let (exact, per, coverage) = evaluate(&model, &hold);
    eprintln!(
        "eval: exact {:.1}%, PER {:.1}%, coverage {:.1}%",
        exact * 100.0,
        per * 100.0,
        coverage * 100.0
    );
    if let Some(path) = metrics_path {
        let json = format!(
            "{{\n  \"train\": {},\n  \"holdout\": {},\n  \"exact_match\": {:.4},\n  \"per\": {:.4},\n  \"coverage\": {:.4},\n  \"order\": {},\n  \"iters\": {},\n  \"gmax\": {},\n  \"pmax\": {},\n  \"states\": {},\n  \"arcs\": {}\n}}\n",
            train.len(),
            hold.len(),
            exact,
            per,
            coverage,
            order,
            iters,
            gmax,
            pmax,
            n_states,
            arc_count
        );
        std::fs::write(path, json).expect("write metrics");
    }
}

/// Intern one side's symbol string and return its table id.
fn label(seg: u32, list: &[String], table: &mut Vec<(String, i32)>, grapheme: bool) -> i32 {
    let s = if seg == 0 || seg == u32::MAX {
        // empty segment (deletion/insertion) or boundary: no-op symbol
        "_".to_string()
    } else {
        // ids start at 1 (0 = empty segment); list[k] holds id k+1
        let raw = &list[seg as usize - 1];
        if grapheme {
            raw.chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join("|")
        } else {
            raw.split(' ').collect::<Vec<_>>().join("|")
        }
    };
    if let Some(pos) = table.iter().position(|(t, _)| *t == s) {
        return pos as i32;
    }
    let id = table.len() as i32;
    table.push((s, id));
    id
}

fn inventories(
    train: &[(String, Vec<String>)],
    gmax: usize,
    pmax: usize,
) -> (
    HashMap<String, u32>,
    Vec<String>,
    HashMap<String, u32>,
    Vec<String>,
) {
    let mut gmap: HashMap<String, u32> = HashMap::new();
    let mut pmap: HashMap<String, u32> = HashMap::new();
    let mut glist: Vec<String> = Vec::new();
    let mut plist: Vec<String> = Vec::new();
    for (w, ph) in train {
        let chars: Vec<char> = w.chars().collect();
        for i in 0..chars.len() {
            for l in 1..=gmax {
                if i + l > chars.len() {
                    break;
                }
                let s: String = chars[i..i + l].iter().collect();
                if !gmap.contains_key(&s) {
                    // id 0 is reserved for the empty segment (deletions)
                    gmap.insert(s.clone(), glist.len() as u32 + 1);
                    glist.push(s);
                }
            }
        }
        for i in 0..ph.len() {
            for l in 1..=pmax {
                if i + l > ph.len() {
                    break;
                }
                let s: String = ph[i..i + l].join(" ");
                if !pmap.contains_key(&s) {
                    pmap.insert(s.clone(), plist.len() as u32 + 1);
                    plist.push(s);
                }
            }
        }
    }
    (gmap, glist, pmap, plist)
}

fn parse<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, flag: &str) -> T {
    if let Some(v) = args.next().and_then(|v| v.parse().ok()) {
        v
    } else {
        eprintln!("{flag} needs a value");
        std::process::exit(2);
    }
}

fn read_lexicon(path: &str) -> Vec<(String, Vec<String>)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        let Some((word, phones)) = line.split_once('\t') else {
            continue;
        };
        let word = word.trim().to_lowercase();
        let phones: Vec<String> = phones.split_whitespace().map(str::to_owned).collect();
        if word.is_empty() || phones.is_empty() || !seen.insert(word.clone()) {
            continue;
        }
        out.push((word, phones));
    }
    out
}

/// Viterbi M2M alignment of one word; returns the best pair path.
fn align(
    word: &str,
    phones: &[String],
    gmap: &HashMap<String, u32>,
    pmap: &HashMap<String, u32>,
    gmax: usize,
    pmax: usize,
    probs: &HashMap<(u32, u32), f64>,
) -> Vec<(u32, u32)> {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    let m = phones.len();
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let mut gids = vec![[INVALID; 3]; n];
    for i in 0..n {
        for l in 1..=gmax {
            if i + l > n {
                break;
            }
            let s: String = chars[i..i + l].iter().collect();
            if let Some(&id) = gmap.get(&s) {
                gids[i][l - 1] = id;
            }
        }
    }
    let mut pids = vec![[INVALID; 3]; m];
    for j in 0..m {
        for l in 1..=pmax {
            if j + l > m {
                break;
            }
            let s: String = phones[j..j + l].join(" ");
            if let Some(&id) = pmap.get(&s) {
                pids[j][l - 1] = id;
            }
        }
    }

    let stride = m + 1;
    let mut cost = vec![f64::INFINITY; (n + 1) * stride];
    let mut back: Vec<Option<(usize, u32, u32)>> = vec![None; (n + 1) * stride];
    cost[0] = 0.0;
    for i in 0..=n {
        for j in 0..=m {
            let c0 = cost[i * stride + j];
            if c0 == f64::INFINITY {
                continue;
            }
            for glen in 0..=gmax {
                if i + glen > n {
                    break;
                }
                let g = if glen == 0 { 0 } else { gids[i][glen - 1] };
                if glen > 0 && g == INVALID {
                    continue;
                }
                for plen in 0..=pmax {
                    if j + plen > m {
                        break;
                    }
                    if glen == 0 && plen == 0 {
                        continue;
                    }
                    let p = if plen == 0 { 0 } else { pids[j][plen - 1] };
                    if plen > 0 && p == INVALID {
                        continue;
                    }
                    let prob = probs.get(&(g, p)).copied().unwrap_or(FLOOR);
                    let nc = c0 - prob.ln();
                    let k = (i + glen) * stride + (j + plen);
                    if nc < cost[k] {
                        cost[k] = nc;
                        back[k] = Some((i * stride + j, g, p));
                    }
                }
            }
        }
    }
    let mut path = Vec::new();
    let mut k = n * stride + m;
    if cost[k] == f64::INFINITY {
        return path;
    }
    while let Some((prev, g, p)) = back[k] {
        path.push((g, p));
        k = prev;
    }
    path.reverse();
    path
}

/// Exact-match rate, mean PER, decode coverage over holdout pairs.
fn evaluate(model: &PhonetisaurusG2p, hold: &[(String, Vec<String>)]) -> (f64, f64, f64) {
    let mut exact = 0usize;
    let mut per_sum = 0.0f64;
    let mut covered = 0usize;
    let n = hold.len().max(1);
    for (w, ref_phones) in hold {
        if let Some(pred) = model.phonemize(w) {
            covered += 1;
            if pred == *ref_phones {
                exact += 1;
            }
            per_sum += edit_distance(&pred, ref_phones) as f64 / ref_phones.len().max(1) as f64;
        }
    }
    (
        exact as f64 / n as f64,
        per_sum / covered.max(1) as f64,
        covered as f64 / n as f64,
    )
}

fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1)
                .min(cur[j - 1] + 1)
                .min(prev[j - 1] + usize::from(a[i - 1] != b[j - 1]));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Tiny xorshift RNG (deterministic across platforms).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}
