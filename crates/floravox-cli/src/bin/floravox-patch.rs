//! `floravox patch` — native duration-graph surgery for ONNX voices.
//!
//! Rust port of `python/add_durations_output.py`: taps the duration
//! predictor's rounded output into a stable `"durations"` graph output
//! via an `Identity` node, so word/mark timings become measured instead of
//! estimated. No `Python`, no `PyTorch` — safe to run in installers and
//! download pipelines on end-user machines.
//!
//! Families: piper/MMS `VITS`, Matcha (acoustic model), Kokoro.
//!
//! Usage:
//!   floravox patch VOICE.onnx [-o OUT.onnx] [--validate]

// Hand-rolled protobuf surgery: casts and shapes are inherent.
#![allow(clippy::all, clippy::pedantic)]

use floravox_core::synth::ControlSymbols;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut validate = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" | "--output" => output = args.next().map(PathBuf::from),
            "--validate" => validate = true,
            "-h" | "--help" => {
                eprintln!("usage: floravox patch VOICE.onnx [-o OUT.onnx] [--validate]");
                return;
            }
            other => input = Some(PathBuf::from(other)),
        }
    }
    let Some(input) = input else {
        eprintln!("usage: floravox patch VOICE.onnx [-o OUT.onnx] [--validate]");
        std::process::exit(2);
    };
    let output = output.unwrap_or_else(|| input.clone());
    if let Err(e) = run(&input, &output, validate) {
        eprintln!("patch failed: {e:#}");
        std::process::exit(1);
    }
}

fn run(input: &PathBuf, output: &PathBuf, validate: bool) -> anyhow::Result<()> {
    let bytes = std::fs::read(input)?;
    let model = parse_model(&bytes)?;

    if has_output(&model, "durations") {
        eprintln!("already has a durations output; nothing to do");
        return Ok(());
    }

    let tensor = find_duration_tensor(&model)?;
    eprintln!("duration tensor: {tensor}");
    let patched = add_durations_output(model, &tensor)?;
    // serialize: we operate on the parsed representation via the onnx crate
    let out_bytes = serialize(&patched)?;
    std::fs::write(output, out_bytes)?;
    eprintln!("wrote {}", output.display());

    if validate {
        validate_model(output)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf handling. The wire format we need:
//   ModelProto { graph: GraphProto }
//   GraphProto { node: [NodeProto], output: [ValueInfoProto] }
//   NodeProto { op_type, input: [string], output: [string], name }
// We parse/emit protobuf with hand-rolled varint wire access (no onnx
// crate dependency; the format subset is stable).
// ---------------------------------------------------------------------------

/// Parsed model = raw bytes plus lazily-decoded pieces we edit.
struct RawModel {
    bytes: Vec<u8>,
}

fn parse_model(bytes: &[u8]) -> anyhow::Result<RawModel> {
    Ok(RawModel {
        bytes: bytes.to_vec(),
    })
}

/// protobuf field numbers
mod pb {
    // ONNX protobuf field numbers (onnx.proto3):
    // NodeProto { input=1, output=2, name=3, op_type=4 }
    // GraphProto { node=1, name=2, initializer=5, output=12 }
    // ModelProto { graph=7 }
    pub const MODEL_GRAPH: u32 = 7;
    pub const GRAPH_NODE: u32 = 1;
    pub const GRAPH_OUTPUT: u32 = 12;
    pub const NODE_OP_TYPE: u32 = 4;
    pub const NODE_INPUT: u32 = 1;
    pub const NODE_OUTPUT: u32 = 2;
    pub const NODE_NAME: u32 = 3;
}

fn has_output(m: &RawModel, name: &str) -> bool {
    // scan graph outputs for a value-info with this name (field 2 of
    // GraphProto -> ValueInfoProto.name field 1)
    let graph = find_field(&m.bytes, pb::MODEL_GRAPH).map(|r| &m.bytes[r]);
    let Some(graph) = graph else { return false };
    for out in iter_fields(graph, pb::GRAPH_OUTPUT) {
        let chunk = &graph[out];
        if let Some(n) = find_field(chunk, 1) {
            if &chunk[n.start..n.end] == name.as_bytes() {
                return true;
            }
        }
    }
    false
}

/// Find the duration tensor: the `Ceil` feeding a `CumSum`, else the `Round`
/// with `duration_proj` ancestry (kokoro); error when neither exists.
fn find_duration_tensor(m: &RawModel) -> anyhow::Result<String> {
    let graph = find_field(&m.bytes, pb::MODEL_GRAPH)
        .map(|r| &m.bytes[r])
        .ok_or_else(|| anyhow::anyhow!("model has no graph"))?;

    // collect nodes: (op_type, inputs, outputs)
    let mut nodes = Vec::new();
    for n in iter_fields(graph, pb::GRAPH_NODE) {
        let chunk = &graph[n];
        let mut op = String::new();
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        for (field, range) in iter_all_fields(chunk) {
            match field {
                pb::NODE_OP_TYPE => op = String::from_utf8_lossy(&chunk[range]).into_owned(),
                pb::NODE_INPUT => ins.push(String::from_utf8_lossy(&chunk[range]).into_owned()),
                pb::NODE_OUTPUT => outs.push(String::from_utf8_lossy(&chunk[range]).into_owned()),
                _ => {}
            }
        }
        nodes.push((op, ins, outs));
    }

    // Prefer a Ceil whose output reaches a CumSum within a few
    // intermediate ops (matcha: Ceil -> Mul -> Squeeze -> CumSum).
    for (op, _, outs) in &nodes {
        if op != "Ceil" {
            continue;
        }
        let Some(o) = outs.first() else { continue };
        let mut frontier: Vec<String> = vec![o.clone()];
        for _ in 0..4 {
            let mut next = Vec::new();
            for t in &frontier {
                for (idx, (nop, nins, _)) in nodes.iter().enumerate() {
                    if !nins.iter().any(|i| i == t) {
                        continue;
                    }
                    if nop.contains("CumSum") {
                        return Ok(o.clone());
                    }
                    next.extend(nodes[idx].2.iter().cloned());
                }
            }
            frontier = next;
        }
    }
    // Case 2 (kokoro): Round with duration_proj ancestry. Node names
    // live in field 5; walk producers checking both op-type and name.
    let mut names: Vec<String> = Vec::with_capacity(nodes.len());
    for n in iter_fields(graph, pb::GRAPH_NODE) {
        let chunk = &graph[n];
        let mut nm = String::new();
        for (field, range) in iter_all_fields(chunk) {
            if field == pb::NODE_NAME {
                nm = String::from_utf8_lossy(&chunk[range]).into_owned();
            }
        }
        names.push(nm);
    }
    let producer: HashMap<String, usize> = {
        let mut p = HashMap::new();
        for (i, (_, _, outs)) in nodes.iter().enumerate() {
            for o in outs {
                p.insert(o.clone(), i);
            }
        }
        p
    };
    for (idx, (op, _, outs)) in nodes.iter().enumerate() {
        if op != "Round" {
            continue;
        }
        let mut cur = idx;
        let mut depth = 0;
        loop {
            let hit =
                nodes[cur].0.contains("duration_proj") || names[cur].contains("duration_proj");
            if hit {
                return Ok(outs[0].clone());
            }
            let Some(first) = nodes[cur].1.first() else {
                break;
            };
            let Some(&p) = producer.get(first) else { break };
            cur = p;
            depth += 1;
            if depth > 20 {
                break;
            }
        }
    }
    anyhow::bail!("no duration tensor found (not a piper/mms VITS, Matcha, or Kokoro export)")
}

/// Append an Identity node + output entry tapping `tensor`.
fn add_durations_output(m: RawModel, tensor: &str) -> anyhow::Result<RawModel> {
    let graph_range = find_field(&m.bytes, pb::MODEL_GRAPH)
        .ok_or_else(|| anyhow::anyhow!("model has no graph"))?;

    // Build NodeProto for the tap: op_type=Identity, input=tensor,
    // output=durations, name=floravox_durations_tap
    let mut node = Vec::new();
    emit_string(&mut node, pb::NODE_OP_TYPE, "Identity");
    emit_string(&mut node, pb::NODE_INPUT, tensor);
    emit_string(&mut node, pb::NODE_OUTPUT, "durations");
    emit_string(&mut node, pb::NODE_NAME, "floravox_durations_tap");

    // Build a ValueInfoProto { name: "durations" } for graph.output
    let mut vi = Vec::new();
    emit_string(&mut vi, 1, "durations");

    // Rebuild the graph payload: original + tap node + output entry.
    let mut new_graph = Vec::with_capacity(graph_range.len() + node.len() + vi.len() + 16);
    new_graph.extend_from_slice(&m.bytes[graph_range.clone()]);
    emit_bytes(&mut new_graph, pb::GRAPH_NODE, &node);
    emit_bytes(&mut new_graph, pb::GRAPH_OUTPUT, &vi);

    // Splice into the model: replace the whole graph field (tag + length
    // + payload) with the rebuilt one.
    let span = find_field_span(&m.bytes, pb::MODEL_GRAPH)
        .ok_or_else(|| anyhow::anyhow!("model has no graph"))?;

    let mut out = Vec::with_capacity(m.bytes.len() + node.len() + vi.len() + 8);
    out.extend_from_slice(&m.bytes[..span.start]);
    emit_bytes(&mut out, pb::MODEL_GRAPH, &new_graph);
    out.extend_from_slice(&m.bytes[span.end..]);

    Ok(RawModel { bytes: out })
}

#[allow(clippy::unnecessary_wraps)]
fn serialize(m: &RawModel) -> anyhow::Result<Vec<u8>> {
    Ok(m.bytes.clone())
}

fn validate_model(path: &PathBuf) -> anyhow::Result<()> {
    // Load through our own backends: proves the patched file parses, the
    // family detector still recognizes it, and the durations output
    // exists. Full inference validation stays in the Python tool (CI);
    // here we check the structural invariant we rely on.
    let backend = floravox_core::load_voice(path)?;
    if !backend.config().has_durations {
        anyhow::bail!("patched model does not expose a durations output");
    }
    eprintln!(
        "validated: {} Hz, durations output present",
        backend.config().sample_rate
    );
    let _ = ControlSymbols::piper(); // keep import meaningful
    Ok(())
}

// ------------------------- protobuf primitives -------------------------

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct Range_ {
    start: usize,
    end: usize,
}

/// Iterate occurrences of `field` (length-delimited) in `buf`.
fn iter_fields<'a>(buf: &'a [u8], field: u32) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    let mut pos = 0usize;
    std::iter::from_fn(move || loop {
        if pos >= buf.len() {
            return None;
        }
        let (tag, n) = read_varint(buf, pos)?;
        pos += n;
        let f = (tag >> 3) as u32;
        let wt = tag & 7;
        if wt == 2 {
            let (len, n2) = read_varint(buf, pos)?;
            pos += n2;
            let start = pos;
            let end = start + len as usize;
            if end > buf.len() {
                return None;
            }
            pos = end;
            if f == field {
                return Some(start..end);
            }
        } else if wt == 0 {
            let (_, n2) = read_varint(buf, pos)?;
            pos += n2;
        } else if wt == 5 {
            pos += 4;
        } else if wt == 1 {
            pos += 8;
        } else {
            return None;
        }
    })
}

/// All (field, range) pairs of length-delimited fields.
fn iter_all_fields(buf: &[u8]) -> Vec<(u32, std::ops::Range<usize>)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < buf.len() {
        let Some((tag, n)) = read_varint(buf, pos) else {
            break;
        };
        pos += n;
        let f = (tag >> 3) as u32;
        let wt = tag & 7;
        if wt == 2 {
            let Some((len, n2)) = read_varint(buf, pos) else {
                break;
            };
            pos += n2;
            let start = pos;
            let end = start + len as usize;
            if end > buf.len() {
                break;
            }
            out.push((f, start..end));
            pos = end;
        } else if wt == 0 {
            let Some((_, n2)) = read_varint(buf, pos) else {
                break;
            };
            pos += n2;
        } else if wt == 5 {
            pos += 4;
        } else if wt == 1 {
            pos += 8;
        } else {
            break;
        }
    }
    out
}

fn find_field(buf: &[u8], field: u32) -> Option<std::ops::Range<usize>> {
    iter_fields(buf, field).next().map(|r| r.start..r.end)
}

/// Full byte span of the first occurrence of `field`, INCLUDING the tag
/// and length varints — a drop-in replacement range for splicing.
fn find_field_span(buf: &[u8], field: u32) -> Option<std::ops::Range<usize>> {
    let mut pos = 0usize;
    while pos < buf.len() {
        let tag_start = pos;
        let Some((tag, n)) = read_varint(buf, pos) else {
            return None;
        };
        pos += n;
        let f = (tag >> 3) as u32;
        let wt = tag & 7;
        if wt == 2 {
            let Some((len, n2)) = read_varint(buf, pos) else {
                return None;
            };
            pos += n2;
            let end = pos + len as usize;
            if end > buf.len() {
                return None;
            }
            if f == field {
                return Some(tag_start..end);
            }
            pos = end;
        } else if wt == 0 {
            let Some((_, n2)) = read_varint(buf, pos) else {
                return None;
            };
            pos += n2;
        } else if wt == 5 {
            pos += 4;
        } else if wt == 1 {
            pos += 8;
        } else {
            return None;
        }
    }
    None
}

fn read_varint(buf: &[u8], pos: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    let mut n = 0;
    loop {
        let b = *buf.get(pos + n)?;
        value |= u64::from(b & 0x7f) << shift;
        n += 1;
        if b & 0x80 == 0 {
            return Some((value, n));
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

fn emit_varint(out: &mut Vec<u8>, v: u64) {
    let mut v = v;
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

fn emit_bytes(out: &mut Vec<u8>, field: u32, payload: &[u8]) {
    emit_varint(out, u64::from(field) << 3 | 2);
    emit_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn emit_string(out: &mut Vec<u8>, field: u32, s: &str) {
    emit_bytes(out, field, s.as_bytes());
}
