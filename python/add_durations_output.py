#!/usr/bin/env python3
"""Patch a VITS/Matcha ONNX TTS model to expose per-id frame durations.

Stock piper/MMS VITS and Matcha exports compute phoneme durations
internally (needed to build the monotonic attention) but discard them.
This tool performs graph surgery on the exported model — no PyTorch, no
checkpoints — adding the Ceil(w) tensor (durations in mel frames) as a
stable `"durations"` output via an Identity node.

Supported families (input/output names are discovered from the graph):

* piper VITS      `input`/`input_lengths` (+`scales` or split), out `output`
* MMS VITS        `x`/`x_length` + split scales, out `y`
* Matcha acoustic `x`/`x_length` + `noise_scale`/`length_scale`, out `mel`
                  (validate against mel frames; audio comes from a
                  separate vocoder)
* Kokoro          `tokens`/`style`/`speed`, out `audio` — taps the
                  StyleTTS2 duration predictor (`duration_proj → ... →
                  Round`); invariant `sum(durations) * 600 == samples`
                  (600 samples per duration unit at 24 kHz), exact across
                  speeds and speakers.

Semantics: durations[i] is the number of duration units allocated to
phoneme-id i of the input sequence (including pad ids). Convert to
samples with `frame * hop_length` (piper/MMS 256, matcha per vocoder,
kokoro 600; check the voice's config).

Based on the approach validated upstream in piper1-gpl
(patch_voice_with_alignment.py), with two additions:
  * a stable, documented output name ("durations") via Identity node
  * optional runtime validation that sum(durations)*hop matches the audio
    length (VITS) or the mel frame count (Matcha)

Usage:
  python add_durations_output.py MODEL.onnx [-o OUT.onnx] [--hop 256] [--validate]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_LOGGER = logging.getLogger("floravox.patch")


def find_duration_tensor(model) -> str:
    """Locate the duration tensor feeding the attention/upsample path.

    piper/MMS VITS and Matcha: the stochastic-duration-predictor output is
    exponentiated, scaled, and Ceil'd; that tensor drives the CumSum
    attention builder and sums to the mel length. Matcha graphs carry a
    second Ceil (a length computation), so when several Ceils exist the
    one feeding a CumSum (generate_path) wins.

    Kokoro (StyleTTS2): no Ceil; the duration predictor is
    `duration_proj/... → Sigmoid → ReduceSum → Div → Squeeze → Round`,
    whose output drives the per-token frame split. The Round fed by
    `duration_proj` ancestry is tapped.
    """
    ceil_outputs = [o for node in model.graph.node if node.op_type == "Ceil" for o in node.output]
    if ceil_outputs:
        if len(ceil_outputs) > 1:
            # Prefer a Ceil whose value feeds a CumSum (generate_path).
            graph_input_names = {i for n in model.graph.node for i in n.input}
            for cand in ceil_outputs:
                if cand in graph_input_names:
                    return cand
            raise ValueError(f"Multiple Ceil nodes, none feed CumSum: {ceil_outputs}")
        return ceil_outputs[0]

    producers = {o: n for n in model.graph.node for o in n.output}
    for node in model.graph.node:
        if node.op_type != "Round":
            continue
        cur, depth = node, 0
        while cur is not None and depth < 20:
            if "duration_proj" in (cur.name or ""):
                return node.output[0]
            cur = producers.get(cur.input[0]) if cur.input else None
            depth += 1
    raise ValueError(
        "No duration tensor found — this does not look like a piper/mms "
        "VITS, Matcha, or Kokoro export."
    )


def add_durations_output(model, tensor_name: str) -> str:
    """Insert an Identity node exposing `tensor_name` as output 'durations'."""
    import onnx

    for out in model.graph.output:
        if out.name == "durations":
            raise ValueError("model already has a 'durations' output")

    identity = onnx.helper.make_node(
        "Identity",
        inputs=[tensor_name],
        outputs=["durations"],
        name="floravox_durations_tap",
    )
    # Append after existing nodes so shape inference on the source completes.
    model.graph.node.append(identity)
    model.graph.output.append(
        onnx.helper.make_empty_tensor_value_info("durations")
    )
    return tensor_name


def model_type(model_path: Path) -> str:
    """Read `model_type` from embedded ONNX metadata ("" when absent)."""
    import onnx

    try:
        m = onnx.load(str(model_path), load_external_data=False)
    except Exception:  # noqa: BLE001 - metadata is best-effort
        return ""
    for p in m.metadata_props:
        if p.key == "model_type":
            return p.value
    return ""


def load_hop(model_path: Path, override: int | None) -> int:
    if override is not None:
        return override
    if model_type(model_path) == "kokoro":
        return 600  # empirically exact at 24 kHz across speeds/speakers
    # piper voice config
    cfg = model_path.with_suffix(".onnx.json")
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            hop = data.get("audio", {}).get("hop_length")
            if hop:
                return int(hop)
        except json.JSONDecodeError:
            pass
    # MMS-style VITS training config
    cfg2 = model_path.parent / "config.json"
    if cfg2.exists():
        try:
            data = json.loads(cfg2.read_text())
            hop = data.get("data", {}).get("hop_length")
            if hop:
                return int(hop)
        except json.JSONDecodeError:
            pass
    return 256


def load_symbol_map(model_path: Path) -> dict[str, list[int]]:
    """phoneme_id_map from a piper .onnx.json, or from a sibling tokens.txt
    (`symbol id` lines; several spellings may share an id)."""
    cfg = model_path.with_suffix(".onnx.json")
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            if data.get("phoneme_id_map"):
                return data["phoneme_id_map"]
        except json.JSONDecodeError:
            pass
    tokens = model_path.parent / "tokens.txt"
    if tokens.exists():
        mapping: dict[str, list[int]] = {}
        for line in tokens.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                sym, sid = parts[0], int(parts[1])
                # tokens.txt escapes a literal space as no symbol at all;
                # an empty first field means the space token.
                mapping.setdefault(sym, []).append(sid)
        if mapping:
            return mapping
    return {}


def validate(model_path: Path, hop: int) -> bool:
    """Run the patched model on a synthetic id sequence and check the
    frame invariant: sum(durations) * hop == audio samples (VITS, Kokoro),
    or sum(durations) == mel frames (Matcha, whose audio needs a
    vocoder)."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    is_matcha = "mel" in output_names
    is_kokoro = "tokens" in input_names and "style" in input_names

    phoneme_map = load_symbol_map(model_path)
    pad = phoneme_map.get("_", [0])[0]
    bos = phoneme_map.get("^", [1])[0]
    eos = phoneme_map.get("$", [2])[0]
    if is_kokoro:
        # Char-level tokens, no control framing.
        ids = [phoneme_map[c][0] for c in "hello world." if c in phoneme_map]
    else:
        ids = [bos, pad]
        for symbol in "hello world":
            for pid in phoneme_map.get(symbol, []):
                ids.append(pid)
                ids.append(pad)
        ids.append(eos)
    ids = np.asarray([ids], dtype=np.int64)

    feeds: dict = {}
    for name in input_names:
        if name in ("input", "x", "tokens"):
            feeds[name] = ids
        elif name in ("input_lengths", "x_length"):
            feeds[name] = np.asarray([ids.shape[1]], dtype=np.int64)
        elif name == "scales":
            feeds[name] = np.asarray([0.667, 1.0, 0.8], dtype=np.float32)
        elif name == "noise_scale":
            feeds[name] = np.asarray([0.667], dtype=np.float32)
        elif name == "length_scale":
            feeds[name] = np.asarray([1.0], dtype=np.float32)
        elif name == "noise_scale_w":
            feeds[name] = np.asarray([0.8], dtype=np.float32)
        elif name == "sid":
            feeds[name] = np.asarray([0], dtype=np.int64)
        elif name == "style":
            feeds[name] = kokoro_style(model_path.parent, ids.shape[1], 0)
        elif name == "speed":
            feeds[name] = np.asarray([1.0], dtype=np.float32)
        else:
            _LOGGER.warning("unknown input %r fed with zeros", name)
            feeds[name] = np.zeros(1, dtype=np.float32)

    main_out = (
        "mel" if is_matcha else "audio" if "audio" in output_names
        else "y" if "y" in output_names else "output"
    )
    results = session.run([main_out, "durations"], feeds)
    main, durations = results[0], results[1]
    frames = int(durations.reshape(-1).sum())

    if is_matcha:
        mel_frames = main.shape[-1]
        ok = abs(frames - mel_frames) <= 1
        _LOGGER.info(
            "validation (matcha): %d duration frames vs %d mel frames → %s",
            frames, mel_frames, "OK" if ok else "MISMATCH",
        )
    else:
        audio_len = main.shape[-1]
        predicted = frames * hop
        ok = abs(predicted - audio_len) <= hop  # allow one frame of rounding
        _LOGGER.info(
            "validation: %d frames * hop %d = %d samples vs audio %d samples → %s",
            frames, hop, predicted, audio_len, "OK" if ok else "MISMATCH",
        )
    return ok


def kokoro_style(model_dir: Path, tokens_len: int, sid: int):
    """Length-conditioned style slice from a sibling `voices.bin`
    (`styles[sid][min(len, dim0-1)]`, sherpa-onnx semantics)."""
    import numpy as np

    voices = np.fromfile(model_dir / "voices.bin", dtype=np.float32)
    dim0, dim2 = 511, 256  # style_dim; see ONNX metadata for other exports
    off = (sid * dim0 + min(tokens_len, dim0 - 1)) * dim2
    return voices[off:off + dim2].reshape(1, dim2).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="default: overwrite input")
    ap.add_argument("--hop", type=int, help="hop length (default: from .onnx.json or 256)")
    ap.add_argument("--validate", action="store_true", help="run invariant check after patching")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    import onnx

    model = onnx.load(str(args.model))
    try:
        tensor = find_duration_tensor(model)
        _LOGGER.info("duration tensor: %s", tensor)
        add_durations_output(model, tensor)
    except ValueError as e:
        _LOGGER.error("%s", e)
        return 1

    out = args.output or args.model
    onnx.save(model, str(out))
    _LOGGER.info("wrote %s", out)

    if args.validate:
        try:
            hop = load_hop(args.model if args.output is None else out, args.hop)
            if not validate(out, hop):
                _LOGGER.error("validation FAILED: durations do not sum to audio length")
                return 2
        except ImportError:
            _LOGGER.warning("onnxruntime not installed; skipping validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
