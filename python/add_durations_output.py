#!/usr/bin/env python3
"""Patch a piper-family VITS ONNX model to expose per-id frame durations.

Stock piper ONNX exports compute phoneme durations internally (needed to
build the monotonic attention) but discard them. This tool performs graph
surgery on the exported model — no PyTorch, no checkpoints — adding the
Ceil(w) tensor (durations in mel frames) as a stable `"durations"` output
via an Identity node.

Semantics: durations[i] is the number of mel frames allocated to phoneme-id
i of the input sequence (including pad ids). Convert to samples with
`frame * hop_length` (default hop 256; check the voice's .onnx.json).

Based on the approach validated upstream in piper1-gpl
(patch_voice_with_alignment.py), with two additions:
  * a stable, documented output name ("durations") via Identity node
  * optional runtime validation that sum(durations)*hop matches the audio length

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
    """Locate the Ceil(w) tensor feeding the attention path.

    In piper VITS exports the stochastic-duration-predictor output is
    exponentiated, scaled, and Ceil'd; that tensor both drives the CumSum
    attention builder and sums to the mel length. There is normally exactly
    one Ceil node in the graph.
    """
    ceil_outputs = [o for node in model.graph.node if node.op_type == "Ceil" for o in node.output]
    if not ceil_outputs:
        raise ValueError(
            "No Ceil node found — this does not look like a piper VITS export. "
            "Matcha/other architectures need their own extractor."
        )
    if len(ceil_outputs) > 1:
        # Prefer a Ceil whose value feeds a CumSum (generate_path).
        graph_input_names = {i for n in model.graph.node for i in n.input}
        for cand in ceil_outputs:
            if cand in graph_input_names:
                return cand
        raise ValueError(f"Multiple Ceil nodes, none feed CumSum: {ceil_outputs}")
    return ceil_outputs[0]


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


def load_hop(model_path: Path, override: int | None) -> int:
    if override is not None:
        return override
    cfg = model_path.with_suffix(".onnx.json")
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            hop = data.get("audio", {}).get("hop_length")
            if hop:
                return int(hop)
        except json.JSONDecodeError:
            pass
    return 256


def validate(model_path: Path, hop: int) -> bool:
    """Run the patched model on a synthetic id sequence and check the
    frame invariant: sum(durations) * hop == audio sample count."""
    import numpy as np
    import onnxruntime as ort

    cfg_path = model_path.with_suffix(".onnx.json")
    phoneme_map = {}
    if cfg_path.exists():
        phoneme_map = json.loads(cfg_path.read_text()).get("phoneme_id_map", {})

    pad = phoneme_map.get("_", [0])[0]
    bos = phoneme_map.get("^", [1])[0]
    eos = phoneme_map.get("$", [2])[0]
    # "hello world" via letters, or fall back to BOS/EOS only.
    ids = [bos, pad]
    for symbol in "hello world":
        for pid in phoneme_map.get(symbol, []):
            ids.append(pid)
            ids.append(pad)
    ids.append(eos)
    ids = np.asarray([ids], dtype=np.int64)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}
    feeds: dict = {
        "input": ids,
        "input_lengths": np.asarray([ids.shape[1]], dtype=np.int64),
    }
    if "scales" in input_names:
        # Old piper export style: one [noise_scale, length_scale, noise_scale_w] tensor.
        feeds["scales"] = np.asarray([0.667, 1.0, 0.8], dtype=np.float32)
    else:
        feeds.update(
            {
                "noise_scale": np.asarray([0.667], dtype=np.float32),
                "length_scale": np.asarray([1.0], dtype=np.float32),
                "noise_scale_w": np.asarray([0.8], dtype=np.float32),
            }
        )
    if "sid" in input_names:
        feeds["sid"] = np.asarray([0], dtype=np.int64)

    results = session.run(["output", "durations"], feeds)
    audio, durations = results[0], results[1]
    audio_len = audio.shape[-1]
    frames = int(durations.reshape(-1).sum())
    predicted = frames * hop
    ok = abs(predicted - audio_len) <= hop  # allow one frame of rounding
    _LOGGER.info(
        "validation: %d frames * hop %d = %d samples vs audio %d samples → %s",
        frames, hop, predicted, audio_len, "OK" if ok else "MISMATCH",
    )
    return ok


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
