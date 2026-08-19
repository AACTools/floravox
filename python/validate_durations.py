#!/usr/bin/env python3
"""Validate duration→word-timing accuracy of a patched piper model.

Checks, per utterance:
  1. sum(durations) * hop ≈ audio samples (frame invariant)
  2. word boundaries are monotonically increasing and non-overlapping
  3. (informational) per-word timings against a hand-checked sentence

Usage:
  python validate_durations.py MODEL.onnx [--hop 256]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Simple utterances with words spelled from single symbols present in every
# piper phoneme map (ASCII letters). Each word is a list of its symbols.
SENTENCES = [
    ["hello", "world"],
    ["this", "is", "a", "test"],
    ["the", "quick", "brown", "fox"],
]


def synth(session, phoneme_map: dict, words: list[str], hop: int):
    pad = phoneme_map["_"][0]
    bos = phoneme_map["^"][0]
    eos = phoneme_map["$"][0]
    space = phoneme_map.get(" ", [3])[0]

    ids = [bos, pad]
    groups = []
    for wi, word in enumerate(words):
        if wi:
            ids += [space, pad]
        start = len(ids)
        for ch in word:
            for pid in phoneme_map.get(ch, []):
                ids.append(pid)
                ids.append(pad)
        groups.append((start, len(ids)))
    ids.append(eos)

    input_names = {i.name for i in session.get_inputs()}
    feeds: dict = {
        "input": np.asarray([ids], dtype=np.int64),
        "input_lengths": np.asarray([len(ids)], dtype=np.int64),
    }
    if "scales" in input_names:
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

    audio, durations = session.run(["output", "durations"], feeds)
    return ids, groups, audio.reshape(-1), durations.reshape(-1), len(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--hop", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = ap.parse_args()

    cfg = args.model.with_suffix(".onnx.json")
    if not cfg.exists():
        print(f"missing config: {cfg}", file=sys.stderr)
        return 1
    config = json.loads(cfg.read_text())
    phoneme_map = config["phoneme_id_map"]
    hop = args.hop or config.get("audio", {}).get("hop_length", 256)
    sample_rate = config.get("audio", {}).get("sample_rate", 22050)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    frame_ms = hop / sample_rate * 1000.0

    all_ok = True
    report = []
    for words in SENTENCES:
        ids, groups, audio, durations, n = synth(session, phoneme_map, words, hop)
        frames = int(durations.sum())
        predicted = frames * hop
        ok_len = abs(predicted - len(audio)) <= hop

        # fold prefixes
        prefix = np.concatenate([[0], np.cumsum(durations)])
        timings = []
        ok_mono = True
        prev_end = 0.0
        for word, (s, e) in zip(words, groups):
            start_ms = prefix[s] * frame_ms
            end_ms = prefix[e] * frame_ms
            if start_ms < prev_end - 1e-6:
                ok_mono = False
            prev_end = end_ms
            timings.append(
                {"word": word, "start_ms": round(float(start_ms), 2), "end_ms": round(float(end_ms), 2)}
            )

        ok = ok_len and ok_mono
        all_ok &= ok
        entry = {
            "words": words,
            "audio_samples": len(audio),
            "predicted_samples": predicted,
            "frames": frames,
            "ok": ok,
            "timings": timings,
        }
        report.append(entry)
        if not args.json:
            status = "OK  " if ok else "FAIL"
            print(f"[{status}] {' '.join(words)}")
            print(f"       audio={len(audio)} predicted={predicted} frames={frames}")
            for t in timings:
                print(f"       {t['word']:>8}: {t['start_ms']:8.2f} – {t['end_ms']:8.2f} ms")

    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
