#!/usr/bin/env python3
"""Evaluate floravox timing accuracy against audible speech energy.

Ground truth trick: `<break>` inserts *exact zero* samples, giving
verifiable silence edges. For each synthesized utterance this script
checks:

1. BREAK ALIGNMENT — the sample where a break starts/ends must sit at a
   silence edge (energy before/after the boundary).
2. ONSET/OFFSET ATTRIBUTION — does the first word's measured start land
   at the first audible sample (or is leading BOS/PAD silence wrongly
   attributed to it)? Same for trailing silence on the last word.
3. FLUENT BOUNDARY DIPS — inside fluent speech, energy at each predicted
   word boundary vs the surrounding speech (soft check; space-token
   gaps should be relative dips).
4. ESTIMATION ERROR — what the stock-voice proportional fallback would
   have reported vs the measured truth, per word.

Usage:
  python eval_timings.py --label NAME --wav F.wav --events F.json [--est]
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, rate


def rms_envelope(x: np.ndarray, rate: int, win_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    win = max(1, int(rate * win_ms / 1000))
    n = len(x) // win
    env = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    times = (np.arange(n) + 0.5) * win
    return env, times


def first_audible(env: np.ndarray, thresh: float) -> int:
    idx = np.argmax(env > thresh)
    return int(idx) if (env > thresh).any() else 0


def last_audible(env: np.ndarray, thresh: float) -> int:
    hits = np.where(env > thresh)[0]
    return int(hits[-1]) if len(hits) else len(env) - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--est", action="store_true",
                    help="also score the proportional-estimate fallback")
    args = ap.parse_args()

    audio, rate = load_wav(args.wav)
    env, env_t = rms_envelope(audio, rate)
    win = int(rate * 10 / 1000)  # env window in samples
    peak = env.max() if len(env) else 0.0
    thresh = peak * 0.02  # 2% of peak = "audible"

    evs = json.loads(args.events.read_text())
    words = [e for e in evs if e["type"] == "word_boundary"]
    breaks = [e for e in evs if e["type"] == "break_started"]
    print(f"== {args.label} ({rate} Hz, {len(audio)} samples, "
          f"{len(words)} words, {len(breaks)} breaks)")

    # 1. break edges are silence edges
    for b in breaks:
        s = b["sample"]
        before = env[max(0, (s // win) - 2): s // win].mean() if s // win >= 1 else 0.0
        after_idx = min(len(env) - 1, (s + b["ms"] * rate // 1000) // win + 1)
        after = env[after_idx] if len(env) else 0.0
        # inside break = silence, before break = speech
        in_brk = s + 10 * rate // 1000
        inside = env[min(len(env) - 1, in_brk // win)]
        ok = inside < thresh <= max(before, 1e-9)
        print(f"   break@{s}: energy before={before:.4f} inside={inside:.4f} "
              f"after={after:.4f} thresh={thresh:.4f} -> {'OK' if ok else 'CHECK'}")

    # 2. onset/offset attribution
    if words:
        fa = first_audible(env, thresh) * win
        la = (last_audible(env, thresh) + 1) * win
        lead_off = words[0]["sample_start"] - fa
        trail_off = words[-1]["sample_end"] - la
        print(f"   onset : predicted {words[0]['sample_start']} vs audible {fa} "
              f"(offset {lead_off / rate * 1000:+.0f} ms)")
        print(f"   offset: predicted {words[-1]['sample_end']} vs audible {la} "
              f"(offset {trail_off / rate * 1000:+.0f} ms)")

    # 3. fluent boundary dips
    dips = []
    for a, b in zip(words, words[1:]):
        s = a["sample_end"]
        at = env[min(len(env) - 1, s // win)]
        around = env[max(0, s // win - 5): s // win + 5]
        dips.append(at / (around.mean() + 1e-9))
    if dips:
        print(f"   fluent boundaries: energy-at-boundary / surrounding = "
              f"{np.median(dips):.2f} (median), max {max(dips):.2f} "
              f"(<1.0 means boundary sits in a dip)")

    # 4. estimation error
    if args.est and words:
        weights = [max(1, len(w["text"])) for w in words]
        total_w = sum(weights)
        total = max(w["sample_end"] for w in words)
        cursor = 0
        errs = []
        for i, w in enumerate(words):
            ln = total - cursor if i == len(words) - 1 else max(1, total * weights[i] // total_w)
            errs.append(abs(cursor - w["sample_start"]) / rate * 1000)
            cursor += ln
        print(f"   estimate fallback: median |start error| = {np.median(errs):.0f} ms, "
              f"p90 = {np.percentile(errs, 90):.0f} ms, max = {max(errs):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
