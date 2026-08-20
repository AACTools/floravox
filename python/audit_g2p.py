#!/usr/bin/env python3
"""Audit floravox G2P output against espeak-ng (what piper voices were
trained on).

Runs `floravox g2p --lexicon en_US --phonetisaurus MODEL` for a word list
and compares the symbol sequences with `espeak-ng --ipa -v en-us`,
splitting the espeak output with the same clustering rules the ingest
module uses (combining marks attach, stress marks standalone).

Reports exact-match rate, symbol edit distance, and where they diverge.

Usage:
  python audit_g2p.py --floravox floravox-bin --lexicon /tmp/en_US \
      --phonetisaurus /tmp/phonetisaurus.fst [--n 300]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Common words (in-CMUDict) and names/OOV (not in CMUDict).
COMMON = """the of and a to in is you that it he was for on are as with his they I at be
this have from or one had by word but not what all were we when your can said there
use an each which she do how their if will up other about out many then them these so
some her would make like him into time has look two more write go see number no way
could people my than first water been call who oil its now find long down day did get
come made may part over new sound take only little work know place year live me back
give most very after thing our just name good sentence man think say great where help
through much before line right too mean old any same tell boy follow came want show
also around form three small set put end does another well large must big even such
because turn here why ask went men read need land different home us move try kind
hand picture again change off play spell air away animal house point page letter
mother answer found study still learn should America world""".split()

OOV_NAMES = """Kokoro Floravox Sherpa Onnx Piper Matcha Vocoder Willwade Hannover
Groningen Aachen Zermatt Kyiv Oulu Tromso Reykjavik Wroclaw Gdansk Brno
Ljubljana Split Zadar Novigrad Sopot Kotlin Rustler Bazel Carnix Zephyr
Aldrich Brom Vander Calyx Doran Elowen Fyren Galen Hollis Ithra Jorven""".split()


def espeak_ipa(word: str) -> list[str]:
    out = subprocess.run(
        ["espeak-ng", "--ipa", "-v", "en-us", "-q", word],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # Same clustering as ingest::ipa_tokens: combining marks + IPA modifier
    # range attach to the previous symbol; stress marks standalone.
    symbols: list[str] = []
    for ch in out:
        cp = ord(ch)
        if ch in "ˈˌ.":
            symbols.append(ch)
        elif 0x02B0 <= cp <= 0x02FF or 0x0300 <= cp <= 0x036F:
            if symbols:
                symbols[-1] += ch
        elif ch.isalpha():
            symbols.append(ch)
    return symbols


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floravox", default="floravox")
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--phonetisaurus", required=True)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    words = (COMMON[: args.n] + OOV_NAMES)[: args.n + 30]
    res = subprocess.run(
        [args.floravox, "g2p", "--lexicon", args.lexicon,
         "--phonetisaurus", args.phonetisaurus, *words],
        capture_output=True, text=True, check=True,
    )
    ours = {}
    for line in res.stdout.splitlines():
        w, _, phones = line.partition("\t")
        ours[w] = phones.split()

    n_exact = n_tot = 0
    dists: list[int] = []
    diverged = []
    for w in words:
        got = ours.get(w)
        if got is None:
            print(f"SKIP {w}: no floravox output")
            continue
        ref = espeak_ipa(w)
        n_tot += 1
        d = edit_distance(got, ref)
        dists.append(d)
        if d == 0:
            n_exact += 1
        else:
            diverged.append((w, got, ref))

    dists.sort()
    med = dists[len(dists) // 2]
    mean = sum(dists) / len(dists)
    print(f"\nwords={n_tot} exact={n_exact} ({100 * n_exact / n_tot:.0f}%) "
          f"median_edit={med} mean_edit={mean:.2f}")
    print("\nsample divergences (ours vs espeak-ng):")
    for w, got, ref in diverged[:20]:
        print(f"  {w:12s} ours={' '.join(got)}")
        print(f"  {'':12s} espk={' '.join(ref)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
