#!/usr/bin/env python3
"""Extract a gruut lexicon.db (MIT, rhasspy/gruut-lang-*) to floravox TSV.

gruut is the phonemizer piper's non-English voices were trained with, so
its lexicon symbols map (nearly) directly onto those voices' phoneme
inventories — floravox's symbol resolver handles the stragglers (length
marks, tie bars, ASCII homoglyphs, unsupported diacritics).

Usage:
  python gruut2tsv.py lexicon.db de.tsv [--min-count N]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    db = sqlite3.connect(str(args.db))
    # Multiple pronunciations per word are ordered (pron_order); keep the
    # first (default) per word — the FST lexicon is one-pronunciation.
    rows = db.execute(
        "SELECT word, phonemes FROM word_phonemes WHERE pron_order = 0 "
        "ORDER BY word COLLATE NOCASE",
    )
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for word, pron in rows:
            pron = " ".join(pron.split())
            if word and pron:
                f.write(f"{word}\t{pron}\n")
                n += 1
    print(f"wrote {n} entries to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
