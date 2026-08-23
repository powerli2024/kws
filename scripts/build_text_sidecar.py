#!/usr/bin/env python3
"""Write the T2 ranking sidecar: known-wake continuous confidence.

This is NOT MMS-FA. It is a frozen-text feature: token NLL / coverage / margin.
Do not invent q_kw from CER (4-char CER is too discrete).

Until a forced-decode dump exists, pass an already-scored jsonl with
  {"uid": "...", "q_kw": {"original": 0.9, "spk1": 0.2, "spk2": 0.1}}
or {"uid": "...", "nll": {...}} (lower NLL is better; the loader negates it).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.sidecar import load_qkw_sidecar  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from-jsonl", type=Path, required=True, help="precomputed q_kw or nll sidecar")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "sidecars" / "q_kw.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.from_jsonl.is_file():
        raise SystemExit(
            f"missing {args.from_jsonl}. Build q_kw with a frozen ASR forced-decode "
            "(known wake text), not from CER and not from cos(track, raw)."
        )
    n = len(load_qkw_sidecar(args.from_jsonl))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.from_jsonl, args.out)
    print(f"[OK] validated q_kw/nll sidecar n={n} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
