#!/usr/bin/env python3
"""Write multiple best_sep trees from T0–T4 picks. No Presence gate here."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.export_groups import GROUP_SPECS, export_one  # noqa: E402
from kws.iojson import load_jsonl, write_json, write_jsonl, limit_rows_balanced  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--picks", type=Path, default=ROOT / "reports" / "t0_t4_picks.jsonl")
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--data-dir", type=Path, default=Path(r"d:\media\datasetA"))
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(r"d:\media\pos_neg\best_sep_groups"),
        help="parent dir; each group becomes {out-root}/{name}/",
    )
    p.add_argument(
        "--group",
        action="append",
        default=[],
        help="repeatable. default: raw_kws,t0,t2,skip_then_t0,skip_then_t2",
    )
    p.add_argument("--se-backend", default="none", help="none | spectral (for t1/t4 groups)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--summary", type=Path, default=ROOT / "reports" / "best_sep_groups.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.picks)
    if args.limit:
        rows = limit_rows_balanced(rows, args.limit)
    default_groups = ["raw_kws", "t0", "t2", "skip_then_t0", "skip_then_t2"]
    groups = args.group or default_groups
    known = {n for n, _ in GROUP_SPECS}
    for g in groups:
        if g not in known:
            raise SystemExit(f"unknown group {g}; known={sorted(known)}")
    if any(g in ("t1_spectral", "t4_spectral") for g in groups) and args.se_backend in (
        "none",
        "off",
        "",
    ):
        args.se_backend = "spectral"
        print("[INFO] t1/t4 requested → se-backend=spectral", flush=True)

    summary: dict = {"groups": {}, "n": len(rows)}
    args.out_root.mkdir(parents=True, exist_ok=True)
    for g in groups:
        dest = args.out_root / g
        dest.mkdir(parents=True, exist_ok=True)
        index: list[dict] = []
        n_ok = 0
        n_miss = 0
        n_se = 0
        chosen_counts: dict[str, int] = {}
        for rec in rows:
            row = export_one(
                rec,
                g,
                dest,
                pos_neg=args.pos_neg,
                data_dir=args.data_dir,
                se_backend=args.se_backend,
            )
            index.append(row)
            if row.get("ok"):
                n_ok += 1
            else:
                n_miss += 1
            if row.get("se_applied"):
                n_se += 1
            ch = str(row.get("chosen") or "")
            chosen_counts[ch] = chosen_counts.get(ch, 0) + 1
        write_jsonl(dest / "index.jsonl", index)
        summary["groups"][g] = {
            "path": str(dest.resolve()),
            "n_ok": n_ok,
            "n_missing": n_miss,
            "n_se_applied": n_se,
            "chosen_counts": chosen_counts,
        }
        print(f"[OK] {g} n_ok={n_ok} missing={n_miss} se={n_se} → {dest}", flush=True)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
