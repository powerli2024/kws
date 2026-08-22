#!/usr/bin/env python3
"""Rebuild an enriched best_sep index: keep CER oracle, write every track CER.

Skips MMS-FA. Does not re-run BSS; reads existing stage indexes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.dual_zero import classify_streams, skip_sep_feasibility  # noqa: E402
from kws.handoff import HandoffError, find_handoff, load_handoff  # noqa: E402
from kws.iojson import index_by_uid, load_jsonl, stage_index_path, write_jsonl  # noqa: E402
from kws.oracle import oracle_of  # noqa: E402
from kws.skip_sep import skip_sep_after_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--best-sep", type=Path, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "best_sep_enriched.jsonl",
    )
    p.add_argument(
        "--require-handoff",
        action="store_true",
        help="fail if kws_handoff.json is missing (extract@sep contract)",
    )
    p.add_argument(
        "--allow-legacy",
        action="store_true",
        help="allow a pos_neg tree without kws_handoff.json (pre-sep-branch dumps)",
    )
    return p.parse_args()


def _check_handoff(pos_neg: Path, *, require: bool, allow_legacy: bool) -> None:
    path = find_handoff(pos_neg)
    if path is not None:
        info = load_handoff(path)
        print(f"[OK] handoff {path} schema={info.get('schema')} mms_fa={info.get('mms_fa')}")
        return
    msg = (
        f"no kws_handoff.json under {pos_neg}. "
        "Redo BSS with git clone -b sep https://github.com/powerli2024/extract.git "
        "then ./run_sep.sh (see docs/PIPELINE.md)."
    )
    if require and not allow_legacy:
        raise HandoffError(msg)
    print(f"[WARN] {msg}")


def main() -> None:
    args = parse_args()
    try:
        _check_handoff(
            args.pos_neg,
            require=args.require_handoff,
            allow_legacy=args.allow_legacy,
        )
    except HandoffError as e:
        raise SystemExit(f"[ERR] {e}") from e
    best_sep = args.best_sep or (args.pos_neg / "best_sep" / "index.jsonl")
    winners = load_jsonl(best_sep)
    cache: dict[tuple[str, str], dict[str, dict]] = {}
    out = []
    n_mismatch = 0
    for rec in winners:
        split, uid, stage = rec["split"], rec["uid"], rec["best_stage"]
        key = (split, stage)
        if key not in cache:
            path = stage_index_path(args.pos_neg, split, stage)
            cache[key] = index_by_uid(load_jsonl(path)) if path.is_file() else {}
        srec = cache[key].get(uid) or {}
        streams = srec.get("streams") or {}
        recomputed = None
        if streams:
            packed = {k: {"cer": float(v["cer"])} for k, v in streams.items() if "cer" in v}
            name, cer = oracle_of(packed, prefer_original=True)
            recomputed = {"oracle_stream": name, "oracle_cer": round(cer, 4)}
            if name != rec.get("oracle_stream"):
                n_mismatch += 1
        cls = classify_streams(streams)
        skip = skip_sep_after_scores(streams)
        out.append(
            {
                **{k: rec[k] for k in rec if k not in ("src_wav", "dest_wav")},
                "streams": streams,
                "recomputed_oracle": recomputed,
                "dual_zero": cls["dual_zero"],
                "orig_unique_zero": cls["orig_unique_zero"],
                "skip_sep_after_scores": skip.skip,
                "skip_sep_reason": skip.reason,
                "feasibility": skip_sep_feasibility(cls),
                "selector": "oracle_cer_prefer_original",
                "mms_fa": False,
            }
        )
    write_jsonl(args.out, out)
    print(f"wrote {args.out} n={len(out)} oracle_mismatch={n_mismatch}")


if __name__ == "__main__":
    main()
