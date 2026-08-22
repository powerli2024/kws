#!/usr/bin/env python3
"""§0 data support: among original winners, how often is sep also CER=0?"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.dual_zero import classify_streams  # noqa: E402
from kws.iojson import index_by_uid, load_jsonl, stage_index_path, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--best-sep", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "dual_zero.json")
    return p.parse_args()


def load_stage_cache(root: Path):
    cache: dict[tuple[str, str], dict[str, dict]] = {}

    def get(split: str, stage: str) -> dict[str, dict]:
        key = (split, stage)
        if key not in cache:
            path = stage_index_path(root, split, stage)
            cache[key] = index_by_uid(load_jsonl(path)) if path.is_file() else {}
        return cache[key]

    return get


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    dual = sum(1 for r in rows if r["cls"]["dual_zero"])
    uniq = sum(1 for r in rows if r["cls"]["orig_unique_zero"])
    orig0 = sum(1 for r in rows if r["cls"]["orig_cer0"])
    miss = sum(1 for r in rows if r["cls"]["missing_original"] or r.get("missing_stage"))
    return {
        "n": n,
        "orig_cer0": orig0,
        "dual_zero": dual,
        "dual_zero_rate_among_n": (dual / n) if n else None,
        "dual_zero_rate_among_orig_cer0": (dual / orig0) if orig0 else None,
        "orig_unique_zero": uniq,
        "orig_unique_zero_rate_among_n": (uniq / n) if n else None,
        "orig_unique_zero_rate_among_orig_cer0": (uniq / orig0) if orig0 else None,
        "missing": miss,
        "evidence": "stage index streams CER; oracle_of prefers original on ties",
    }


def main() -> None:
    args = parse_args()
    pos_neg = args.pos_neg
    best_sep = args.best_sep or (pos_neg / "best_sep" / "index.jsonl")
    winners = load_jsonl(best_sep)
    getter = load_stage_cache(pos_neg)

    orig_winners = []
    all_aligned = []
    by_stage = Counter()
    by_stream = Counter()
    missing_stage = []

    for rec in winners:
        split = rec["split"]
        uid = rec["uid"]
        stage = rec["best_stage"]
        stream = rec.get("oracle_stream")
        by_stage[stage] += 1
        by_stream[str(stream)] += 1
        stage_map = getter(split, stage)
        srec = stage_map.get(uid)
        if srec is None:
            missing_stage.append({"uid": uid, "stage": stage})
            cls = classify_streams({})
            row = {
                "uid": uid,
                "split": split,
                "best_stage": stage,
                "oracle_stream": stream,
                "oracle_cer": rec.get("oracle_cer"),
                "missing_stage": True,
                "cls": cls,
            }
        else:
            cls = classify_streams(srec.get("streams") or {})
            row = {
                "uid": uid,
                "split": split,
                "best_stage": stage,
                "oracle_stream": stream,
                "oracle_cer": rec.get("oracle_cer"),
                "missing_stage": False,
                "cls": cls,
                "stage_oracle_stream": srec.get("oracle_stream"),
            }
        all_aligned.append(row)
        if stream == "original":
            orig_winners.append(row)

    # s1-only view (always original/spk1/spk2)
    s1_rows = []
    for rec in winners:
        split, uid = rec["split"], rec["uid"]
        srec = getter(split, "s1_onnx_full").get(uid)
        if not srec:
            continue
        cls = classify_streams(srec.get("streams") or {})
        s1_rows.append(
            {
                "uid": uid,
                "split": split,
                "s1_oracle": srec.get("oracle_stream"),
                "cls": cls,
            }
        )
    s1_orig_win = [r for r in s1_rows if r["s1_oracle"] == "original"]

    # What fraction of dual-zero original winners would flip if ties preferred sep?
    would_flip = 0
    sep_zero_n = Counter()
    for r in orig_winners:
        if not r["cls"]["dual_zero"]:
            continue
        would_flip += 1
        sep_zero_n[len(r["cls"].get("sep_cer0_names") or [])] += 1

    report = {
        "n_best_sep": len(winners),
        "oracle_stream_counts": dict(by_stream),
        "best_stage_counts": dict(by_stage),
        "n_original_winners": len(orig_winners),
        "missing_stage_index": len(missing_stage),
        "original_winners_at_winning_stage": summarize(orig_winners),
        "all_items_at_winning_stage": summarize(all_aligned),
        "s1_all": summarize(s1_rows),
        "s1_original_winners": summarize(s1_orig_win),
        "interpretation": {
            "dual_zero_means": (
                "original CER=0 and at least one sep track CER=0. "
                "argmin CER has no gradient; current oracle_of keeps original."
            ),
            "orig_unique_zero_means": (
                "original CER=0 and every sep track CER>0. "
                "Text-safe to skip using the sep wav (BSS hurt the transcript)."
            ),
            "skip_sep_branch": "feasibility from orig_unique_zero; dual_zero needs L2 not skip",
            "tie_break_original_on_dual_zero": would_flip,
            "dual_zero_n_sep_tracks_at_cer0": {str(k): v for k, v in sorted(sep_zero_n.items())},
            "kind": "evidence",
        },
    }
    write_json(args.out, report)
    ow = report["original_winners_at_winning_stage"]
    print(f"wrote {args.out}")
    print(
        f"original winners n={ow['n']}  orig_cer0={ow['orig_cer0']}  "
        f"dual_zero={ow['dual_zero']} ({ow['dual_zero_rate_among_n']})  "
        f"unique_zero={ow['orig_unique_zero']} ({ow['orig_unique_zero_rate_among_n']})"
    )
    s1o = report["s1_original_winners"]
    print(
        f"s1 original winners n={s1o['n']} dual_zero={s1o['dual_zero']} "
        f"unique_zero={s1o['orig_unique_zero']}"
    )


if __name__ == "__main__":
    main()
