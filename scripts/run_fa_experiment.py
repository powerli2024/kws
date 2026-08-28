#!/usr/bin/env python3
"""Run isolated FA routing/cropping arms without changing frozen T0--T4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.fa_experiment import fa_winner, load_alignment_sidecar, route_by_agreement, safe_crop_plan  # noqa: E402
from kws.iojson import limit_rows_balanced, load_jsonl, write_json, write_jsonl  # noqa: E402
from kws.select_l2 import l1_eligible  # noqa: E402
from kws.sidecar import SidecarError, load_qkw_sidecar_with_kind  # noqa: E402
from kws.t0_t4 import t0_stream  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--enriched", type=Path, default=ROOT / "reports" / "best_sep_enriched.jsonl")
    p.add_argument("--qkw-jsonl", type=Path, required=True, help="q_kw or raw NLL sidecar")
    p.add_argument("--route-fa-jsonl", type=Path, required=True, help="MMS-FA alignment evidence")
    p.add_argument("--crop-fa-jsonl", type=Path, default=None, help="Qwen3-FA/MMS boundary sidecar; defaults to route FA")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "fa_experiment_picks.jsonl")
    p.add_argument("--summary", type=Path, default=ROOT / "reports" / "fa_experiment_summary.json")
    p.add_argument("--min-coverage", type=float, default=1.0)
    p.add_argument("--max-star-fraction", type=float, default=0.25, help="provisional; calibrate on listen100")
    p.add_argument("--crop-margin-sec", type=float, default=0.24)
    p.add_argument("--crop-min-output-sec", type=float, default=1.50)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true", help="smoke only: omit UIDs absent from a sidecar")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.enriched, args.qkw_jsonl, args.route_fa_jsonl):
        if not path.is_file():
            raise SidecarError(f"missing input: {path}")
    rows = load_jsonl(args.enriched)
    if args.limit:
        rows = limit_rows_balanced(rows, args.limit)
    qkw, qkind = load_qkw_sidecar_with_kind(args.qkw_jsonl)
    route_fa, route_models = load_alignment_sidecar(args.route_fa_jsonl)
    crop_path = args.crop_fa_jsonl or args.route_fa_jsonl
    crop_fa, crop_models = load_alignment_sidecar(crop_path)

    out: list[dict] = []
    counts = {"n_input": len(rows), "n_scored": 0, "n_agree": 0, "n_changed_from_t0": 0, "n_crop": 0, "n_fallback": 0}
    for rec in rows:
        uid = str(rec["uid"])
        missing = [name for name, table in (("qkw", qkw), ("route_fa", route_fa), ("crop_fa", crop_fa)) if uid not in table]
        if missing:
            if args.allow_partial:
                continue
            raise SidecarError(f"uid={uid} missing sidecars {missing}")
        streams = rec.get("streams") or {}
        t0 = t0_stream(rec)
        eligible, _ = l1_eligible(streams)
        f_only = fa_winner(
            eligible,
            route_fa[uid],
            min_coverage=args.min_coverage,
            max_star_fraction=args.max_star_fraction,
        )
        route = route_by_agreement(
            streams,
            t0=t0,
            qkw=qkw[uid],
            evidence=route_fa[uid],
            min_coverage=args.min_coverage,
            max_star_fraction=args.max_star_fraction,
        )
        chosen_ev = crop_fa[uid].get(route.chosen)
        if chosen_ev is None:
            raise SidecarError(f"uid={uid}: crop sidecar missing chosen stream {route.chosen!r}")
        crop = safe_crop_plan(
            chosen_ev,
            margin_sec=args.crop_margin_sec,
            min_output_sec=args.crop_min_output_sec,
            min_coverage=args.min_coverage,
            max_star_fraction=args.max_star_fraction,
        )
        row = {
            "uid": uid,
            "split": rec.get("split"),
            "id": rec.get("id"),
            "wake_text": rec.get("wake_text"),
            "lang": rec.get("lang"),
            "best_stage": rec.get("best_stage"),
            "kws_rel": rec.get("kws_rel"),
            "t0": t0,
            "eligible": list(route.eligible),
            "qkw_kind": qkind[uid],
            "route_model": route_models[uid],
            "crop_model": crop_models[uid],
            "arms": {
                "F0_t0_full": {"chosen": t0, "crop": None},
                "F1_qkw_full": {"chosen": route.qkw_winner, "crop": None},
                "F2_fa_full": {"chosen": f_only or t0, "crop": None, "valid_fa": f_only is not None},
                "F3_agree_full": {"chosen": route.chosen, "reason": route.reason, "crop": None},
                "F4_agree_safe_crop": {
                    "chosen": route.chosen,
                    "reason": route.reason,
                    "crop": {"start_sec": crop.start_sec, "end_sec": crop.end_sec} if crop.apply else None,
                    "crop_reason": crop.reason,
                },
            },
            "diagnostics": {
                "qkw_winner": route.qkw_winner,
                "fa_winner": route.fa_winner,
                "agreed": route.agreed,
                "crop_applied": crop.apply,
            },
        }
        out.append(row)
        counts["n_scored"] += 1
        counts["n_agree"] += int(route.agreed)
        counts["n_changed_from_t0"] += int(route.chosen != t0)
        counts["n_crop"] += int(crop.apply)
        counts["n_fallback"] += int(not route.agreed)

    if not args.allow_partial and len(out) != len(rows):
        raise SidecarError(f"coverage mismatch: scored={len(out)} input={len(rows)}")
    write_jsonl(args.out, out)
    summary = {
        "schema": "fa_experiment/v1",
        **counts,
        "coverage": (len(out) / len(rows)) if rows else 0.0,
        "config": {
            "min_coverage": args.min_coverage,
            "max_star_fraction": args.max_star_fraction,
            "crop_margin_sec": args.crop_margin_sec,
            "crop_min_output_sec": args.crop_min_output_sec,
            "allow_partial": args.allow_partial,
        },
        "note": "Experimental only. Frozen T0--T4 and Presence thresholds are unchanged.",
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] {args.out} n={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
