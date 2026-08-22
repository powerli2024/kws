#!/usr/bin/env python3
"""T0–T4 enroll selection arms. One factor at a time. No MMS-FA.

T0: CER oracle (enriched best_sep)
T1: T0 + conditional SE (requires SE backend)
T2: L2 cos-to-raw under CER slack, no SE
T3: T2 + conditional SE
T4: SE on everyone (ablation; expected to lose)

Without embeddings / SE backends, T2 falls back to recording eligible dual-zero
items and T1/T3/T4 are marked skipped. Presence veto is not computed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.iojson import load_jsonl, write_json  # noqa: E402
from kws.need_se import need_se, se_safety_ok  # noqa: E402
from kws.select_l2 import CER_SLACK_DEFAULT, select_l1_l2  # noqa: E402


ARMS = ("T0", "T1", "T2", "T3", "T4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--enriched",
        type=Path,
        default=ROOT / "reports" / "best_sep_enriched.jsonl",
    )
    p.add_argument("--cos-jsonl", type=Path, default=None, help="uid -> {stream: cos_to_raw}")
    p.add_argument("--pmusic-jsonl", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "t0_t4.json")
    p.add_argument("--arm", choices=ARMS, default=None)
    return p.parse_args()


def load_sidecar(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    out: dict[str, dict] = {}
    for row in load_jsonl(path):
        out[str(row["uid"])] = row.get("scores") or row.get("cos") or row
    return out


def apply_se_placeholder(
    chosen: str,
    *,
    arm: str,
    winner_is_original: bool,
    p_music: float | None,
    snr: float | None,
) -> dict:
    """SE backends are optional. Record the trigger; do not pretend denoise ran."""
    if arm == "T0":
        return {"se_applied": False, "reason": "t0_no_se"}
    if arm == "T2":
        return {"se_applied": False, "reason": "t2_no_se"}
    if arm == "T4":
        return {
            "se_applied": False,
            "would_apply": True,
            "reason": "t4_global_se_backend_missing",
            "safety": "must_check_cos_se_pre_and_presence",
        }
    trig = need_se(
        winner_is_original=winner_is_original,
        p_music=p_music,
        snr_med_db=snr,
    )
    return {
        "se_applied": False,
        "would_apply": trig.need,
        "need_se_reason": trig.reason,
        "reason": "conditional_se_backend_missing" if trig.need else "need_se_false",
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.enriched)
    cos_map = load_sidecar(args.cos_jsonl)
    pm_map = load_sidecar(args.pmusic_jsonl)
    arms = (args.arm,) if args.arm else ARMS
    summary: dict[str, dict] = {}
    for arm in arms:
        n_dual = 0
        n_l2_diff = 0
        n_cat = 0
        n_need_se = 0
        chosen_counts: dict[str, int] = {}
        examples: list[dict] = []
        for rec in rows:
            uid = rec["uid"]
            streams = rec.get("streams") or {}
            if not streams:
                continue
            t0 = rec.get("oracle_stream") or rec.get("recomputed_oracle", {}).get("oracle_stream")
            pm = None
            if uid in pm_map:
                v = pm_map[uid]
                pm = float(v.get("p_music", v.get(t0, 0.0))) if isinstance(v, dict) else float(v)
            if arm in ("T0", "T1"):
                chosen = t0
                dual = bool(rec.get("dual_zero"))
                reverted = False
                reason = "t0_cer_oracle"
            else:
                sel = select_l1_l2(
                    streams,
                    cos_to_raw=cos_map.get(uid),
                    p_music=pm_map.get(uid) if isinstance(pm_map.get(uid), dict) else None,
                )
                chosen = sel.chosen
                dual = sel.dual_zero
                reverted = sel.reverted_catastrophe
                reason = sel.reason
                if sel.chosen != t0:
                    n_l2_diff += 1
                if reverted:
                    n_cat += 1
            if dual:
                n_dual += 1
            se = apply_se_placeholder(
                chosen,
                arm=arm,
                winner_is_original=(chosen == "original"),
                p_music=pm,
                snr=None,
            )
            if se.get("would_apply"):
                n_need_se += 1
            chosen_counts[str(chosen)] = chosen_counts.get(str(chosen), 0) + 1
            if len(examples) < 8 and (dual or chosen != t0):
                examples.append(
                    {
                        "uid": uid,
                        "t0": t0,
                        "chosen": chosen,
                        "reason": reason,
                        "dual_zero": dual,
                        "se": se,
                    }
                )
        summary[arm] = {
            "n": len(rows),
            "chosen_counts": chosen_counts,
            "n_dual_zero": n_dual,
            "n_l2_diff_from_t0": n_l2_diff,
            "n_catastrophe_revert": n_cat,
            "n_would_se": n_need_se,
            "cos_sidecar_loaded": bool(cos_map),
            "examples": examples,
            "answers": {
                "T0": "current CER oracle enroll",
                "T1": "does conditional SE on CER winners help Presence?",
                "T2": "does cos-to-raw under CER slack pick a different track on dual-zero?",
                "T3": "T2 plus conditional SE",
                "T4": "global SE should lose; if it wins, need_se is miscalibrated",
            }[arm],
        }
    write_json(args.out, summary)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in summary.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
