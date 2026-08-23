#!/usr/bin/env python3
"""T0–T4 enroll selection arms. One factor at a time. No MMS-FA.

T4 is CER-oracle + always-SE. It must not call L2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.arms import ALL_ARMS, CER_ORACLE_ARMS, L2_ARMS, se_mode  # noqa: E402
from kws.iojson import load_jsonl, write_json, write_jsonl, limit_rows_balanced  # noqa: E402
from kws.sidecar import (  # noqa: E402
    SidecarError,
    clip_p_music,
    load_cos_sidecar,
    load_pmusic_sidecar,
    load_qkw_sidecar,
)
from kws.t0_t4 import pick_track, t0_stream, apply_se_placeholder  # noqa: E402

ARMS = tuple(sorted(ALL_ARMS))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--enriched",
        type=Path,
        default=ROOT / "reports" / "best_sep_enriched.jsonl",
    )
    p.add_argument("--cos-jsonl", type=Path, default=None, help="catastrophe gate only: uid + cos_to_raw|scores|cos")
    p.add_argument("--qkw-jsonl", type=Path, default=None, help="T2 rank: uid + q_kw or nll dict")
    p.add_argument("--pmusic-jsonl", type=Path, default=None, help="diagnostic only unless λ≠0")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "t0_t4.json")
    p.add_argument(
        "--picks",
        type=Path,
        default=ROOT / "reports" / "t0_t4_picks.jsonl",
        help="per-uid chosen stream for every arm (needed to materialize best_sep groups)",
    )
    p.add_argument("--arm", choices=ARMS, default=None)
    p.add_argument("--limit", type=int, default=0, help="score only the first N enriched rows")
    p.add_argument(
        "--strict-cos",
        action="store_true",
        help="legacy alias of --strict-text",
    )
    p.add_argument(
        "--strict-text",
        action="store_true",
        help="T2/T3 fail if --qkw-jsonl is missing (default: degrade to T0 and flag)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.enriched)
    if args.limit:
        rows = limit_rows_balanced(rows, args.limit)
    if args.cos_jsonl is not None:
        if not args.cos_jsonl.is_file():
            raise SidecarError(f"cos sidecar not found: {args.cos_jsonl}")
        cos_map = load_cos_sidecar(args.cos_jsonl)
        if not cos_map:
            raise SidecarError(f"cos sidecar is empty: {args.cos_jsonl}")
    else:
        cos_map = {}
    if args.pmusic_jsonl is not None:
        if not args.pmusic_jsonl.is_file():
            raise SidecarError(f"p_music sidecar not found: {args.pmusic_jsonl}")
        pm_map = load_pmusic_sidecar(args.pmusic_jsonl)
        if not pm_map:
            raise SidecarError(f"p_music sidecar is empty: {args.pmusic_jsonl}")
    else:
        pm_map = {}
    if args.qkw_jsonl is not None:
        if not args.qkw_jsonl.is_file():
            raise SidecarError(f"q_kw sidecar not found: {args.qkw_jsonl}")
        qkw_map = load_qkw_sidecar(args.qkw_jsonl)
        if not qkw_map:
            raise SidecarError(f"q_kw sidecar is empty: {args.qkw_jsonl}")
    else:
        qkw_map = {}

    arms = (args.arm,) if args.arm else ARMS
    if (args.strict_text or args.strict_cos) and any(a in L2_ARMS for a in arms) and not qkw_map:
        raise SidecarError("T2/T3 require --qkw-jsonl under --strict-text")

    picks: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for arm in arms:
        n_dual = 0
        n_l2_diff = 0
        n_cat = 0
        n_need_se = 0
        n_degraded = 0
        chosen_counts: dict[str, int] = {}
        examples: list[dict] = []
        for rec in rows:
            streams = rec.get("streams") or {}
            if not streams:
                continue
            uid = str(rec["uid"])
            picked = pick_track(arm, rec, cos_map=cos_map, pm_map=pm_map, qkw_map=qkw_map)
            t0 = t0_stream(rec)
            chosen = picked["chosen"]
            pm_clip = None
            if uid in pm_map and chosen not in ("reject", ""):
                pm_clip = clip_p_music(pm_map.get(uid), chosen)
            se = (
                {"se_applied": False, "would_apply": False, "reason": "rejected_enroll"}
                if chosen == "reject"
                else apply_se_placeholder(chosen, arm=arm, p_music=pm_clip, snr=None)
            )
            slot = picks.setdefault(
                uid,
                {
                    "uid": uid,
                    "split": rec.get("split"),
                    "id": rec.get("id"),
                    "kws_rel": rec.get("kws_rel"),
                    "best_stage": rec.get("best_stage"),
                    "oracle_stream": t0,
                    "oracle_cer": rec.get("oracle_cer"),
                    "dual_zero": bool(rec.get("dual_zero")),
                    "orig_unique_zero": bool(rec.get("orig_unique_zero")),
                    "skip_sep_after_scores": bool(rec.get("skip_sep_after_scores")),
                    "wake_text": rec.get("wake_text"),
                    "lang": rec.get("lang"),
                    "dest_rel": rec.get("dest_rel"),
                    "src_wav_rel": rec.get("src_wav_rel"),
                    "stream_names": list(streams),
                    "arms": {},
                },
            )
            slot["arms"][arm] = {
                "chosen": chosen,
                "reason": picked["reason"],
                "dual_zero": picked["dual_zero"],
                "reverted_catastrophe": picked["reverted_catastrophe"],
                "l2_degraded": picked["l2_degraded"],
                "se": se,
            }
            if picked["dual_zero"]:
                n_dual += 1
            if picked["l2_degraded"]:
                n_degraded += 1
            if chosen != t0 and arm in L2_ARMS:
                n_l2_diff += 1
            if picked["reverted_catastrophe"]:
                n_cat += 1
            if se.get("would_apply"):
                n_need_se += 1
            chosen_counts[str(chosen)] = chosen_counts.get(str(chosen), 0) + 1
            if len(examples) < 8 and (picked["dual_zero"] or chosen != t0 or picked["l2_degraded"]):
                examples.append(
                    {
                        "uid": uid,
                        "t0": t0,
                        "chosen": chosen,
                        "reason": picked["reason"],
                        "dual_zero": picked["dual_zero"],
                        "se": se,
                    }
                )
        summary[arm] = {
            "n": len(rows),
            "select_mode": "cer_oracle" if arm in CER_ORACLE_ARMS else "l2",
            "se_mode": se_mode(arm),
            "chosen_counts": chosen_counts,
            "n_dual_zero": n_dual,
            "n_l2_diff_from_t0": n_l2_diff,
            "n_catastrophe_revert": n_cat,
            "n_would_se": n_need_se,
            "n_l2_degraded_no_text": n_degraded,
            "n_l2_degraded_no_cos": n_degraded,
            "cos_sidecar_loaded": bool(cos_map),
            "qkw_sidecar_loaded": bool(qkw_map),
            "examples": examples,
            "answers": {
                "T0": "current CER oracle enroll",
                "T1": "does conditional SE on CER winners help Presence?",
                "T2": "does q_kw + catastrophe gate pick a different dual-zero track?",
                "T3": "T2 plus conditional SE",
                "T4": "global SE on CER-oracle enroll should lose; if it wins, need_se is miscalibrated",
            }[arm],
        }
        if arm in L2_ARMS and n_degraded == summary[arm]["n"] and not qkw_map:
            summary[arm]["stop_loss"] = "L2 had no q_kw sidecar; do not interpret as T2 failure; skip T3"
    write_jsonl(args.picks, [picks[k] for k in picks])
    write_json(args.out, summary)
    print(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in summary.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] picks {args.picks} n={len(picks)}")


if __name__ == "__main__":
    main()
