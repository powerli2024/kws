#!/usr/bin/env python3
"""Rank real s1--s8 audio candidates for each UID and fixed routes globally."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.iojson import write_json, write_jsonl  # noqa: E402
from kws.stage_compare import build_report, discover_stage_arms  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--splits", default="pos,neg")
    p.add_argument("--expected-uids", type=int, default=1838, help="0 disables assertion")
    p.add_argument("--top-k", type=int, default=20, help="ranked unique audio kept per UID; 0=all")
    p.add_argument("--no-hash-wav", action="store_true", help="faster, but copied audio cannot be exactly deduplicated")
    p.add_argument("--allow-missing-wav", action="store_true", help="smoke only")
    p.add_argument(
        "--score-conflict-policy", choices=("median", "min", "max", "fail"), default="median",
        help="combine repeated ASR scores for byte-identical WAVs; median is robust default",
    )
    p.add_argument("--allow-score-conflict", action="store_true", help="deprecated alias: use median instead of fail")
    p.add_argument("--out-jsonl", type=Path, default=ROOT / "reports" / "same_uid_audio_rank.jsonl")
    p.add_argument("--out-summary", type=Path, default=ROOT / "reports" / "same_uid_audio_rank_summary.json")
    p.add_argument("--out-conflicts", type=Path, default=ROOT / "reports" / "same_uid_audio_score_conflicts.jsonl")
    p.add_argument("--out-md", type=Path, default=ROOT / "reports" / "same_uid_audio_rank.md")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _wav_path(index_path: Path, uid: str, stream: str) -> Path:
    tag = "peak" if stream == "original" else stream
    return (index_path.parent / "wav" / f"{uid}_{tag}.wav").resolve()


def _route_leaderboard(report: dict, expected_uids: int) -> list[dict]:
    merged: dict[str, dict] = defaultdict(lambda: {"n": 0, "cer_sum": 0.0, "cer0": 0, "splits": {}})
    for split, block in report["splits"].items():
        for label, arm in block["arms"].items():
            metrics = arm.get("metrics_full_parent_fallback") or arm["metrics_subset"]
            n = int(metrics["n"])
            mean = metrics.get("mean_cer")
            if not n or mean is None:
                continue
            slot = merged[label]
            slot["n"] += n
            slot["cer_sum"] += float(mean) * n
            slot["cer0"] += int(metrics["cer0"])
            slot["splits"][split] = metrics
    rows = []
    for label, value in merged.items():
        n = value["n"]
        rows.append({
            "route": label,
            "n": n,
            "full_coverage": not expected_uids or n == expected_uids,
            "mean_cer": round(value["cer_sum"] / n, 8),
            "cer0": value["cer0"],
            "cer0_rate": round(value["cer0"] / n, 8),
            "splits": value["splits"],
        })
    rows.sort(key=lambda row: (not row["full_coverage"], row["mean_cer"], -row["cer0_rate"], row["route"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def render_md(summary: dict) -> str:
    lines = [
        "# Same-UID s1--s8 audio ranking", "",
        f"- UID coverage: `{summary['n_uid']}`",
        f"- raw candidate references: `{summary['n_candidate_refs']}`",
        f"- unique audio candidates: `{summary['n_unique_audio']}`",
        f"- exact WAV hashing: `{summary['hash_wav']}`",
        f"- missing WAV: `{summary['n_missing_wav']}`", "",
        f"- byte-identical audio with conflicting CER: `{summary['n_score_conflict_audio']}`",
        f"- conflict policy: `{summary['score_conflict_policy']}`", "",
        "CER ranking is text-preservation evidence only. Equal-CER audio remains tied;",
        "speaker purity must be resolved later by q_kw/FA and frozen Presence/CMD.", "",
        "## Fixed independent-route leaderboard", "",
        "| rank | route | coverage | mean CER | CER0 rate |", "|---:|---|---:|---:|---:|",
    ]
    for row in summary["route_leaderboard"]:
        lines.append(f"| {row['rank']} | {row['route']} | {row['n']} | {row['mean_cer']} | {row['cer0_rate']} |")
    lines += ["", "## Cross-stage best-audio tie leaderboard (offline only)", "", "| rank | stage | eligible UID | best ties | unique best | tie credit |", "|---:|---|---:|---:|---:|---:|"]
    for row in summary["best_audio_stage_leaderboard"]:
        lines.append(
            f"| {row['rank']} | {row['stage']} | {row['eligible_uid']} | {row['best_tie_uid']} | "
            f"{row['unique_best_uid']} | {row['tie_credit']} |"
        )
    lines += [
        "", "The second table is an oracle diagnostic across stages and must not be",
        "exported as a production route. Inspect per-UID candidates in",
        "`same_uid_audio_rank.jsonl`.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not splits or any(value not in {"pos", "neg"} for value in splits):
        raise SystemExit("[ERR] --splits must contain pos and/or neg")
    hash_wav = not args.no_hash_wav
    comparison = build_report(args.pos_neg, splits, hash_wav=False)
    by_uid: dict[str, list[dict]] = defaultdict(list)
    missing: list[str] = []
    stage_eligible: dict[str, set[str]] = defaultdict(set)
    n_refs = 0
    for split in splits:
        for arm in discover_stage_arms(args.pos_neg, split):
            for row in arm.rows:
                uid = str(row["uid"])
                streams = row.get("streams") or {}
                for stream, score in streams.items():
                    if not isinstance(score, dict) or score.get("cer") is None:
                        continue
                    path = _wav_path(arm.index_path, uid, str(stream))
                    if not path.is_file():
                        missing.append(str(path))
                        if not args.allow_missing_wav:
                            raise SystemExit(f"[ERR] missing WAV: {path}")
                        continue
                    n_refs += 1
                    stage_eligible[arm.label].add(uid)
                    by_uid[uid].append({
                        "stage": arm.label,
                        "stream": str(stream),
                        "cer": float(score["cer"]),
                        "hyp": score.get("hyp"),
                        "wav": str(path),
                        "audio_sha256": _sha256(path) if hash_wav else None,
                    })
    if args.expected_uids and len(by_uid) != args.expected_uids:
        raise SystemExit(f"[ERR] UID coverage={len(by_uid)} expected={args.expected_uids}")

    ranked_rows: list[dict] = []
    conflict_rows: list[dict] = []
    stage_wins: dict[str, dict[str, float]] = defaultdict(lambda: {"best_tie_uid": 0, "unique_best_uid": 0, "tie_credit": 0.0})
    n_unique_audio = 0
    for uid in sorted(by_uid):
        refs = by_uid[uid]
        groups: dict[str, list[dict]] = defaultdict(list)
        for ref in refs:
            key = ref["audio_sha256"] if hash_wav else f"{ref['stage']}|{ref['stream']}"
            groups[str(key)].append(ref)
        unique = []
        for key, aliases in groups.items():
            aliases.sort(key=lambda item: (item["stage"], item["stream"]))
            cer_values = sorted({round(float(item["cer"]), 9) for item in aliases})
            policy = "median" if args.allow_score_conflict and args.score_conflict_policy == "fail" else args.score_conflict_policy
            if len(cer_values) > 1 and policy == "fail":
                raise SystemExit(
                    f"[ERR] uid={uid}: byte-identical WAV has conflicting CER {cer_values}; "
                    "rerun with --score-conflict-policy median and inspect the conflict report"
                )
            observed = [float(item["cer"]) for item in aliases]
            aggregate_cer = (
                min(observed) if policy == "min" else
                max(observed) if policy == "max" else
                float(median(observed))
            )
            representative = min(
                aliases,
                key=lambda item: (abs(float(item["cer"]) - aggregate_cer), item["stage"], item["stream"]),
            )
            canonical = dict(representative)
            canonical["cer"] = round(aggregate_cer, 9)
            canonical["aliases"] = [
                {"stage": x["stage"], "stream": x["stream"], "wav": x["wav"], "cer": x["cer"], "hyp": x["hyp"]}
                for x in aliases
            ]
            canonical["duplicate_refs"] = len(aliases)
            canonical["cer_values"] = cer_values
            canonical["score_conflict"] = len(cer_values) > 1
            canonical["score_aggregate"] = policy
            if len(cer_values) > 1:
                conflict_rows.append({
                    "uid": uid,
                    "audio_sha256": None if key.startswith(uid + "|") else key,
                    "cer_values": cer_values,
                    "resolved_cer": canonical["cer"],
                    "policy": policy,
                    "references": canonical["aliases"],
                })
            unique.append(canonical)
        unique.sort(key=lambda item: (item["cer"], item["stage"], item["stream"]))
        n_unique_audio += len(unique)
        dense_rank = 0
        last_cer = None
        for item in unique:
            if last_cer is None or abs(float(item["cer"]) - last_cer) > 1e-9:
                dense_rank += 1
                last_cer = float(item["cer"])
            item["rank"] = dense_rank
        best = [item for item in unique if item["rank"] == 1]
        # Attribute a byte-identical copied waveform only to its canonical
        # (earliest deterministic) stage, so copied parent streams do not gain
        # repeated win credit in cascade/thr directories.
        best_stages = sorted({item["stage"] for item in best})
        for stage in best_stages:
            stage_wins[stage]["best_tie_uid"] += 1
            stage_wins[stage]["tie_credit"] += 1.0 / len(best_stages)
        if len(best_stages) == 1:
            stage_wins[best_stages[0]]["unique_best_uid"] += 1
        split = uid.split("_", 1)[0]
        ranked_rows.append({
            "uid": uid,
            "split": split,
            "best_cer": best[0]["cer"] if best else None,
            "n_candidate_refs": len(refs),
            "n_unique_audio": len(unique),
            "n_best_audio_ties": len(best),
            "n_score_conflict_audio": sum(bool(item["score_conflict"]) for item in unique),
            "best_stages": best_stages,
            "ranking": unique if args.top_k == 0 else unique[: args.top_k],
            "ranking_truncated": bool(args.top_k and len(unique) > args.top_k),
        })

    win_rows = []
    for stage in sorted(stage_eligible):
        wins = stage_wins[stage]
        win_rows.append({
            "stage": stage,
            "eligible_uid": len(stage_eligible[stage]),
            "best_tie_uid": int(wins["best_tie_uid"]),
            "unique_best_uid": int(wins["unique_best_uid"]),
            "tie_credit": round(wins["tie_credit"], 6),
            "best_tie_rate_on_eligible": round(wins["best_tie_uid"] / len(stage_eligible[stage]), 8),
        })
    win_rows.sort(key=lambda row: (-row["tie_credit"], -row["unique_best_uid"], row["stage"]))
    for i, row in enumerate(win_rows, start=1):
        row["rank"] = i

    summary = {
        "schema": "same_uid_audio_rank/v1",
        "pos_neg": str(args.pos_neg.resolve()),
        "hash_wav": hash_wav,
        "n_uid": len(by_uid),
        "n_candidate_refs": n_refs,
        "n_unique_audio": n_unique_audio,
        "n_missing_wav": len(missing),
        "missing_wav_head": missing[:20],
        "top_k_per_uid": args.top_k,
        "score_conflict_policy": args.score_conflict_policy,
        "n_score_conflict_audio": len(conflict_rows),
        "n_score_conflict_uid": len({row["uid"] for row in conflict_rows}),
        "score_conflict_uid_head": sorted({row["uid"] for row in conflict_rows})[:20],
        "route_leaderboard": _route_leaderboard(comparison, args.expected_uids),
        "best_audio_stage_leaderboard": win_rows,
        "warning": "Cross-stage per-UID best is an offline CER oracle, not a deployable route.",
    }
    write_jsonl(args.out_jsonl, ranked_rows)
    write_jsonl(args.out_conflicts, conflict_rows)
    write_json(args.out_summary, summary)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_md(summary), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"route_leaderboard", "best_audio_stage_leaderboard"}}, ensure_ascii=False, indent=2))
    print(f"[OK] {args.out_jsonl}")
    print(f"[OK] {args.out_summary}")
    print(f"[OK] {args.out_conflicts} n={len(conflict_rows)}")
    print(f"[OK] {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
