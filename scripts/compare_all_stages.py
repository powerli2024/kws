#!/usr/bin/env python3
"""Systematic s1--s8/thr comparison with duplicate detection and fair fallback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.iojson import write_json  # noqa: E402
from kws.stage_compare import build_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--splits", default="pos,neg")
    p.add_argument("--hash-wav", action="store_true", help="slow exact-content duplicate check")
    p.add_argument("--expected-uids", type=int, default=1838, help="0 disables global coverage assertion")
    p.add_argument("--out-json", type=Path, default=ROOT / "reports" / "all_stage_comparison.json")
    p.add_argument("--out-md", type=Path, default=ROOT / "reports" / "all_stage_comparison.md")
    return p.parse_args()


def render_md(report: dict) -> str:
    lines = [
        "# extract-sep all-stage comparison", "",
        f"- root: `{report['pos_neg']}`",
        f"- exact WAV hashing: `{report['hash_wav']}`", "",
        "`thr_*` means are never compared as if they were full datasets. Gated arms are",
        "reported against their parent on the same UID subset and with parent fallback", 
        "over the full parent cohort.", "",
    ]
    for split, block in report["splits"].items():
        lines += [f"## {split}", "", f"- arms: {block['n_arms']}", f"- union UIDs: {block['n_union_uid']}"]
        upper = block["all_stage_oracle_upper_bound"]
        lines.append(f"- all-stage oracle upper bound: n={upper['n']} mean={upper['mean_cer']} CER0={upper['cer0']}")
        lines += ["", "### Duplicate threshold arms", ""]
        for group in block["duplicate_same_threshold_value"]:
            lines.append(f"- same numerical threshold: `{', '.join(group)}`")
        cohorts = block["duplicate_same_gate_cohort"]
        if cohorts:
            lines.extend(f"- same selected UID cohort: `{', '.join(group)}`" for group in cohorts)
        else:
            lines.append("- none")
        for group in block["duplicate_same_semantic_results"]:
            lines.append(f"- same index scores/hypotheses: `{', '.join(group)}`")
        if isinstance(block["duplicate_same_wav"], list):
            for group in block["duplicate_same_wav"]:
                lines.append(f"- exact same WAV content: `{', '.join(group)}`")
        lines += ["", "### Arms", "", "| arm | ok/fail | threshold | subset mean | parent | paired delta | full fallback mean |", "|---|---:|---:|---:|---|---:|---:|"]
        for label, arm in sorted(block["arms"].items()):
            paired = arm.get("vs_parent_on_subset") or {}
            full = arm.get("metrics_full_parent_fallback") or {}
            lines.append(
                f"| {label} | {arm['n_ok']}/{arm['n_error']} | {arm['threshold_values']} | "
                f"{arm['metrics_subset']['mean_cer']} | {arm.get('parent') or '-'} | "
                f"{paired.get('mean_delta', '-')} | {full.get('mean_cer', '-')} |"
            )
        lines.append("")
    lines += [
        "## Interpretation", "",
        "- `same selected UID cohort` means threshold configurations are redundant even if names differ.",
        "- `same index scores/hypotheses` means the scored experiment result is duplicated.",
        "- Only `--hash-wav` can prove byte-identical audio.",
        "- `all-stage oracle` is an offline ceiling, not a deployable selector.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not splits or any(value not in {"pos", "neg"} for value in splits):
        raise SystemExit("[ERR] --splits must contain pos and/or neg")
    report = build_report(args.pos_neg, splits, hash_wav=args.hash_wav)
    union = sum(int(block["n_union_uid"]) for block in report["splits"].values())
    if args.expected_uids and union != args.expected_uids:
        raise SystemExit(f"[ERR] union UID coverage={union}, expected={args.expected_uids}")
    write_json(args.out_json, report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_md(report), encoding="utf-8")
    compact = {
        split: {
            "arms": block["n_arms"], "uids": block["n_union_uid"],
            "same_threshold_groups": len(block["duplicate_same_threshold_value"]),
            "same_cohort_groups": len(block["duplicate_same_gate_cohort"]),
            "same_semantic_groups": len(block["duplicate_same_semantic_results"]),
            "same_wav_groups": len(block["duplicate_same_wav"]) if isinstance(block["duplicate_same_wav"], list) else None,
            "oracle_upper": block["all_stage_oracle_upper_bound"],
        }
        for split, block in report["splits"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"[OK] {args.out_json}")
    print(f"[OK] {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
