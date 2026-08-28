#!/usr/bin/env python3
"""Export fixed, independent stage/thr routes; never oracle-mix stages per UID."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.iojson import write_json, write_jsonl  # noqa: E402
from kws.stage_compare import PARENT_BY_PREFIX, StageArm, discover_stage_arms  # noqa: E402

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--route", action="append", required=True, help="repeatable exact label, e.g. s5_onnx_then_cv_gate/thr_a")
    p.add_argument("--splits", default="pos,neg")
    p.add_argument("--out-root", type=Path, default=Path(r"d:\media\pos_neg\stage_routes"))
    p.add_argument("--expected-uids", type=int, default=1838, help="0 disables combined assertion")
    return p.parse_args()


def _parent_label(arm: StageArm, labels: set[str]) -> str | None:
    aliases = {"s1": "s1_onnx_full", "s2": "s2_cv_full", "s3": "s3_onnx_cascade", "s4": "s4_cv_cascade"}
    values = {str(row.get("parent_stage") or "") for row in arm.rows if row.get("parent_stage")}
    for value in sorted(values):
        parent = aliases.get(value, value)
        if parent in labels:
            return parent
    stage = arm.label.split("/", 1)[0]
    for prefix, parent in PARENT_BY_PREFIX.items():
        if stage.startswith(prefix) and parent in labels:
            return parent
    return None


def _wav(arm: StageArm, row: dict) -> Path:
    stream = str(row.get("oracle_stream") or "")
    tag = "peak" if stream == "original" else stream
    path = arm.index_path.parent / "wav" / f"{row['uid']}_{tag}.wav"
    if not path.is_file():
        raise FileNotFoundError(f"uid={row['uid']} route={arm.label}: missing {path}")
    return path.resolve()


def _slug(route: str) -> str:
    return route.replace("/", "__").replace("\\", "__")


def main() -> int:
    args = parse_args()
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not splits or any(value not in {"pos", "neg"} for value in splits):
        raise SystemExit("[ERR] --splits must contain pos and/or neg")
    all_arms = {split: {arm.label: arm for arm in discover_stage_arms(args.pos_neg, split)} for split in splits}
    summaries: dict[str, dict] = {}
    for route in args.route:
        route_rows: list[dict] = []
        route_root = args.out_root / _slug(route)
        counts = {"route": 0, "parent_fallback": 0}
        for split in splits:
            arms = all_arms[split]
            if route not in arms:
                raise SystemExit(f"[ERR] split={split}: unknown route {route!r}; known={sorted(arms)}")
            arm = arms[route]
            parent_name = _parent_label(arm, set(arms))
            selected = {str(row["uid"]): (arm, row, False) for row in arm.rows}
            if parent_name:
                parent = arms[parent_name]
                parent_map = {str(row["uid"]): row for row in parent.rows}
                extra = sorted(set(selected) - set(parent_map))
                if extra:
                    raise SystemExit(f"[ERR] route={route}: UIDs outside parent {parent_name}: {extra[:10]}")
                for uid, row in parent_map.items():
                    selected.setdefault(uid, (parent, row, True))
            for uid in sorted(selected):
                source_arm, row, fallback = selected[uid]
                src = _wav(source_arm, row)
                dest_rel = f"{split}/{uid}.wav"
                dest = route_root / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                wake = str(row.get("wake_text") or "")
                route_rows.append({
                    "uid": uid,
                    "split": split,
                    "id": row.get("id"),
                    "kws_rel": row.get("kws_rel"),
                    "wake_text": wake,
                    "lang": "zh" if CJK_RE.search(wake) else "en",
                    "metric": row.get("metric"),
                    "route": route,
                    "best_stage": source_arm.label,
                    "parent_fallback": fallback,
                    "oracle_stream": row.get("oracle_stream"),
                    "oracle_cer": row.get("oracle_cer"),
                    "oracle_hyp": row.get("oracle_hyp"),
                    "streams": row.get("streams") or {},
                    "src_wav": str(src),
                    "dest_rel": dest_rel,
                    "selector": "independent_stage_route_with_parent_fallback",
                    "ok": dest.is_file(),
                    "bytes": dest.stat().st_size,
                })
                counts["parent_fallback" if fallback else "route"] += 1
        if args.expected_uids and len(route_rows) != args.expected_uids:
            raise SystemExit(f"[ERR] route={route}: coverage={len(route_rows)} expected={args.expected_uids}")
        write_jsonl(route_root / "index.jsonl", route_rows)
        summary = {
            "schema": "independent_stage_route/v1",
            "route": route,
            "path": str(route_root.resolve()),
            "n": len(route_rows),
            "n_from_route": counts["route"],
            "n_parent_fallback": counts["parent_fallback"],
            "cross_stage_oracle": False,
            "note": "One fixed stage/thr policy; gated UIDs absent from the cohort use its declared parent.",
        }
        write_json(route_root / "summary.json", summary)
        summaries[route] = summary
        print(f"[OK] {route} n={len(route_rows)} fallback={counts['parent_fallback']} -> {route_root}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
