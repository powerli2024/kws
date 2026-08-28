"""Inventory and compare every extract@sep stage/threshold arm fairly."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .iojson import load_jsonl

SKIP_STAGE_DIRS = {"meta", "reports", "best_sep", "packs"}
PARENT_BY_PREFIX = {
    "s5_": "s1_onnx_full",
    "s6_": "s1_onnx_full",
    "s7_": "s2_cv_full",
    "s8_": "s2_cv_full",
}


@dataclass(frozen=True)
class StageArm:
    split: str
    label: str
    index_path: Path
    rows: tuple[dict[str, Any], ...]
    errors: int


def discover_stage_arms(root: Path, split: str) -> list[StageArm]:
    split_root = Path(root) / split
    out: list[StageArm] = []
    if not split_root.is_dir():
        return out
    for stage in sorted(split_root.iterdir(), key=lambda p: p.name):
        if not stage.is_dir() or stage.name in SKIP_STAGE_DIRS or stage.name.startswith("."):
            continue
        candidates = [(stage.name, stage / "index.jsonl")]
        candidates.extend((f"{stage.name}/{thr.name}", thr / "index.jsonl") for thr in sorted(stage.glob("thr_*")))
        for label, path in candidates:
            if not path.is_file():
                continue
            raw = load_jsonl(path)
            valid = []
            errors = 0
            seen: set[str] = set()
            for row in raw:
                uid = str(row.get("uid") or "")
                if not uid:
                    raise ValueError(f"{path}: row missing uid")
                if uid in seen:
                    raise ValueError(f"{path}: duplicate uid={uid}")
                seen.add(uid)
                if row.get("error") or row.get("oracle_cer") is None:
                    errors += 1
                else:
                    valid.append(row)
            out.append(StageArm(split, label, path.resolve(), tuple(valid), errors))
    return out


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cohort_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    return _hash_json(sorted(str(row["uid"]) for row in rows))


def semantic_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    packed = []
    for row in sorted(rows, key=lambda r: str(r["uid"])):
        streams = row.get("streams") or {}
        packed_streams = {
            str(name): {
                key: rec.get(key)
                for key in ("hyp", "cer", "cer_char", "cer_py")
                if key in rec
            }
            for name, rec in sorted(streams.items())
            if isinstance(rec, Mapping)
        }
        packed.append({
            "uid": str(row["uid"]),
            "oracle_stream": row.get("oracle_stream"),
            "oracle_cer": row.get("oracle_cer"),
            "oracle_hyp": row.get("oracle_hyp"),
            "streams": packed_streams,
        })
    return _hash_json(packed)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 8) if values else None


def _metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    vals = [float(row["oracle_cer"]) for row in rows]
    return {
        "n": len(vals),
        "mean_cer": _mean(vals),
        "cer0": sum(v <= 1e-9 for v in vals),
        "cer0_rate": round(sum(v <= 1e-9 for v in vals) / len(vals), 8) if vals else None,
    }


def _paired(base: Mapping[str, Mapping[str, Any]], new: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    common = sorted(set(base) & set(new))
    deltas = [float(new[uid]["oracle_cer"]) - float(base[uid]["oracle_cer"]) for uid in common]
    improved = sum(d < -1e-9 for d in deltas)
    worsened = sum(d > 1e-9 for d in deltas)
    return {
        "n_common": len(common),
        "mean_delta": _mean(deltas),
        "n_improved": improved,
        "n_worsened": worsened,
        "n_same": len(deltas) - improved - worsened,
    }


def _parent_label(label: str, rows: Iterable[Mapping[str, Any]], labels: set[str]) -> str | None:
    values = {str(row.get("parent_stage") or "") for row in rows if row.get("parent_stage")}
    aliases = {"s1": "s1_onnx_full", "s2": "s2_cv_full", "s3": "s3_onnx_cascade", "s4": "s4_cv_cascade"}
    for value in sorted(values):
        resolved = aliases.get(value, value)
        if resolved in labels:
            return resolved
    stage = label.split("/", 1)[0]
    for prefix, parent in PARENT_BY_PREFIX.items():
        if stage.startswith(prefix) and parent in labels:
            return parent
    return None


def _threshold_values(rows: Iterable[Mapping[str, Any]]) -> list[float]:
    values = {float(row["thr"]) for row in rows if row.get("thr") is not None}
    return sorted(values)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wav_fingerprint(arm: StageArm) -> tuple[str | None, int, int]:
    arm_root = arm.index_path.parent
    wav_root = arm_root / "wav"
    packed: list[tuple[str, str, str]] = []
    missing = 0
    for row in arm.rows:
        uid = str(row["uid"])
        for stream in sorted((row.get("streams") or {}).keys()):
            tag = "peak" if stream == "original" else str(stream)
            path = wav_root / f"{uid}_{tag}.wav"
            if not path.is_file():
                missing += 1
                continue
            packed.append((uid, str(stream), _sha256(path)))
    return (_hash_json(packed) if packed else None, len(packed), missing)


def compare_split(arms: list[StageArm], *, hash_wav: bool = False) -> dict[str, Any]:
    labels = {arm.label for arm in arms}
    maps = {arm.label: {str(row["uid"]): row for row in arm.rows} for arm in arms}
    details: dict[str, dict[str, Any]] = {}
    for arm in arms:
        parent = _parent_label(arm.label, arm.rows, labels)
        row_map = maps[arm.label]
        detail: dict[str, Any] = {
            "label": arm.label,
            "index": str(arm.index_path),
            "stage_family": arm.label.split("/", 1)[0],
            "threshold_values": _threshold_values(arm.rows),
            "n_ok": len(arm.rows),
            "n_error": arm.errors,
            "metrics_subset": _metrics(arm.rows),
            "cohort_fingerprint": cohort_fingerprint(arm.rows),
            "semantic_fingerprint": semantic_fingerprint(arm.rows),
            "parent": parent,
        }
        if parent:
            pmap = maps[parent]
            detail["vs_parent_on_subset"] = _paired(pmap, row_map)
            missing_parent = sorted(set(row_map) - set(pmap))
            if missing_parent:
                detail["parent_coverage_error"] = missing_parent[:10]
            else:
                effective = [row_map.get(uid, prow) for uid, prow in pmap.items()]
                oracle_union = [
                    row_map[uid] if uid in row_map and float(row_map[uid]["oracle_cer"]) < float(prow["oracle_cer"]) else prow
                    for uid, prow in pmap.items()
                ]
                detail["metrics_full_parent_fallback"] = _metrics(effective)
                detail["metrics_full_parent_new_oracle_upper_bound"] = _metrics(oracle_union)
                detail["parent_n"] = len(pmap)
                detail["selected_fraction"] = round(len(row_map) / len(pmap), 8) if pmap else None
        if hash_wav:
            fp, n_files, n_missing = wav_fingerprint(arm)
            detail.update({"wav_fingerprint": fp, "n_wav_hashed": n_files, "n_wav_missing": n_missing})
        details[arm.label] = detail

    same_cohort: dict[tuple[str, str], list[str]] = {}
    same_threshold: dict[tuple[str, tuple[float, ...]], list[str]] = {}
    same_semantic: dict[str, list[str]] = {}
    same_wav: dict[str, list[str]] = {}
    for label, detail in details.items():
        same_cohort.setdefault((detail["stage_family"], detail["cohort_fingerprint"]), []).append(label)
        if detail["threshold_values"]:
            same_threshold.setdefault(
                (detail["stage_family"], tuple(detail["threshold_values"])), []
            ).append(label)
        same_semantic.setdefault(detail["semantic_fingerprint"], []).append(label)
        if detail.get("wav_fingerprint"):
            same_wav.setdefault(detail["wav_fingerprint"], []).append(label)
    cohort_groups = [sorted(group) for group in same_cohort.values() if len(group) > 1]
    threshold_groups = [sorted(group) for group in same_threshold.values() if len(group) > 1]
    semantic_groups = [sorted(group) for group in same_semantic.values() if len(group) > 1]
    wav_groups = [sorted(group) for group in same_wav.values() if len(group) > 1]

    pairwise = []
    ordered = sorted(arms, key=lambda a: a.label)
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            lm, rm = maps[left.label], maps[right.label]
            common = set(lm) & set(rm)
            union = set(lm) | set(rm)
            paired = _paired(lm, rm)
            pairwise.append({
                "left": left.label,
                "right": right.label,
                "same_cohort": set(lm) == set(rm),
                "cohort_jaccard": round(len(common) / len(union), 8) if union else None,
                **paired,
            })

    all_uids = sorted({uid for row_map in maps.values() for uid in row_map})
    oracle_vals = [min(float(row_map[uid]["oracle_cer"]) for row_map in maps.values() if uid in row_map) for uid in all_uids]
    return {
        "n_arms": len(arms),
        "n_union_uid": len(all_uids),
        "arms": details,
        "duplicate_same_threshold_value": threshold_groups,
        "duplicate_same_gate_cohort": cohort_groups,
        "duplicate_same_semantic_results": semantic_groups,
        "duplicate_same_wav": wav_groups if hash_wav else "not_computed",
        "pairwise": pairwise,
        "all_stage_oracle_upper_bound": {
            "n": len(oracle_vals), "mean_cer": _mean(oracle_vals),
            "cer0": sum(v <= 1e-9 for v in oracle_vals),
        },
    }


def build_report(root: Path, splits: Iterable[str], *, hash_wav: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "kws_stage_compare/v1",
        "pos_neg": str(Path(root).resolve()),
        "hash_wav": bool(hash_wav),
        "splits": {},
    }
    for split in splits:
        arms = discover_stage_arms(root, split)
        if not arms:
            raise FileNotFoundError(f"no stage indexes under {Path(root) / split}")
        result["splits"][split] = compare_split(arms, hash_wav=hash_wav)
    return result
