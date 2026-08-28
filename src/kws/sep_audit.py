"""Strict extract-sep input audit before KWS comparison/ranking."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .handoff import LEGACY_SCHEMA, SCHEMA, find_handoff, load_handoff
from .iojson import load_jsonl

CJK = re.compile(r"[\u4e00-\u9fff]")
STAGES = {
    "s1": "s1_onnx_full",
    "s2": "s2_cv_full",
    "s3": "s3_onnx_cascade",
    "s4": "s4_cv_cascade",
    "s5": "s5_onnx_then_cv_gate",
    "s6": "s6_onnx_then_onnx_gate",
    "s7": "s7_cv_then_onnx_gate",
    "s8": "s8_cv_then_cv_gate",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"JSON object required: {path}")
    return obj


def _duration(path: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def _append(failures: list[dict[str, Any]], **item: Any) -> None:
    if len(failures) < 200:
        failures.append(item)


def _check_rows(
    path: Path,
    *,
    split: str,
    check_duration: bool,
    failures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.is_file():
        _append(failures, path=str(path), error="missing_index")
        return [], set()
    rows = load_jsonl(path)
    seen: set[str] = set()
    valid: list[dict[str, Any]] = []
    for row in rows:
        uid = str(row.get("uid") or "")
        if not uid or not uid.startswith(split + "_"):
            _append(failures, path=str(path), uid=uid, error="bad_uid_or_split")
            continue
        if uid in seen:
            _append(failures, path=str(path), uid=uid, error="duplicate_uid")
            continue
        seen.add(uid)
        if row.get("error") or row.get("oracle_cer") is None:
            _append(failures, path=str(path), uid=uid, error="failed_or_unscored_row")
            continue
        expected_metric = "pinyin" if CJK.search(str(row.get("wake_text") or "")) else "char"
        if row.get("metric") != expected_metric:
            _append(
                failures, path=str(path), uid=uid, error="metric_mismatch",
                observed=row.get("metric"), expected=expected_metric,
            )
        streams = row.get("streams") or {}
        if not isinstance(streams, dict) or not streams:
            _append(failures, path=str(path), uid=uid, error="missing_stream_scores")
            continue
        wav_root = path.parent / "wav"
        wavs: list[Path] = []
        for name, score in streams.items():
            if not isinstance(score, dict) or score.get("cer") is None:
                _append(failures, path=str(path), uid=uid, stream=name, error="missing_stream_cer")
            tag = "peak" if name == "original" else str(name)
            wav = wav_root / f"{uid}_{tag}.wav"
            if not wav.is_file():
                _append(failures, path=str(path), uid=uid, stream=name, error="missing_wav")
            else:
                wavs.append(wav)
        if check_duration and wavs:
            try:
                durations = [_duration(wav) for wav in wavs]
                if max(durations) - min(durations) > 0.020:
                    _append(
                        failures, path=str(path), uid=uid, error="stream_duration_mismatch",
                        min_sec=min(durations), max_sec=max(durations),
                    )
            except Exception as exc:
                _append(failures, path=str(path), uid=uid, error=f"wav_info:{exc}")
        valid.append(row)
    return valid, seen


def audit_sep_root(
    root: Path,
    splits: Iterable[str] = ("pos", "neg"),
    *,
    expected_uids: int = 1838,
    check_duration: bool = False,
    require_handoff: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    handoff_path = find_handoff(root)
    handoff = None
    if handoff_path is None:
        if require_handoff:
            _append(failures, path=str(root), error="missing_handoff")
        else:
            warnings.append({"error": "missing_handoff_legacy_tree"})
    else:
        try:
            handoff = load_handoff(handoff_path)
            if handoff.get("schema") == LEGACY_SCHEMA:
                warnings.append({"error": "legacy_handoff_no_full_audio_guarantee"})
                if require_handoff:
                    _append(failures, path=str(handoff_path), error="v2_handoff_required")
            elif handoff.get("schema") != SCHEMA:
                _append(failures, path=str(handoff_path), error="unexpected_handoff_schema")
            if expected_uids and int(handoff.get("n_records") or 0) != expected_uids:
                _append(
                    failures, path=str(handoff_path), error="handoff_coverage",
                    observed=handoff.get("n_records"), expected=expected_uids,
                )
        except Exception as exc:
            _append(failures, path=str(handoff_path), error=f"handoff:{exc}")

    best_path = root / "best_sep" / "index.jsonl"
    best_rows = load_jsonl(best_path) if best_path.is_file() else []
    if not best_rows:
        _append(failures, path=str(best_path), error="missing_or_empty_best_sep")
    best_seen: set[str] = set()
    split_expected: dict[str, int] = {}
    best_stage_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    for row in best_rows:
        uid = str(row.get("uid") or "")
        split = str(row.get("split") or "")
        if not uid or uid in best_seen:
            _append(failures, path=str(best_path), uid=uid, error="duplicate_or_empty_uid")
        best_seen.add(uid)
        split_expected[split] = split_expected.get(split, 0) + 1
        expected_metric = "cer_py" if CJK.search(str(row.get("wake_text") or "")) else "cer_char"
        if row.get("metric") != expected_metric:
            _append(failures, path=str(best_path), uid=uid, error="best_metric_mismatch")
        dest = root / "best_sep" / str(row.get("dest_rel") or "")
        if not dest.is_file():
            _append(failures, path=str(dest), uid=uid, error="missing_best_wav")
        stage_index = root / split / str(row.get("best_stage") or "") / "index.jsonl"
        if not stage_index.is_file():
            _append(failures, path=str(stage_index), uid=uid, error="missing_best_stage_index")
        else:
            if stage_index not in best_stage_cache:
                stage_rows = load_jsonl(stage_index)
                stage_map: dict[str, dict[str, Any]] = {}
                for stage_row in stage_rows:
                    stage_uid = str(stage_row.get("uid") or "")
                    if stage_uid in stage_map:
                        _append(failures, path=str(stage_index), uid=stage_uid, error="duplicate_uid")
                    stage_map[stage_uid] = stage_row
                best_stage_cache[stage_index] = stage_map
            source_row = best_stage_cache[stage_index].get(uid)
            stream = str(row.get("oracle_stream") or "")
            stream_score = ((source_row or {}).get("streams") or {}).get(stream)
            if source_row is None or not isinstance(stream_score, dict):
                _append(failures, path=str(stage_index), uid=uid, error="best_stream_not_in_stage")
            else:
                try:
                    cer_mismatch = abs(
                        float(stream_score.get("cer")) - float(row.get("oracle_cer"))
                    ) > 1e-6
                except (TypeError, ValueError):
                    cer_mismatch = True
                if cer_mismatch:
                    _append(failures, path=str(stage_index), uid=uid, error="best_cer_mismatch")
            if check_duration and dest.is_file() and source_row is not None:
                tag = "peak" if stream == "original" else stream
                source_wav = stage_index.parent / "wav" / f"{uid}_{tag}.wav"
                try:
                    if abs(_duration(dest) - _duration(source_wav)) > 0.020:
                        _append(failures, uid=uid, error="best_wav_duration_mismatch")
                except Exception as exc:
                    _append(failures, uid=uid, error=f"best_wav_info:{exc}")

    if expected_uids and len(best_seen) != expected_uids:
        _append(failures, error="best_uid_coverage", observed=len(best_seen), expected=expected_uids)

    split_reports: dict[str, Any] = {}
    for split in splits:
        expected = split_expected.get(split, 0)
        stage_stats: dict[str, Any] = {}
        for key, dirname in STAGES.items():
            stage_root = root / split / dirname
            if key in {"s1", "s2", "s3", "s4"}:
                rows, seen = _check_rows(
                    stage_root / "index.jsonl", split=split,
                    check_duration=check_duration, failures=failures,
                )
                if expected and len(seen) != expected:
                    _append(
                        failures, split=split, stage=key, error="full_stage_coverage",
                        observed=len(seen), expected=expected,
                    )
                stage_stats[key] = {"n_rows": len(rows), "n_uid": len(seen)}
                continue
            summary_path = stage_root / "summary.json"
            if not summary_path.is_file():
                _append(failures, path=str(summary_path), error="missing_gate_summary")
                continue
            summary = _read_json(summary_path)
            if summary.get("partial"):
                _append(failures, path=str(summary_path), error="partial_gate_run")
            if expected and int(summary.get("catalog_n") or 0) != expected:
                _append(
                    failures, path=str(summary_path), error="gate_catalog_coverage",
                    observed=summary.get("catalog_n"), expected=expected,
                )
            aliases = (summary.get("gate_dedup") or {}).get("aliases") or {}
            by_thr = summary.get("by_thr") or {}
            for name in ("a", "b", "c"):
                if name not in by_thr:
                    _append(failures, path=str(summary_path), error="missing_gate_threshold", threshold=name)
                    continue
                info = by_thr.get(name) or {}
                duplicate_of = info.get("duplicate_of") or aliases.get(name)
                if duplicate_of:
                    if duplicate_of == name or duplicate_of not in by_thr:
                        _append(failures, path=str(summary_path), error="bad_gate_alias", alias=name)
                    if int(info.get("n_subset") or 0) != int((by_thr.get(duplicate_of) or {}).get("n_subset") or 0):
                        _append(failures, path=str(summary_path), error="gate_alias_size_mismatch", alias=name)
                    continue
                n_subset = int(info.get("n_subset") or 0)
                if n_subset:
                    rows, seen = _check_rows(
                        stage_root / f"thr_{name}" / "index.jsonl", split=split,
                        check_duration=check_duration, failures=failures,
                    )
                    if len(seen) != n_subset:
                        _append(
                            failures, path=str(summary_path), threshold=name,
                            error="gate_subset_coverage", observed=len(seen), expected=n_subset,
                        )
            stage_stats[key] = {"aliases": aliases, "n_unique_cohorts": (summary.get("gate_dedup") or {}).get("n_unique_cohorts")}
        split_reports[split] = {"expected_uids": expected, "stages": stage_stats}

    return {
        "schema": "kws_sep_input_audit/v1",
        "root": str(root),
        "check_duration": check_duration,
        "require_handoff": require_handoff,
        "handoff_schema": handoff.get("schema") if handoff else None,
        "n_best_uid": len(best_seen),
        "splits": split_reports,
        "warnings": warnings,
        "failures": failures,
        "ok": not failures,
    }
