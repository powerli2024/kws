import json
from pathlib import Path

import pytest

from kws.stage_compare import build_report


def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(uid: str, cer: float, **extra):
    return {
        "uid": uid, "oracle_cer": cer, "oracle_stream": "original", "oracle_hyp": "x",
        "streams": {"original": {"hyp": "x", "cer": cer}}, **extra,
    }


def test_duplicate_threshold_cohort_and_full_parent_fallback(tmp_path):
    parent = [_row("pos_0", 0.0), _row("pos_1", 0.5), _row("pos_2", 1.0)]
    gated_a = [_row("pos_1", 0.0, parent_stage="s1", parent_oracle_cer=0.5, thr=0.5), _row("pos_2", 0.5, parent_stage="s1", parent_oracle_cer=1.0, thr=0.5)]
    gated_b = [dict(row, thr=0.75) for row in gated_a]
    _write(tmp_path / "pos" / "s1_onnx_full" / "index.jsonl", parent)
    _write(tmp_path / "pos" / "s5_onnx_then_cv_gate" / "thr_a" / "index.jsonl", gated_a)
    _write(tmp_path / "pos" / "s5_onnx_then_cv_gate" / "thr_b" / "index.jsonl", gated_b)
    report = build_report(tmp_path, ["pos"])
    block = report["splits"]["pos"]
    assert block["duplicate_same_threshold_value"] == []
    assert block["duplicate_same_gate_cohort"] == [[
        "s5_onnx_then_cv_gate/thr_a", "s5_onnx_then_cv_gate/thr_b"
    ]]
    assert block["duplicate_same_semantic_results"] == [[
        "s5_onnx_then_cv_gate/thr_a", "s5_onnx_then_cv_gate/thr_b"
    ]]
    arm = block["arms"]["s5_onnx_then_cv_gate/thr_a"]
    assert arm["metrics_full_parent_fallback"]["n"] == 3
    assert arm["metrics_full_parent_fallback"]["mean_cer"] == pytest.approx(1 / 6)
