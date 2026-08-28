import json
import subprocess
import sys
from pathlib import Path

import pytest

from kws.fa_experiment import AlignmentEvidence, parse_alignment_stream, route_by_agreement, safe_crop_plan
from kws.sidecar import SidecarError


def ev(p10=-0.2, mean=-0.1, *, start=0.4, end=1.1, duration=2.0, coverage=1.0, edge=False):
    return AlignmentEvidence(coverage, start, end, duration, mean, p10, 0.0, edge)


def test_agreement_switches_only_when_qkw_and_fa_match():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.5}}
    result = route_by_agreement(
        streams,
        t0="original",
        qkw={"original": 0.1, "spk1": 0.9, "spk2": 0.0},
        evidence={"original": ev(-1.0), "spk1": ev(-0.1), "spk2": ev(-2.0)},
    )
    assert result.agreed and result.chosen == "spk1"


def test_disagreement_falls_back_to_t0():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}}
    result = route_by_agreement(
        streams,
        t0="original",
        qkw={"original": 0.1, "spk1": 0.9},
        evidence={"original": ev(-0.1), "spk1": ev(-1.0)},
    )
    assert not result.agreed and result.chosen == "original"
    assert result.reason == "qkw_fa_disagree_fallback_t0"


def test_safe_crop_expands_to_minimum_duration():
    plan = safe_crop_plan(ev(start=0.8, end=1.0, duration=2.5), margin_sec=0.1, min_output_sec=1.5)
    assert plan.apply
    assert plan.end_sec - plan.start_sec == pytest.approx(1.5)


def test_unsafe_crop_falls_back_full():
    plan = safe_crop_plan(ev(duration=1.2), min_output_sec=1.5)
    assert not plan.apply and (plan.start_sec, plan.end_sec) == (0.0, 1.2)
    assert not safe_crop_plan(ev(edge=True)).apply


def test_alignment_contract_rejects_bad_span():
    with pytest.raises(SidecarError, match="invalid alignment span"):
        parse_alignment_stream(
            {"coverage": 1.0, "start_sec": 1.0, "end_sec": 0.5, "duration_sec": 2.0},
            uid="u", stream="spk1",
        )


def test_fa_experiment_script_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    enriched = tmp_path / "enriched.jsonl"
    qkw = tmp_path / "qkw.jsonl"
    fa = tmp_path / "fa.jsonl"
    out = tmp_path / "picks.jsonl"
    summary = tmp_path / "summary.json"
    enriched.write_text(json.dumps({
        "uid": "pos_0", "split": "pos", "id": 0, "wake_text": "你好科慕", "lang": "zh",
        "best_stage": "s1", "oracle_stream": "original", "dual_zero": True,
        "streams": {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    qkw.write_text(json.dumps({"uid": "pos_0", "q_kw": {"original": 0.1, "spk1": 0.9}}) + "\n")
    fa.write_text(json.dumps({
        "uid": "pos_0", "model": "mms_fa", "streams": {
            "original": {"coverage": 1.0, "mean_logp": -1.0, "p10_logp": -1.5, "start_sec": 0.3, "end_sec": 1.1, "duration_sec": 2.0},
            "spk1": {"coverage": 1.0, "mean_logp": -0.1, "p10_logp": -0.2, "start_sec": 0.4, "end_sec": 1.2, "duration_sec": 2.0},
        },
    }) + "\n")
    subprocess.run([
        sys.executable, str(root / "scripts" / "run_fa_experiment.py"),
        "--enriched", str(enriched), "--qkw-jsonl", str(qkw),
        "--route-fa-jsonl", str(fa), "--out", str(out), "--summary", str(summary),
    ], cwd=root, check=True, capture_output=True, text=True)
    pick = json.loads(out.read_text(encoding="utf-8"))
    meta = json.loads(summary.read_text(encoding="utf-8"))
    assert pick["arms"]["F3_agree_full"]["chosen"] == "spk1"
    assert meta["coverage"] == 1.0 and meta["n_agree"] == 1
