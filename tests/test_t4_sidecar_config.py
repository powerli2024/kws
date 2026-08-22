import pytest

from kws.arms import CER_ORACLE_ARMS, L2_ARMS
from kws.config import matrix, validate_code_defaults
from kws.select_l2 import DEFAULT_CATASTROPHE_COS, select_l1_l2
from kws.sidecar import SidecarError, parse_cos_row
from kws.t0_t4 import pick_track


def test_yaml_and_code_defaults_agree():
    m = matrix()
    assert m["runtime"]["catastrophe_cos"] == 0.92
    assert DEFAULT_CATASTROPHE_COS == 0.92
    validate_code_defaults()


def test_t4_not_in_l2_set():
    assert "T4" in CER_ORACLE_ARMS
    assert "T4" not in L2_ARMS


def test_t4_keeps_cer_oracle_when_sidecar_prefers_sep():
    rec = {
        "uid": "u",
        "oracle_stream": "original",
        "dual_zero": True,
        "streams": {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}},
    }
    cos = {"u": {"original": 0.5, "spk1": 0.99}}
    t4 = pick_track("T4", rec, cos_map=cos, pm_map={})
    t2 = pick_track("T2", rec, cos_map=cos, pm_map={})
    assert t4["chosen"] == "original"
    assert t2["chosen"] == "spk1"


def test_sidecar_empty_scores_raises():
    with pytest.raises(SidecarError, match="empty dict"):
        parse_cos_row({"uid": "a", "scores": {}})


def test_sidecar_whole_row_forbidden():
    with pytest.raises(SidecarError, match="exactly one"):
        parse_cos_row({"uid": "a", "split": "pos", "original": 1.0})


def test_partial_cos_sidecar_raises():
    rec = {
        "uid": "u",
        "oracle_stream": "original",
        "dual_zero": True,
        "streams": {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}},
    }
    with pytest.raises(SidecarError, match="missing from cos sidecar"):
        pick_track("T2", rec, cos_map={"other": {"original": 1.0, "spk1": 0.9}}, pm_map={})


def test_l2_missing_stream_cos_raises():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}}
    with pytest.raises(SidecarError, match="missing streams"):
        select_l1_l2(streams, cos_to_raw={"original": 1.0})
