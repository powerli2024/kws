import numpy as np

from kws.catastrophe import pick_threshold_from_clean_floor
from kws.need_se import need_se, se_safety_ok
from kws.presence_protocol import enroll_go
from kws.window_mincos import allow_mincos_veto, pairwise_min_cos, window_starts


def test_short_kws_skips_window_metric():
    spec = window_starts(int(0.7 * 16000), 16000)
    assert spec.skipped and spec.reason == "dur_lt_0p8s"


def test_two_windows_on_one_second():
    spec = window_starts(16000, 16000)
    assert not spec.skipped and len(spec.starts) >= 2


def test_mincos_veto_needs_1p2s_speech():
    assert not allow_mincos_veto(1.0)
    assert allow_mincos_veto(1.2)


def test_pairwise_min_cos_identical():
    e = np.ones(8)
    assert abs(pairwise_min_cos([e, e, e]) - 1.0) < 1e-6


def test_need_se_original_music():
    n = need_se(winner_is_original=True, p_music=0.55, snr_med_db=20.0)
    assert n.need and "p_music" in n.reason
    n2 = need_se(winner_is_original=True, p_music=0.1, snr_med_db=20.0)
    assert not n2.need


def test_se_safety_rejects_cos_collapse():
    ok, why = se_safety_ok(cos_se_pre=0.80, cer_se=0.0, cer_pre=0.0)
    assert not ok and why == "cos_collapse"


def test_catastrophe_grid_clip():
    thr = pick_threshold_from_clean_floor([0.93, 0.94, 0.95, 0.99], quantile=0.0)
    assert thr in (0.90, 0.91, 0.92, 0.93, 0.94, 0.95)


def test_presence_veto_blocks_far_up():
    v = enroll_go(
        baseline={"frr": 0.17, "far": 0.11},
        candidate={"frr": 0.12, "far": 0.20},
        cer_mean=0.02,
        cer0_rate=0.94,
        cer0_rate_baseline=0.94,
    )
    assert not v.accept and v.reason == "frr_down_but_far_up"


def test_presence_veto_accepts_frr_drop():
    v = enroll_go(
        baseline={"frr": 0.17, "far": 0.11},
        candidate={"frr": 0.15, "far": 0.11},
        cer_mean=0.02,
        cer0_rate=0.94,
        cer0_rate_baseline=0.94,
    )
    assert v.accept
