from kws.oracle import oracle_of
from kws.select_l2 import select_l1_l2
from kws.skip_sep import skip_bss_before_sep, skip_sep_after_scores


def test_oracle_prefers_original_on_tie():
    name, cer = oracle_of(
        {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.3}}
    )
    assert name == "original" and cer == 0.0


def test_oracle_picks_better_sep():
    name, cer = oracle_of(
        {"original": {"cer": 0.4}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.2}}
    )
    assert name == "spk1" and cer == 0.0


def test_skip_only_when_original_uniquely_zero():
    d = skip_sep_after_scores(
        {"original": {"cer": 0.0}, "spk1": {"cer": 0.2}, "spk2": {"cer": 0.5}}
    )
    assert d.skip and d.reason == "skip_sep_text_safe"
    d2 = skip_sep_after_scores(
        {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.8}}
    )
    assert not d2.skip and d2.reason == "need_l2_not_cer"


def test_pre_bss_ignores_duration_and_defaults_off():
    d = skip_bss_before_sep(p_music=0.01, snr_med_db=30.0, dur_sec=0.4, enabled=False)
    assert not d.skip
    d2 = skip_bss_before_sep(p_music=0.01, snr_med_db=30.0, dur_sec=0.4, enabled=True)
    assert d2.skip and d2.reason == "pre_bss_residual_clean"


def test_l2_without_qkw_degrades_and_does_not_rank_by_cos():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.4}}
    sel = select_l1_l2(streams, cos_to_raw={"original": 0.50, "spk1": 0.99, "spk2": 0.1})
    assert sel.l2_degraded and sel.chosen == "original"
    assert sel.reason == "l2_degraded_no_text_sidecar"


def test_l2_ranks_by_qkw_not_cos_to_raw():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.4}}
    sel = select_l1_l2(
        streams,
        q_kw={"original": 0.4, "spk1": 0.9, "spk2": 0.1},
        cos_to_raw={"original": 0.99, "spk1": 0.95, "spk2": 0.2},
    )
    assert sel.chosen == "spk1" and not sel.reverted_catastrophe and not sel.l2_degraded


def test_l2_reverts_sep_below_catastrophe():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}}
    sel = select_l1_l2(
        streams,
        q_kw={"original": 0.4, "spk1": 0.95},
        cos_to_raw={"original": 1.0, "spk1": 0.80},
        catastrophe_cos=0.90,
    )
    assert sel.chosen == "original" and sel.reverted_catastrophe


def test_l2_rejects_two_high_text_speakers():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.0}}
    sel = select_l1_l2(
        streams,
        q_kw={"original": 0.2, "spk1": 0.92, "spk2": 0.91},
        pair_cos={"spk1|spk2": 0.10},
    )
    assert sel.rejected and sel.chosen == "reject"


def test_l2_nll_can_rank_but_cannot_use_absolute_reject_threshold():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.0}}
    sel = select_l1_l2(
        streams,
        q_kw={"original": -3.0, "spk1": -0.1, "spk2": -0.2},
        q_kw_kind="nll",
        pair_cos={"spk1|spk2": 0.10},
    )
    assert not sel.rejected and sel.chosen == "spk1"
