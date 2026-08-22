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


def test_l2_allows_sep_on_dual_zero_when_cos_ok():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}, "spk2": {"cer": 0.4}}
    sel = select_l1_l2(streams, cos_to_raw={"original": 1.0, "spk1": 0.96, "spk2": 0.2})
    # original 1.0 still wins because score is higher
    assert sel.chosen == "original" and sel.dual_zero
    sel2 = select_l1_l2(
        streams,
        cos_to_raw={"original": 0.91, "spk1": 0.97, "spk2": 0.2},
    )
    assert sel2.chosen == "spk1" and not sel2.reverted_catastrophe


def test_l2_reverts_sep_below_catastrophe():
    streams = {"original": {"cer": 0.0}, "spk1": {"cer": 0.0}}
    sel = select_l1_l2(
        streams,
        cos_to_raw={"original": 1.0, "spk1": 0.80},
        catastrophe_cos=0.90,
    )
    # L1 eligible both; L2 would pick original anyway due to higher cos
    assert sel.chosen == "original"
    sel2 = select_l1_l2(
        streams,
        cos_to_raw={"original": 0.85, "spk1": 0.88},
        catastrophe_cos=0.90,
    )
    assert sel2.chosen == "original" and sel2.reverted_catastrophe
