from kws.cer_metric import cer_detail
from kws.se_route import choose, paired, route_one
import pytest


def candidate(role="s1", view="raw", cer=0.0, nll=None, stream="original", eligible=True):
    return {
        "role": role, "view": view, "cer": cer, "nll": nll, "stream": stream,
        "arm": role, "se_eligible": eligible,
    }


def test_chinese_uses_toneless_pinyin():
    pytest.importorskip("pypinyin")
    assert cer_detail("科目", "科慕")["cer"] == 0.0
    assert cer_detail("科目", "科慕")["cer_char"] > 0.0


def test_raw_wins_exact_tie_but_nll_can_select_safe_se():
    raw = candidate(nll=0.5)
    se = candidate(view="se", nll=0.5)
    assert choose([se, raw])["view"] == "raw"
    se["nll"] = 0.4
    assert choose([raw, se])["view"] == "se"


def test_s7_is_triggered_but_never_force_switched():
    worse = route_one([candidate(cer=0.2), candidate(role="s7", cer=0.3)], allow_se=False)
    assert worse["triggered_s7"] and not worse["switched_s7"]
    better = route_one([candidate(cer=0.2), candidate(role="s7", cer=0.0)], allow_se=False)
    assert better["switched_s7"] and better["selected"]["role"] == "s7"


def test_unsafe_se_cannot_enter_route():
    result = route_one(
        [candidate(cer=0.2), candidate(view="se", cer=0.0, eligible=False)],
        allow_se=True,
    )
    assert result["selected"]["view"] == "raw"


def test_paired_counts_regressions():
    base = [{"uid": "a", "ok": True, "selected": {"cer": 0.2}}]
    trial = [{"uid": "a", "ok": True, "selected": {"cer": 0.0}}]
    assert paired(base, trial)["n_improved"] == 1
