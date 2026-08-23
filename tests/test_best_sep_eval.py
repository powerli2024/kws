from kws.best_sep_eval import cer_constraint, pairwise_disagreement, verdict
from kws.presence_protocol import enroll_go


def _summ(**kw):
    base = {
        "oracle_cer_mean": 0.01,
        "cer0_rate": 0.94,
        "n_ok": 10,
        "n_missing_wav": 0,
        "uids": {},
    }
    base.update(kw)
    return base


def test_cer_holds():
    c = cer_constraint(_summ(), _summ())
    assert c["ok"]


def test_cer_mean_rejects():
    c = cer_constraint(_summ(oracle_cer_mean=0.04), _summ())
    assert not c["ok"]
    assert "cer_mean_over_0.03" in c["reasons"]


def test_cer0_drop_rejects():
    c = cer_constraint(_summ(cer0_rate=0.90), _summ(cer0_rate=0.94))
    assert not c["ok"]
    assert "cer0_rate_drop_over_2pp" in c["reasons"]


def test_pairwise_fingerprint():
    a = {
        "uids": {
            "pos_0": {"fp": "aa", "oracle_stream": "spk1", "best_stage": "s1"},
            "pos_1": {"fp": "bb", "oracle_stream": "original", "best_stage": "s1"},
        }
    }
    b = {
        "uids": {
            "pos_0": {"fp": "aa", "oracle_stream": "spk1", "best_stage": "s1"},
            "pos_1": {"fp": "cc", "oracle_stream": "spk2", "best_stage": "s2"},
        }
    }
    d = pairwise_disagreement(a, b)
    assert d["n_common"] == 2
    assert d["n_wav_fingerprint_diff"] == 1
    assert d["n_oracle_stream_diff"] == 1
    assert d["n_best_stage_diff"] == 1


def test_verdict_points_at_cmd_cosine_without_presence():
    s = {"cur": _summ(), "alt": _summ(oracle_cer_mean=0.0, cer0_rate=0.95)}
    v = verdict(names=["cur", "alt"], summaries=s, presence=None, baseline="cur")
    assert v["adopt"] is None
    assert "CMD cosine" in v["note"] or "enroll↔CMD" in v["note"]


def test_verdict_adopts_when_presence_improves():
    s = {"cur": _summ(), "alt": _summ()}
    presence = {
        "cur": {"frr": 0.10, "far": 0.20},
        "alt": {"frr": 0.08, "far": 0.20},
    }
    v = verdict(names=["cur", "alt"], summaries=s, presence=presence, baseline="cur")
    assert v["adopt"] == "alt"
    go = enroll_go(
        baseline=presence["cur"],
        candidate=presence["alt"],
        cer_mean=0.01,
        cer0_rate=0.94,
        cer0_rate_baseline=0.94,
    )
    assert go.accept
