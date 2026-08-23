from kws.cmd_eval import auc_scores, eer_and_threshold, rank_groups, summarize_cmd_scores
from kws.sidecar import parse_qkw_row
from kws.stats import wilson_interval
from kws.export_groups import chosen_stream
from kws.iojson import limit_rows_balanced


def test_limit_rows_balanced_keeps_neg():
    rows = [{"uid": f"pos_{i}", "split": "pos"} for i in range(10)]
    rows += [{"uid": f"neg_{i}", "split": "neg"} for i in range(10)]
    out = limit_rows_balanced(rows, 4)
    assert len(out) == 4
    assert {r["split"] for r in out} == {"pos", "neg"}


def test_auc_perfect_and_chance():
    assert auc_scores([0.9, 0.8], [0.1, 0.2]) == 1.0
    a = auc_scores([0.5, 0.5], [0.5, 0.5])
    assert a is not None and abs(a - 0.5) < 1e-9


def test_eer_separable():
    e = eer_and_threshold([0.8, 0.9, 0.7], [0.1, 0.2, 0.15])
    assert e["eer"] is not None and e["eer"] < 0.05


def test_summarize_gap():
    s = summarize_cmd_scores([0.8, 0.9], [0.1, 0.2])
    assert s["mean_gap"] is not None and s["mean_gap"] > 0.5
    assert s["n_pos"] == 2 and s["n_neg"] == 2
    assert "locked_tau_probe_not_adopt" in s


def test_rank_prefers_lower_eer():
    a = summarize_cmd_scores([0.9, 0.85], [0.1, 0.15])
    b = summarize_cmd_scores([0.6, 0.55], [0.5, 0.45])
    r = rank_groups({"t0": b, "t2": a}, baseline="t0")
    assert r["best"] == "t2"
    assert r["beats_baseline"]


def test_wilson_and_qkw_nll():
    p, lo, hi = wilson_interval(0, 474)
    assert p == 0.0 and hi > 0
    uid, payload = parse_qkw_row({"uid": "pos_0", "nll": {"original": 2.0, "spk1": 0.5}})
    assert uid == "pos_0" and payload["spk1"] > payload["original"]


def test_chosen_stream_skip_and_t2():
    rec = {
        "oracle_stream": "original",
        "orig_unique_zero": True,
        "skip_sep_after_scores": True,
        "arms": {"T0": {"chosen": "original"}, "T2": {"chosen": "spk1"}},
    }
    assert chosen_stream(rec, "raw_kws")[0] == "raw_kws"
    assert chosen_stream(rec, "t2")[0] == "spk1"
    assert chosen_stream(rec, "skip_then_t2") == ("original", "skip_unique_zero")
    rec2 = {
        "oracle_stream": "original",
        "orig_unique_zero": False,
        "arms": {"T0": {"chosen": "original"}, "T2": {"chosen": "spk1"}},
    }
    assert chosen_stream(rec2, "skip_then_t2")[0] == "spk1"
    assert chosen_stream(rec2, "t0")[0] == "original"
