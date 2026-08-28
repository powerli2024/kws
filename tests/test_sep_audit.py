import json
from pathlib import Path

from kws.sep_audit import STAGES, audit_sep_root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _row(uid: str, split: str) -> dict:
    return {
        "uid": uid, "split": split, "wake_text": "你好", "metric": "pinyin",
        "oracle_stream": "original", "oracle_cer": 0.0,
        "streams": {"original": {"hyp": "你好", "cer": 0.0}},
    }


def _tree(root: Path) -> None:
    handoff = {
        "schema": "kws_sep_handoff/v2", "mms_fa": False,
        "selector_within_stage": "oracle_cer_prefer_original",
        "audio_length_policy": "full_utterance_no_truncation", "max_sep_sec": 0,
        "n_records": 2,
    }
    (root / "kws_handoff.json").write_text(json.dumps(handoff), encoding="utf-8")
    best = []
    for split in ("pos", "neg"):
        uid = f"{split}_1"
        row = _row(uid, split)
        for key in ("s1", "s2", "s3", "s4"):
            stage = root / split / STAGES[key]
            _write_jsonl(stage / "index.jsonl", [row])
            (stage / "wav").mkdir(exist_ok=True)
            (stage / "wav" / f"{uid}_peak.wav").write_bytes(b"wav")
        for key in ("s5", "s6", "s7", "s8"):
            stage = root / split / STAGES[key]
            _write_jsonl(stage / "thr_a" / "index.jsonl", [row])
            (stage / "thr_a" / "wav").mkdir(exist_ok=True)
            (stage / "thr_a" / "wav" / f"{uid}_peak.wav").write_bytes(b"wav")
            summary = {
                "partial": False, "catalog_n": 1,
                "by_thr": {
                    "a": {"n_subset": 1},
                    "b": {"n_subset": 1, "duplicate_of": "a"},
                    "c": {"n_subset": 1, "duplicate_of": "a"},
                },
                "gate_dedup": {
                    "aliases": {"b": "a", "c": "a"}, "n_unique_cohorts": 1,
                },
            }
            (stage / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        dest = root / "best_sep" / split / f"{uid}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wav")
        best.append({
            "uid": uid, "split": split, "wake_text": "你好", "metric": "cer_py",
            "best_stage": STAGES["s1"], "oracle_stream": "original", "oracle_cer": 0.0,
            "dest_rel": f"{split}/{uid}.wav",
        })
    _write_jsonl(root / "best_sep" / "index.jsonl", best)


def test_strict_sep_tree_contract(tmp_path: Path) -> None:
    _tree(tmp_path)
    report = audit_sep_root(tmp_path, expected_uids=2, require_handoff=True)
    assert report["ok"], report["failures"]


def test_rejects_wrong_stage_metric(tmp_path: Path) -> None:
    _tree(tmp_path)
    path = tmp_path / "pos" / STAGES["s1"] / "index.jsonl"
    row = _row("pos_1", "pos")
    row["metric"] = "char"
    _write_jsonl(path, [row])
    report = audit_sep_root(tmp_path, expected_uids=2, require_handoff=True)
    assert not report["ok"]
    assert any(item.get("error") == "metric_mismatch" for item in report["failures"])
