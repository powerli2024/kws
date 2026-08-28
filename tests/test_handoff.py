from pathlib import Path

import pytest

from kws.handoff import HandoffError, SCHEMA, load_handoff


def test_load_handoff_ok(tmp_path: Path) -> None:
    p = tmp_path / "kws_handoff.json"
    p.write_text(
        '{"schema":"%s","mms_fa":false,"selector_within_stage":"oracle_cer_prefer_original",'
        '"audio_length_policy":"full_utterance_no_truncation","max_sep_sec":0}'
        % SCHEMA,
        encoding="utf-8",
    )
    obj = load_handoff(p)
    assert obj["mms_fa"] is False


def test_rejects_mms_fa(tmp_path: Path) -> None:
    p = tmp_path / "kws_handoff.json"
    p.write_text('{"schema":"%s","mms_fa":true}' % SCHEMA, encoding="utf-8")
    with pytest.raises(HandoffError, match="mms_fa"):
        load_handoff(p)


def test_rejects_wrong_schema(tmp_path: Path) -> None:
    p = tmp_path / "kws_handoff.json"
    p.write_text('{"schema":"nope","mms_fa":false}', encoding="utf-8")
    with pytest.raises(HandoffError, match="schema"):
        load_handoff(p)


def test_rejects_v2_truncation(tmp_path: Path) -> None:
    p = tmp_path / "kws_handoff.json"
    p.write_text(
        '{"schema":"%s","mms_fa":false,"audio_length_policy":"energy_crop",'
        '"max_sep_sec":3}' % SCHEMA,
        encoding="utf-8",
    )
    with pytest.raises(HandoffError, match="full_utterance"):
        load_handoff(p)
