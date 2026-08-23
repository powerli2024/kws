from pathlib import Path

from kws.wav_paths import (
    parse_uid,
    resolve_cmd_wav,
    resolve_kws_wav,
    resolve_stream_wav,
    stream_wav_tag,
)


def test_original_stream_is_peak_tag():
    assert stream_wav_tag("original") == "peak"
    assert stream_wav_tag("spk1") == "spk1"
    assert stream_wav_tag("spk1_r1") == "spk1_r1"


def test_parse_uid():
    assert parse_uid("pos_0", {"split": "pos"}) == ("pos", "0")
    assert parse_uid("neg_1000") == ("neg", "1000")


def test_resolve_kws_and_cmd(tmp_path: Path):
    data = tmp_path / "datasetA"
    (data / "pos").mkdir(parents=True)
    kws = data / "pos" / "kws_0.wav"
    cmd = data / "pos" / "cmd_0.wav"
    kws.write_bytes(b"x")
    cmd.write_bytes(b"y")
    rec = {"uid": "pos_0", "split": "pos", "kws_rel": "pos/kws_0.wav"}
    # AutoDL absolute path must be ignored
    rec_bad = {**rec, "kws_path": "/root/datasetA/pos/kws_0.wav"}
    assert resolve_kws_wav(data, rec_bad) == kws.resolve()
    assert resolve_cmd_wav(data, rec) == cmd.resolve()


def test_resolve_stream_peak_and_nested_stage(tmp_path: Path):
    pos_neg = tmp_path / "pos_neg"
    wav_dir = pos_neg / "pos" / "s1_onnx_full" / "wav"
    wav_dir.mkdir(parents=True)
    peak = wav_dir / "pos_0_peak.wav"
    spk1 = wav_dir / "pos_0_spk1.wav"
    peak.write_bytes(b"a")
    spk1.write_bytes(b"b")
    rec = {"uid": "pos_0", "split": "pos", "best_stage": "s1_onnx_full"}
    assert resolve_stream_wav(pos_neg, rec, "original") == peak.resolve()
    assert resolve_stream_wav(pos_neg, rec, "spk1") == spk1.resolve()

    nested = pos_neg / "pos" / "s7_cv_then_onnx_gate" / "thr_a" / "wav"
    nested.mkdir(parents=True)
    r1 = nested / "pos_3_spk1_r1.wav"
    r1.write_bytes(b"c")
    rec7 = {"uid": "pos_3", "split": "pos", "best_stage": "s7_cv_then_onnx_gate/thr_a"}
    assert resolve_stream_wav(pos_neg, rec7, "spk1_r1") == r1.resolve()
