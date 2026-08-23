from pathlib import Path

from kws.audio import cosine_sim, load_wav_mono, save_wav_mono
from kws.eres import FftProxyEncoder
from kws.export_groups import export_one
from kws.se_backend import spectral_subtract
from kws.sidecar import parse_cos_row
import numpy as np


def test_fft_encoder_and_cos_payload(tmp_path: Path):
    enc = FftProxyEncoder()
    a = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
    b = a.copy()
    c = np.random.RandomState(1).randn(16000).astype(np.float32) * 0.1
    ea, eb, ec = enc.embed(a), enc.embed(b), enc.embed(c)
    assert cosine_sim(ea, eb) > cosine_sim(ea, ec)
    row = {"uid": "pos_0", "cos_to_raw": {"original": 0.99, "spk1": 0.4}}
    uid, payload = parse_cos_row(row)
    assert uid == "pos_0" and payload["original"] == 0.99


def test_spectral_se_keeps_length():
    x = np.random.RandomState(2).randn(8000).astype(np.float32) * 0.05
    y = spectral_subtract(x, 16000)
    assert y.shape == x.shape


def test_export_raw_kws(tmp_path: Path):
    data = tmp_path / "data"
    (data / "pos").mkdir(parents=True)
    src = data / "pos" / "kws_0.wav"
    wav = np.zeros(160, dtype=np.float32)
    save_wav_mono(src, wav, 16000)
    rec = {
        "uid": "pos_0",
        "split": "pos",
        "kws_rel": "pos/kws_0.wav",
        "oracle_stream": "spk2",
        "arms": {"T0": {"chosen": "spk2"}, "T2": {"chosen": "spk1"}},
    }
    dest = tmp_path / "best"
    row = export_one(rec, "raw_kws", dest, pos_neg=tmp_path / "pn", data_dir=data)
    assert row["ok"]
    loaded, sr = load_wav_mono(dest / "pos" / "pos_0.wav")
    assert sr == 16000 and loaded.size > 0
