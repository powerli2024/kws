"""Wav I/O and cosine. soundfile preferred; stdlib wave is the fallback."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a / na, b / nb))


def load_wav_mono(path: Path, *, sr: int = 16000) -> tuple[np.ndarray, int]:
    path = Path(path)
    wav, file_sr = _read(path)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(file_sr) != int(sr):
        wav = resample_linear(wav, int(file_sr), int(sr))
        file_sr = sr
    return wav.astype(np.float32), int(file_sr)


def _read(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        x, file_sr = sf.read(str(path), always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        return x.reshape(-1), int(file_sr)
    except Exception:
        pass
    with wave.open(str(path), "rb") as w:
        file_sr = int(w.getframerate())
        nch = int(w.getnchannels())
        sw = int(w.getsampwidth())
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported sample width {sw} for {path}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x.reshape(-1), file_sr


def resample_linear(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr or x.size == 0:
        return x
    n_out = max(1, int(round(x.size * target_sr / orig_sr)))
    t = np.linspace(0.0, 1.0, n_out, endpoint=False)
    src = np.linspace(0.0, 1.0, x.size, endpoint=False)
    return np.interp(t, src, x).astype(np.float32)


def save_wav_mono(path: Path, wav: np.ndarray, sr: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    try:
        import soundfile as sf

        sf.write(str(path), x, int(sr))
        return
    except Exception:
        pass
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
