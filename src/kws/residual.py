"""Lightweight residual proxies. Neural p_music / DNSMOS are optional backends.

Heuristic SNR is only a trigger candidate, never a speaker-quality score.
"""

from __future__ import annotations

import numpy as np


def snr_med_db(wav: np.ndarray, sr: int = 16000, frame_sec: float = 0.02) -> float:
    """Energy-percentile SNR: p80 / p20 of frame RMS, in dB."""
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    n = max(1, int(round(frame_sec * sr)))
    if x.size < n * 4:
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        return 20.0 * np.log10(rms / 1e-3)
    n_frames = x.size // n
    frames = x[: n_frames * n].reshape(n_frames, n)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    p20 = float(np.percentile(rms, 20))
    p80 = float(np.percentile(rms, 80))
    return float(20.0 * np.log10((p80 + 1e-12) / (p20 + 1e-12)))


def p_music_heuristic(wav: np.ndarray, sr: int = 16000) -> float:
    """Cheap music-ish score in [0, 1]: tonal / noisy via spectral flatness inverse.

    Direction-B placeholder until YAMNet/PANNs are wired. Do not treat as
    ground truth. Calibrate against listen labels before using as a gate.
    """
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size < 512:
        return 0.0
    n = 1 << int(np.floor(np.log2(min(x.size, 4096))))
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
    spec = np.maximum(spec, 1e-12)
    log_mean = float(np.mean(np.log(spec)))
    gmean = float(np.exp(log_mean))
    amean = float(np.mean(spec))
    flat = gmean / amean
    # lower flatness → more tonal. Map to (0,1).
    return float(np.clip(1.0 - flat * 4.0, 0.0, 1.0))
