"""Landable SE backends. Neural SE stays optional; spectral is numpy-only.

Denoise ≠ better speaker. Export must still check cos(se, pre) before keeping SE.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .need_se import se_safety_ok


def spectral_subtract(
    wav: np.ndarray,
    sr: int = 16000,
    *,
    n_fft: int = 512,
    hop: int = 128,
    noise_pct: float = 0.15,
    floor: float = 0.12,
) -> np.ndarray:
    """Light Wiener-like gate from quietest STFT frames. Conservative floor."""
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if x.size < n_fft:
        return x.copy()
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (x.size - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * win for i in range(n_frames)], axis=1)
    spec = np.fft.rfft(frames, axis=0)
    mag = np.abs(spec)
    energy = np.mean(mag * mag, axis=0)
    k = max(1, int(round(n_frames * noise_pct)))
    quiet = np.argpartition(energy, k - 1)[:k]
    noise = np.median(mag[:, quiet], axis=1, keepdims=True)
    gain = np.maximum(1.0 - (noise / np.maximum(mag, 1e-8)), floor)
    cleaned = np.fft.irfft(spec * gain, n=n_fft, axis=0)
    out = np.zeros(n_fft + hop * (n_frames - 1), dtype=np.float32)
    wsum = np.zeros_like(out)
    for i in range(n_frames):
        sl = slice(i * hop, i * hop + n_fft)
        out[sl] += cleaned[:, i].real.astype(np.float32) * win
        wsum[sl] += win * win
    out = out / np.maximum(wsum, 1e-8)
    if out.size < x.size:
        pad = np.zeros(x.size, dtype=np.float32)
        pad[: out.size] = out
        out = pad
    peak = float(np.max(np.abs(out))) + 1e-9
    in_peak = float(np.max(np.abs(x))) + 1e-9
    return (out[: x.size] * (in_peak / peak)).astype(np.float32)


def apply_se(
    wav: np.ndarray,
    sr: int,
    *,
    backend: str,
    cos_se_pre: float | None = None,
    cer_se: float | None = None,
    cer_pre: float | None = None,
) -> dict[str, Any]:
    backend = (backend or "none").lower().strip()
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if backend in ("", "none", "off"):
        return {"se_applied": False, "wav": x, "reason": "backend_none"}
    if backend not in ("spectral", "spectral_subtract"):
        return {"se_applied": False, "wav": x, "reason": f"unknown_backend_{backend}"}
    y = spectral_subtract(x, sr)
    if cos_se_pre is not None and cer_se is not None and cer_pre is not None:
        ok, why = se_safety_ok(cos_se_pre=cos_se_pre, cer_se=cer_se, cer_pre=cer_pre)
        if not ok:
            return {"se_applied": False, "wav": x, "reason": f"se_safety_{why}", "would_apply": True}
    return {"se_applied": True, "wav": y, "reason": "spectral_subtract", "backend": "spectral"}
