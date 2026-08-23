"""Small-n eval helpers. 474 neg ≈ 0.211 pp per error; always report intervals."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(p), float(max(0.0, center - half)), float(min(1.0, center + half))


def bootstrap_mean_ci(xs: Sequence[float], *, n_boot: int = 2000, seed: int = 0) -> dict[str, float | None]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = float(rng.choice(a, size=a.size, replace=True).mean())
    return {
        "mean": float(a.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": int(a.size),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar; b,c are discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb

    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
    p = min(1.0, 2.0 * tail / (2**n))
    return float(p)
