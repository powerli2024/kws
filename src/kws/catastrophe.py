"""cos(e*, e_raw) as catastrophe detection, not as 'purer speaker'.

Search range 0.90–0.95. Calibrate from data; do not treat a rise as success.
"""

from __future__ import annotations

from dataclasses import dataclass

CATASTROPHE_COS_GRID = tuple(round(0.90 + i * 0.01, 2) for i in range(6))


@dataclass(frozen=True)
class CatastropheHit:
    cos_to_raw: float
    threshold: float
    is_catastrophe: bool


def is_catastrophe(cos_to_raw: float, threshold: float) -> bool:
    return float(cos_to_raw) < float(threshold)


def pick_threshold_from_clean_floor(
    clean_cos: list[float],
    *,
    quantile: float = 0.05,
    grid: tuple[float, ...] = CATASTROPHE_COS_GRID,
) -> float:
    """Set the floor near a low quantile of a 'should not fire' set.

    Typical clean set: original-winner items' cos(original, original)=1, so use
    sep-track vs raw on items where original uniquely has CER=0 (sep is a
    failed BSS, embeddings should drop). We want the gate to fire on those
    drops, and not on mild enhancement.

    Returns the grid value closest to the empirical quantile, clipped to grid.
    """
    if not clean_cos:
        from .config import runtime

        return float(runtime()["catastrophe_cos"])
    xs = sorted(float(x) for x in clean_cos)
    q = min(max(quantile, 0.0), 1.0)
    idx = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
    target = xs[idx]
    return min(grid, key=lambda t: abs(t - target))


def hit(cos_to_raw: float, threshold: float) -> CatastropheHit:
    c = float(cos_to_raw)
    t = float(threshold)
    return CatastropheHit(c, t, is_catastrophe(c, t))
