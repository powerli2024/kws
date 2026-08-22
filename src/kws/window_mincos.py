"""Same-file window min-cosine for short residual / other-speaker bursts.

Skip if KWS duration < 0.8 s (unstable). Need ≥2 windows of 0.6 s.
Threshold is a percentile of the corpus min-cos distribution, locked with
listen labels when available — not a guessed absolute cosine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_DUR_SEC = 0.8
WIN_SEC = 0.6
HOP_SEC = 0.3
PERCENTILE_GRID = (5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class WindowSpec:
    starts: tuple[int, ...]
    win: int
    skipped: bool
    reason: str


def window_starts(
    n: int,
    sr: int = 16000,
    *,
    min_dur_sec: float = MIN_DUR_SEC,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
) -> WindowSpec:
    dur = n / float(sr) if sr else 0.0
    win = int(round(win_sec * sr))
    hop = max(1, int(round(hop_sec * sr)))
    if dur + 1e-9 < min_dur_sec:
        return WindowSpec((), win, True, "dur_lt_0p8s")
    if n < win:
        return WindowSpec((), win, True, "shorter_than_window")
    starts = list(range(0, n - win + 1, hop))
    last = n - win
    if not starts or starts[-1] != last:
        starts.append(last)
    # unique, ordered
    uniq: list[int] = []
    for s in starts:
        if s not in uniq:
            uniq.append(s)
    if len(uniq) < 2:
        uniq = [0, last] if last > 0 else []
    if len(uniq) < 2:
        return WindowSpec(tuple(uniq), win, True, "need_two_windows")
    return WindowSpec(tuple(uniq), win, False, "ok")


def slice_windows(wav: np.ndarray, spec: WindowSpec) -> list[np.ndarray]:
    if spec.skipped:
        return []
    x = np.asarray(wav).reshape(-1)
    return [x[s : s + spec.win].copy() for s in spec.starts]


def pairwise_min_cos(embs: list[np.ndarray]) -> float | None:
    if len(embs) < 2:
        return None
    mats = []
    for e in embs:
        v = np.asarray(e, dtype=np.float64).reshape(-1)
        n = np.linalg.norm(v)
        if n < 1e-12:
            return None
        mats.append(v / n)
    m = np.stack(mats, axis=0)
    g = m @ m.T
    iu = np.triu_indices(len(m), k=1)
    return float(np.min(g[iu]))


def anomaly_by_percentile(
    min_cos: float,
    corpus_min_cos: list[float],
    *,
    percentile: float = 10.0,
) -> bool:
    if not corpus_min_cos:
        raise ValueError("empty corpus; cannot set percentile gate")
    thr = float(np.percentile(np.asarray(corpus_min_cos, dtype=np.float64), percentile))
    return float(min_cos) < thr
