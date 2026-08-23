"""Train/dev ranking: can this enroll separate pos CMD from neg CMD?

This is NOT the contest Presence veto. Higher cosine on pos CMD and lower on
neg CMD means a cleaner speaker enroll. Locked VE τ is reported as a probe.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

LOCKED_TAU_PROBE = {"zh": 0.29305, "en": 0.357868, "default": 0.29305}


def auc_scores(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """P(genuine > impostor) + 0.5 P(equal). Mann–Whitney."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    if p.size == 0 or n.size == 0:
        return None
    # Broadcast: (n_pos, n_neg)
    gt = np.sum(p[:, None] > n[None, :])
    eq = np.sum(p[:, None] == n[None, :])
    return float((gt + 0.5 * eq) / (p.size * n.size))


def _frr_far(pos: np.ndarray, neg: np.ndarray, thr: float) -> tuple[float, float]:
    frr = float(np.mean(pos < thr)) if pos.size else float("nan")
    far = float(np.mean(neg >= thr)) if neg.size else float("nan")
    return frr, far


def eer_and_threshold(pos: Sequence[float], neg: Sequence[float]) -> dict[str, float | None]:
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    if p.size == 0 or n.size == 0:
        return {"eer": None, "threshold": None, "frr": None, "far": None}
    cands = np.unique(np.concatenate([p, n]))
    best_gap = 1e9
    best: dict[str, float] = {"eer": 1.0, "threshold": float(cands[0]), "frr": 1.0, "far": 1.0}
    for t in cands:
        frr, far = _frr_far(p, n, float(t))
        gap = abs(frr - far)
        eer = 0.5 * (frr + far)
        if gap < best_gap - 1e-15 or (abs(gap - best_gap) <= 1e-15 and eer < best["eer"]):
            best_gap = gap
            best = {"eer": float(eer), "threshold": float(t), "frr": float(frr), "far": float(far)}
    return best


def at_threshold(pos: Sequence[float], neg: Sequence[float], thr: float) -> dict[str, float]:
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    frr, far = _frr_far(p, n, float(thr))
    return {"threshold": float(thr), "frr": frr, "far": far, "rr": 1.0 - far}


def summarize_cmd_scores(
    pos: Sequence[float],
    neg: Sequence[float],
    *,
    tau_probe: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    p = [float(x) for x in pos]
    n = [float(x) for x in neg]
    pa = np.asarray(p, dtype=np.float64) if p else np.asarray([], dtype=np.float64)
    na = np.asarray(n, dtype=np.float64) if n else np.asarray([], dtype=np.float64)
    eer = eer_and_threshold(p, n)
    tau_probe = dict(tau_probe or LOCKED_TAU_PROBE)
    probes = {f"tau_{k}": at_threshold(p, n, float(v)) for k, v in tau_probe.items()}
    return {
        "n_pos": len(p),
        "n_neg": len(n),
        "pos_mean": float(pa.mean()) if pa.size else None,
        "neg_mean": float(na.mean()) if na.size else None,
        "pos_p10": float(np.percentile(pa, 10)) if pa.size else None,
        "pos_p50": float(np.percentile(pa, 50)) if pa.size else None,
        "neg_p50": float(np.percentile(na, 50)) if na.size else None,
        "neg_p90": float(np.percentile(na, 90)) if na.size else None,
        "mean_gap": (float(pa.mean() - na.mean()) if pa.size and na.size else None),
        "auc": auc_scores(p, n),
        "eer": eer,
        "locked_tau_probe_not_adopt": probes,
        "role": "kws_local_cmd_separation_not_contest_veto",
    }


def rank_groups(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    baseline: str,
) -> dict[str, Any]:
    """Rank by EER (lower), then mean_gap (higher), then AUC (higher)."""

    def key(name: str) -> tuple[float, float, float]:
        s = summaries[name]
        eer = (s.get("eer") or {}).get("eer")
        gap = s.get("mean_gap")
        auc = s.get("auc")
        eer_k = float(eer) if eer is not None else 2.0
        gap_k = -float(gap) if gap is not None else 0.0
        auc_k = -float(auc) if auc is not None else 0.0
        return eer_k, gap_k, auc_k

    names = list(summaries)
    ordered = sorted(names, key=key)
    winner = ordered[0] if ordered else None
    note = (
        f"CMD-cosine rank (not Presence adopt): best={winner}. "
        "Use these best_sep dirs later on extract@main for the real contest veto."
    )
    return {
        "baseline": baseline,
        "order": ordered,
        "best": winner,
        "beats_baseline": bool(winner and winner != baseline and key(winner) < key(baseline)),
        "note": note,
    }


def collect_split_scores(rows: Iterable[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    pos: list[float] = []
    neg: list[float] = []
    for rec in rows:
        s = rec.get("cos_enroll_cmd")
        if s is None:
            continue
        split = str(rec.get("split") or "")
        if split == "pos":
            pos.append(float(s))
        elif split == "neg":
            neg.append(float(s))
    return pos, neg
