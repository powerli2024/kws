"""Train/dev ranking: can this enroll separate pos CMD from neg CMD?

This is NOT the contest Presence veto. Higher cosine on pos CMD and lower on
neg CMD means a cleaner speaker enroll. Locked VE τ is reported as a probe.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .stats import wilson_interval

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
        "locked_tau_by_lang": None,
        "intervals": {
            "neg_n": len(n),
            "far_pp_per_error": (100.0 / len(n)) if n else None,
            "note": "474 neg ≈ 0.211 pp per miss; do not claim FAR < 0.2% without a CI",
        },
        "role": "kws_local_cmd_separation_not_contest_veto",
    }


def rate_with_wilson(k: int, n: int) -> dict[str, float]:
    p, lo, hi = wilson_interval(k, n)
    return {"rate": p, "lo": lo, "hi": hi, "k": k, "n": n}


def summarize_by_lang(
    rows: Iterable[Mapping[str, Any]],
    *,
    tau_by_lang: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    tau_by_lang = dict(tau_by_lang or LOCKED_TAU_PROBE)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in rows:
        lang = str(rec.get("lang") or "default")
        if lang not in ("zh", "en"):
            lang = "zh" if lang not in tau_by_lang else lang
        buckets.setdefault(lang, []).append(dict(rec))
    out: dict[str, Any] = {}
    for lang, items in buckets.items():
        pos, neg = collect_split_scores(items)
        base = summarize_cmd_scores(pos, neg, tau_probe={lang: float(tau_by_lang.get(lang, tau_by_lang["default"]))})
        thr = float(tau_by_lang.get(lang, tau_by_lang["default"]))
        pa = np.asarray(pos, dtype=np.float64)
        na = np.asarray(neg, dtype=np.float64)
        frr_k = int(np.sum(pa < thr)) if pa.size else 0
        far_k = int(np.sum(na >= thr)) if na.size else 0
        base["locked_tau_by_lang"] = {
            "lang": lang,
            "threshold": thr,
            "frr": rate_with_wilson(frr_k, len(pos)),
            "far": rate_with_wilson(far_k, len(neg)),
        }
        out[lang] = base
    return out


def paired_deltas(
    base_rows: Sequence[Mapping[str, Any]],
    cand_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    b = {str(r["uid"]): r for r in base_rows if r.get("cos_enroll_cmd") is not None}
    c = {str(r["uid"]): r for r in cand_rows if r.get("cos_enroll_cmd") is not None}
    common = sorted(set(b) & set(c))
    pos_up = pos_n = neg_down = neg_n = 0
    deltas: list[float] = []
    for uid in common:
        d = float(c[uid]["cos_enroll_cmd"]) - float(b[uid]["cos_enroll_cmd"])
        deltas.append(d)
        split = str(c[uid].get("split") or b[uid].get("split") or "")
        if split == "pos":
            pos_n += 1
            pos_up += int(d > 0)
        elif split == "neg":
            neg_n += 1
            neg_down += int(d < 0)
    return {
        "n_common": len(common),
        "pos_improved_rate": (pos_up / pos_n) if pos_n else None,
        "neg_improved_rate": (neg_down / neg_n) if neg_n else None,
        "delta_mean": float(np.mean(deltas)) if deltas else None,
    }


def rank_groups(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    baseline: str,
) -> dict[str, Any]:
    """Rank by locked-τ FAR then FRR (lower), then -pos_p10, then EER. Not EER-only."""

    def key(name: str) -> tuple[float, float, float, float]:
        s = summaries[name]
        probe = s.get("locked_tau_probe_not_adopt") or {}
        zh = probe.get("tau_zh") or {}
        far = zh.get("far")
        frr = zh.get("frr")
        p10 = s.get("pos_p10")
        eer = (s.get("eer") or {}).get("eer")
        return (
            float(far) if far is not None else 2.0,
            float(frr) if frr is not None else 2.0,
            -float(p10) if p10 is not None else 0.0,
            float(eer) if eer is not None else 2.0,
        )

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
