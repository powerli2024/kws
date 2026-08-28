"""Selection and evaluation primitives for the independent s1 -> s7 -> SE route."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """L1 CER, L2 target NLL, then conservative deterministic preferences."""
    cer = _finite(row.get("cer"))
    nll = _finite(row.get("nll"))
    return (
        float("inf") if cer is None else cer,
        float("inf") if nll is None else nll,
        0 if row.get("view") == "raw" else 1,
        0 if row.get("role") == "s1" else 1,
        0 if row.get("stream") == "original" else 1,
        str(row.get("arm") or ""),
        str(row.get("stream") or ""),
    )


def choose(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    valid = [dict(row) for row in candidates if _finite(row.get("cer")) is not None]
    return min(valid, key=candidate_key) if valid else None


def improves(new: Mapping[str, Any], old: Mapping[str, Any], *, eps: float = 1e-9) -> bool:
    new_cer, old_cer = float(new["cer"]), float(old["cer"])
    if new_cer < old_cer - eps:
        return True
    if abs(new_cer - old_cer) > eps:
        return False
    new_nll, old_nll = _finite(new.get("nll")), _finite(old.get("nll"))
    return new_nll is not None and old_nll is not None and new_nll < old_nll - eps


def route_one(
    candidates: Iterable[Mapping[str, Any]],
    *,
    allow_se: bool,
    trigger_cer: float = 0.0,
) -> dict[str, Any]:
    rows = [dict(row) for row in candidates]

    def eligible(role: str) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            if row.get("role") != role:
                continue
            if row.get("view") == "se" and (not allow_se or not row.get("se_eligible")):
                continue
            if not allow_se and row.get("view") != "raw":
                continue
            out.append(row)
        return out

    first = choose(eligible("s1"))
    if first is None:
        return {"ok": False, "error": "missing_s1_candidate"}
    trigger = float(first["cer"]) > float(trigger_cer) + 1e-9
    second = choose(eligible("s7")) if trigger else None
    switched = bool(second is not None and improves(second, first))
    selected = second if switched else first
    return {
        "ok": True,
        "selected": selected,
        "s1_selected": first,
        "s7_selected": second,
        "triggered_s7": trigger,
        "switched_s7": switched,
        "reason": "s7_strict_improvement" if switched else ("s7_no_improvement" if second else "keep_s1"),
    }


def summarize(decisions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(decisions)
    good = [row for row in rows if row.get("ok") and isinstance(row.get("selected"), Mapping)]
    cers = [float(row["selected"]["cer"]) for row in good]
    counts = Counter(
        f"{row['selected'].get('role')}:{row['selected'].get('view')}" for row in good
    )
    return {
        "n": len(rows),
        "n_ok": len(good),
        "n_missing": len(rows) - len(good),
        "mean_cer": round(sum(cers) / len(cers), 8) if cers else None,
        "cer0": sum(value <= 1e-9 for value in cers),
        "cer0_rate": round(sum(value <= 1e-9 for value in cers) / len(cers), 8) if cers else None,
        "n_triggered_s7": sum(bool(row.get("triggered_s7")) for row in good),
        "n_switched_s7": sum(bool(row.get("switched_s7")) for row in good),
        "selected_counts": dict(sorted(counts.items())),
    }


def paired(base: Iterable[Mapping[str, Any]], trial: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    left = {str(row["uid"]): row for row in base if row.get("ok")}
    right = {str(row["uid"]): row for row in trial if row.get("ok")}
    common = sorted(set(left) & set(right))
    delta = [float(right[uid]["selected"]["cer"]) - float(left[uid]["selected"]["cer"]) for uid in common]
    return {
        "n_common": len(common),
        "mean_delta": round(sum(delta) / len(delta), 8) if delta else None,
        "n_improved": sum(value < -1e-9 for value in delta),
        "n_worsened": sum(value > 1e-9 for value in delta),
        "n_same": sum(abs(value) <= 1e-9 for value in delta),
    }
