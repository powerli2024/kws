"""Frozen Presence protocol: the only enroll-adoption veto.

Online test-time code must not require CMD labels, e_cmd_crop, or hard-neg
speaker identity. This module only defines the eval contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

LOCKED_THR_BY_LANG = {"zh": 0.29305, "en": 0.357868, "default": 0.29305}
CER_MEAN_MAX = 0.03
CER0_DROP_MAX = 0.02
GO_CONTEST_DELTA = 0.005  # contest lock; enroll-only eval uses FRR/FAR first


@dataclass(frozen=True)
class PresenceVeto:
    accept: bool
    reason: str
    metrics: dict[str, Any]


def contest_score(rr: float, cer_total: float) -> float:
    return 0.5 * rr + 0.5 * (1.0 - cer_total)


def enroll_go(
    *,
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    cer_mean: float,
    cer0_rate: float,
    cer0_rate_baseline: float,
) -> PresenceVeto:
    """Adopt enroll change only if downstream Presence improves and CER holds.

    Required keys on baseline/candidate: frr, far (or rr = 1-far).
    FAR here is contest false-accept on neg = 1 - RR_reject.
    """
    frr0, frr1 = float(baseline["frr"]), float(candidate["frr"])

    def _far(row: Mapping[str, float]) -> float:
        if "far" in row:
            return float(row["far"])
        if "rr" in row:
            return 1.0 - float(row["rr"])
        raise KeyError("need far or rr")

    far0, far1 = _far(baseline), _far(candidate)
    m: dict[str, Any] = {
        "frr0": frr0,
        "frr1": frr1,
        "far0": far0,
        "far1": far1,
        "cer_mean": cer_mean,
        "cer0_rate": cer0_rate,
        "cer0_rate_baseline": cer0_rate_baseline,
    }
    if cer_mean > CER_MEAN_MAX:
        return PresenceVeto(False, "cer_mean_over_0.03", m)
    if cer0_rate_baseline - cer0_rate > CER0_DROP_MAX + 1e-12:
        return PresenceVeto(False, "cer0_rate_drop_over_2pp", m)
    frr_better = frr1 < frr0 - 1e-12
    far_better = far1 < far0 - 1e-12
    frr_worse = frr1 > frr0 + 1e-12
    far_worse = far1 > far0 + 1e-12
    if (frr_better or far_better) and not (frr_worse or far_worse):
        return PresenceVeto(True, "presence_improved_other_not_worse", m)
    if frr_better and far_worse:
        return PresenceVeto(False, "frr_down_but_far_up", m)
    if far_better and frr_worse:
        return PresenceVeto(False, "far_down_but_frr_up", m)
    return PresenceVeto(False, "no_downstream_move", m)
