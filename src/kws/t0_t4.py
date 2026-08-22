"""T0–T4 track pick. T4 stays on CER oracle even if a cosine sidecar is present."""

from __future__ import annotations

from typing import Any

from .arms import CER_ORACLE_ARMS, L2_ARMS, se_mode
from .need_se import need_se
from .select_l2 import select_l1_l2
from .sidecar import SidecarError


def t0_stream(rec: dict[str, Any]) -> str:
    t0 = rec.get("oracle_stream") or (rec.get("recomputed_oracle") or {}).get("oracle_stream")
    if not t0:
        raise SidecarError(f"uid={rec.get('uid')}: missing oracle_stream")
    return str(t0)


def pick_track(
    arm: str,
    rec: dict[str, Any],
    *,
    cos_map: dict[str, dict[str, float]],
    pm_map: dict[str, dict[str, float] | float],
) -> dict[str, Any]:
    uid = str(rec["uid"])
    streams = rec.get("streams") or {}
    t0 = t0_stream(rec)
    dual = bool(rec.get("dual_zero"))
    if arm in CER_ORACLE_ARMS:
        return {
            "chosen": t0,
            "reason": "cer_oracle" if arm != "T4" else "t4_cer_oracle",
            "dual_zero": dual,
            "reverted_catastrophe": False,
            "l2_degraded": False,
        }
    if arm not in L2_ARMS:
        raise ValueError(arm)
    cos = cos_map.get(uid)
    if not cos:
        if cos_map:
            raise SidecarError(f"uid={uid} missing from cos sidecar")
        return {
            "chosen": t0,
            "reason": "l2_degraded_no_cos_sidecar",
            "dual_zero": dual,
            "reverted_catastrophe": False,
            "l2_degraded": True,
        }
    pm_raw = pm_map.get(uid)
    pm_per_stream = pm_raw if isinstance(pm_raw, dict) else None
    sel = select_l1_l2(streams, cos_to_raw=cos, p_music=pm_per_stream)
    return {
        "chosen": sel.chosen,
        "reason": sel.reason,
        "dual_zero": sel.dual_zero,
        "reverted_catastrophe": sel.reverted_catastrophe,
        "l2_degraded": False,
    }


def apply_se_placeholder(
    chosen: str,
    *,
    arm: str,
    p_music: float | None,
    snr: float | None,
) -> dict[str, Any]:
    mode = se_mode(arm)
    if mode == "none":
        return {"se_applied": False, "would_apply": False, "reason": f"{arm.lower()}_no_se"}
    if mode == "always":
        return {
            "se_applied": False,
            "would_apply": True,
            "reason": "t4_global_se_backend_missing",
            "safety": "must_check_cos_se_pre_and_presence",
        }
    trig = need_se(
        winner_is_original=(chosen == "original"),
        p_music=p_music,
        snr_med_db=snr,
    )
    return {
        "se_applied": False,
        "would_apply": trig.need,
        "need_se_reason": trig.reason,
        "reason": "conditional_se_backend_missing" if trig.need else "need_se_false",
    }
