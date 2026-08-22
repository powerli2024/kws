"""Conditional SE trigger and safety gate.

Denoise ≠ better speaker embedding. BAK/p_music only trigger; speaker
effect must be checked with cos(se, pre) and frozen Presence.
"""

from __future__ import annotations

from dataclasses import dataclass

P_MUSIC_ORIG_GRID = (0.30, 0.40, 0.50)
P_MUSIC_SEP_GRID = (0.25, 0.35, 0.45)
SNR_ORIG_GRID = (3.0, 5.0, 8.0)
SNR_SEP_GRID = (5.0, 8.0, 12.0)
SE_COS_GRID = tuple(round(0.90 + i * 0.01, 2) for i in range(6))
CER_SLACK = 0.05
TARGET_TRIGGER_RATE = (0.15, 0.30)


@dataclass(frozen=True)
class NeedSe:
    need: bool
    reason: str
    p_music: float | None
    snr_med_db: float | None


def need_se(
    *,
    winner_is_original: bool,
    p_music: float | None,
    snr_med_db: float | None,
    p_music_orig: float = 0.40,
    p_music_sep: float = 0.35,
    snr_orig: float = 5.0,
    snr_sep: float = 8.0,
) -> NeedSe:
    pm, snr = p_music, snr_med_db
    if pm is None and snr is None:
        return NeedSe(False, "no_residual_scores", pm, snr)
    thr_pm = p_music_orig if winner_is_original else p_music_sep
    thr_snr = snr_orig if winner_is_original else snr_sep
    flags: list[str] = []
    if pm is not None and pm > thr_pm:
        flags.append("p_music")
    if snr is not None and snr < thr_snr:
        flags.append("snr")
    return NeedSe(bool(flags), "+".join(flags) if flags else "clean", pm, snr)


def se_safety_ok(
    *,
    cos_se_pre: float,
    cer_se: float,
    cer_pre: float,
    cos_thr: float = 0.92,
    cer_slack: float = CER_SLACK,
) -> tuple[bool, str]:
    if cos_se_pre < cos_thr:
        return False, "cos_collapse"
    if cer_se > cer_pre + cer_slack:
        return False, "cer_regression"
    return True, "ok"
