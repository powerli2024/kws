"""Skip-separation policy.

Hard rule: never skip BSS because the clip is short.
Old VB `should_skip_separation(dur<=1.8s)` raised enroll CER; do not revive it.

Two different questions:
  after scores: can we drop sep tracks and keep original?
  before BSS:   can we skip MossFormer entirely? only with a residual detector,
                after calibration. Default OFF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .dual_zero import classify_streams, skip_sep_feasibility


@dataclass(frozen=True)
class SkipSepDecision:
    skip: bool
    reason: str
    cls: dict[str, Any]


def skip_sep_after_scores(
    streams: Mapping[str, Mapping[str, Any]] | None,
    *,
    allow_unique_zero_only: bool = True,
) -> SkipSepDecision:
    """Use after ONNX has already scored original/spk1/spk2.

    skip=True only when original uniquely has CER=0 (sep strictly worse on text).
    Dual-zero is NOT skip: that is the L2 selector's job.
    """
    cls = classify_streams(streams)
    tag = skip_sep_feasibility(cls)
    if allow_unique_zero_only and tag == "skip_sep_text_safe":
        return SkipSepDecision(True, tag, cls)
    return SkipSepDecision(False, tag, cls)


def skip_bss_before_sep(
    *,
    p_music: float | None,
    snr_med_db: float | None,
    dur_sec: float | None = None,
    p_music_max: float = 0.15,
    snr_min_db: float = 15.0,
    enabled: bool = False,
) -> SkipSepDecision:
    """Optional pre-BSS skip. Default disabled until residual models are calibrated.

    Duration is accepted as an input but never used as a trigger.
    """
    _ = dur_sec  # explicitly unused
    cls = {
        "p_music": p_music,
        "snr_med_db": snr_med_db,
        "enabled": enabled,
        "p_music_max": p_music_max,
        "snr_min_db": snr_min_db,
    }
    if not enabled:
        return SkipSepDecision(False, "pre_bss_disabled_until_calibrated", cls)
    if p_music is None or snr_med_db is None:
        return SkipSepDecision(False, "pre_bss_missing_residual", cls)
    if p_music <= p_music_max and snr_med_db >= snr_min_db:
        return SkipSepDecision(True, "pre_bss_residual_clean", cls)
    return SkipSepDecision(False, "pre_bss_residual_dirty", cls)
