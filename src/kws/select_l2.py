"""L1 text constraint + L2 speaker/residual pick.

L1: drop tracks with CER > min_cer + slack (default 0.05).
L2: score = cos(track, raw) - lambda * p_music
    On dual-zero, sep is allowed unless cos(spk, raw) < catastrophe_floor.

Does not call MMS-FA. Does not use CER as the unique argmin when slack ties exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .oracle import is_sep_stream, oracle_of, stream_cer

CER_SLACK_DEFAULT = 0.05
LAMBDA_GRID = (0.0, 0.05, 0.10)
CATASTROPHE_COS_GRID = tuple(round(0.90 + i * 0.01, 2) for i in range(6))
DEFAULT_CATASTROPHE_COS = 0.90
DEFAULT_LAMBDA = 0.0


@dataclass(frozen=True)
class SelectResult:
    chosen: str
    reason: str
    min_cer: float
    eligible: tuple[str, ...]
    scores: dict[str, float]
    dual_zero: bool
    reverted_catastrophe: bool


def _cers(streams: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, rec in streams.items():
        c = stream_cer(rec)
        if c is not None:
            out[k] = c
    return out


def l1_eligible(
    streams: Mapping[str, Mapping[str, Any]],
    *,
    slack: float = CER_SLACK_DEFAULT,
) -> tuple[list[str], float]:
    cers = _cers(streams)
    if not cers:
        raise ValueError("no CER scores")
    min_cer = min(cers.values())
    names = [k for k, v in cers.items() if v <= min_cer + slack]
    names.sort(key=lambda n: (0 if n == "original" else 1, n))
    return names, min_cer


def select_l1_l2(
    streams: Mapping[str, Mapping[str, Any]],
    *,
    cos_to_raw: Mapping[str, float] | None = None,
    p_music: Mapping[str, float] | None = None,
    lam: float = DEFAULT_LAMBDA,
    slack: float = CER_SLACK_DEFAULT,
    catastrophe_cos: float = DEFAULT_CATASTROPHE_COS,
    fallback: str | None = None,
) -> SelectResult:
    eligible, min_cer = l1_eligible(streams, slack=slack)
    cers = _cers(streams)
    orig0 = cers.get("original", 1.0) <= 1e-9
    any_sep0 = any(is_sep_stream(n) and cers.get(n, 1.0) <= 1e-9 for n in cers)
    dual_zero = orig0 and any_sep0

    if cos_to_raw is None:
        # No speaker scores: keep VM oracle (original wins ties).
        packed = {k: {"cer": cers[k]} for k in eligible}
        chosen, _ = oracle_of(packed, prefer_original=True)
        return SelectResult(
            chosen=chosen,
            reason="l1_oracle_cer_no_cos",
            min_cer=min_cer,
            eligible=tuple(eligible),
            scores={k: -cers[k] for k in eligible},
            dual_zero=dual_zero,
            reverted_catastrophe=False,
        )

    p_music = p_music or {}
    scores: dict[str, float] = {}
    for name in eligible:
        cos = float(cos_to_raw.get(name, 0.0))
        pm = float(p_music.get(name, 0.0))
        scores[name] = cos - lam * pm

    # Higher score wins; exact ties keep original (conservative).
    chosen = max(scores, key=lambda n: (scores[n], 1 if n == "original" else 0, n))
    reverted = False
    reason = "l2_max_cos_minus_pmusic"

    if is_sep_stream(chosen) and dual_zero:
        if float(cos_to_raw.get(chosen, 0.0)) < catastrophe_cos:
            chosen = "original" if "original" in eligible else chosen
            reverted = True
            reason = "dual_zero_sep_catastrophe_revert_original"

    if fallback and chosen not in streams:
        chosen = fallback
        reason = "missing_chosen_fallback"

    return SelectResult(
        chosen=chosen,
        reason=reason,
        min_cer=min_cer,
        eligible=tuple(eligible),
        scores=scores,
        dual_zero=dual_zero,
        reverted_catastrophe=reverted,
    )
