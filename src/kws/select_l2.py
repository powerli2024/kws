"""L0–L3 enroll pick. cos(track, raw) is a catastrophe gate, not a purity score.

L1 text: same-CER hard gate (4-char wake slack≈0). Rank by q_kw / token NLL
when a text sidecar exists. Heuristic p_music is not an official score.
Without q_kw, L2 degrades to CER oracle — do not interpret as a speaker result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .oracle import is_sep_stream, oracle_of, stream_cer
from .sidecar import SidecarError
from .config import runtime

_RT = runtime()
CER_SLACK_DEFAULT = float(_RT["l1_slack"])
LAMBDA_GRID = (0.0, 0.05, 0.10)
CATASTROPHE_COS_GRID = tuple(round(0.90 + i * 0.01, 2) for i in range(6))
DEFAULT_CATASTROPHE_COS = float(_RT["catastrophe_cos"])
DEFAULT_LAMBDA = float(_RT["lambda"])
REJECT_Q_HIGH = float(_RT.get("reject_q_high", 0.80))
REJECT_PAIR_COS = float(_RT.get("reject_pair_cos", 0.35))


@dataclass(frozen=True)
class SelectResult:
    chosen: str
    reason: str
    min_cer: float
    eligible: tuple[str, ...]
    scores: dict[str, float]
    dual_zero: bool
    reverted_catastrophe: bool
    rejected: bool = False
    l2_degraded: bool = False


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


def _companion_margin(scores: Mapping[str, float], name: str) -> float:
    others = [v for k, v in scores.items() if k != name]
    if not others:
        return 0.0
    return float(scores[name] - max(others))


def select_l1_l2(
    streams: Mapping[str, Mapping[str, Any]],
    *,
    cos_to_raw: Mapping[str, float] | None = None,
    q_kw: Mapping[str, float] | None = None,
    q_kw_kind: str = "q_kw",
    p_music: Mapping[str, float] | None = None,
    pair_cos: Mapping[str, float] | None = None,
    lam: float = DEFAULT_LAMBDA,
    slack: float = CER_SLACK_DEFAULT,
    catastrophe_cos: float = DEFAULT_CATASTROPHE_COS,
    fallback: str | None = None,
) -> SelectResult:
    """Pick a track. `p_music` is accepted but ignored unless λ≠0 (off by default).

    Official rank is q_kw (higher better). cos_to_raw only reverts a sep winner
    that collapses vs raw. λ≠0 is not a supported official path.
    """
    _ = lam  # official score does not use λ p_music
    eligible, min_cer = l1_eligible(streams, slack=slack)
    cers = _cers(streams)
    orig0 = cers.get("original", 1.0) <= 1e-9
    any_sep0 = any(is_sep_stream(n) and cers.get(n, 1.0) <= 1e-9 for n in cers)
    dual_zero = orig0 and any_sep0

    if not q_kw:
        packed = {k: {"cer": cers[k]} for k in eligible}
        chosen, _ = oracle_of(packed, prefer_original=True)
        return SelectResult(
            chosen=chosen,
            reason="l2_degraded_no_text_sidecar",
            min_cer=min_cer,
            eligible=tuple(eligible),
            scores={k: -cers[k] for k in eligible},
            dual_zero=dual_zero,
            reverted_catastrophe=False,
            l2_degraded=True,
        )

    missing_q = [n for n in eligible if n not in q_kw]
    if missing_q:
        raise SidecarError(f"q_kw missing streams {missing_q}")
    scores = {name: float(q_kw[name]) for name in eligible}

    # The absolute high-confidence threshold is meaningful only for calibrated
    # q_kw in [0, 1]. Raw NLL is usable for ranking after negation, not for this
    # registration-reject gate.
    high = (
        [n for n in eligible if is_sep_stream(n) and scores[n] >= REJECT_Q_HIGH]
        if q_kw_kind == "q_kw"
        else []
    )
    if len(high) >= 2 and pair_cos:
        worst = 1.0
        for i, a in enumerate(high):
            for b in high[i + 1 :]:
                key = f"{a}|{b}"
                alt = f"{b}|{a}"
                if key in pair_cos:
                    worst = min(worst, float(pair_cos[key]))
                elif alt in pair_cos:
                    worst = min(worst, float(pair_cos[alt]))
        if worst < REJECT_PAIR_COS:
            return SelectResult(
                chosen="reject",
                reason="reject_two_speakers_high_text",
                min_cer=min_cer,
                eligible=tuple(eligible),
                scores=scores,
                dual_zero=dual_zero,
                reverted_catastrophe=False,
                rejected=True,
            )

    chosen = max(scores, key=lambda n: (scores[n], _companion_margin(scores, n), 1 if n == "original" else 0, n))
    reason = "l2_max_qkw_margin"
    reverted = False

    if is_sep_stream(chosen) and cos_to_raw is not None:
        if chosen not in cos_to_raw:
            raise SidecarError(f"cos_to_raw missing stream {chosen!r} for catastrophe gate")
        if float(cos_to_raw[chosen]) < catastrophe_cos:
            chosen = "original" if "original" in eligible else chosen
            reverted = True
            reason = "sep_catastrophe_revert_original"

    if fallback and chosen not in streams and chosen != "reject":
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
        rejected=chosen == "reject",
    )
