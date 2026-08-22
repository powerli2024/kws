"""Dual-zero CER: original and at least one sep track both CER==0.

This is the missing data support for 'skip BSS' vs 'CER cannot pick a track'.
"""

from __future__ import annotations

from typing import Any, Mapping

from .oracle import is_sep_stream, stream_cer

CER_EPS = 1e-9


def classify_streams(
    streams: Mapping[str, Mapping[str, Any]] | None,
    *,
    eps: float = CER_EPS,
) -> dict[str, Any]:
    streams = streams or {}
    orig = stream_cer(streams.get("original") or {})
    sep: dict[str, float] = {}
    for name, rec in streams.items():
        if not is_sep_stream(name):
            continue
        c = stream_cer(rec)
        if c is not None:
            sep[name] = c

    orig0 = orig is not None and orig <= eps
    sep0 = {k: v for k, v in sep.items() if v <= eps}
    any_sep0 = bool(sep0)
    all_sep_gt0 = bool(sep) and all(v > eps for v in sep.values())
    return {
        "original_cer": orig,
        "n_sep": len(sep),
        "min_sep_cer": min(sep.values()) if sep else None,
        "orig_cer0": orig0,
        "any_sep_cer0": any_sep0,
        "sep_cer0_names": sorted(sep0),
        "dual_zero": orig0 and any_sep0,
        "orig_unique_zero": orig0 and all_sep_gt0,
        "all_collapsed": (orig is not None and orig > eps) and (not any_sep0) and bool(sep),
        "missing_original": orig is None,
    }


def skip_sep_feasibility(cls: Mapping[str, Any]) -> str:
    """Post-hoc label for whether BSS output should be discarded.

    Not a before-BSS trigger. Duration is never used.
    """
    if cls.get("orig_unique_zero"):
        return "skip_sep_text_safe"
    if cls.get("dual_zero"):
        return "need_l2_not_cer"
    if cls.get("orig_cer0"):
        return "orig_cer0_no_sep_scores"
    return "do_not_skip"
