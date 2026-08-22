"""Oracle track pick: argmin CER, original wins ties.

This is the current production selector. L2 may override it when CER
cannot distinguish tracks (dual-zero). Never uses MMS-FA.
"""

from __future__ import annotations

from typing import Any, Mapping


def is_sep_stream(name: str) -> bool:
    return str(name) != "original"


def stream_cer(rec: Mapping[str, Any]) -> float | None:
    if rec is None:
        return None
    v = rec.get("cer")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def oracle_of(
    cers: Mapping[str, Mapping[str, Any]],
    *,
    prefer_original: bool = True,
) -> tuple[str, float]:
    """Pick min CER. Ties: original first, then name.

    `prefer_original=True` matches the existing VM `oracle_of`. That
    tie-break is the structural CER blind spot on dual-zero items.
    """

    def key(k: str) -> tuple[float, int, str]:
        cer = float(cers[k]["cer"])
        orig_rank = 0 if (prefer_original and k == "original") else 1
        return cer, orig_rank, k

    name = min(cers, key=key)
    return name, float(cers[name]["cer"])


def pack_streams(cers: Mapping[str, Mapping[str, Any]]) -> dict[str, dict]:
    return {
        k: {
            "hyp": v.get("hyp"),
            "cer": round(float(v["cer"]), 4),
            "cer_char": round(float(v.get("cer_char", v["cer"])), 4),
            "cer_py": round(float(v.get("cer_py", v["cer"])), 4),
        }
        for k, v in cers.items()
    }


def min_sep_cer(streams: Mapping[str, Mapping[str, Any]]) -> float | None:
    vals = [stream_cer(v) for k, v in streams.items() if is_sep_stream(k)]
    vals = [x for x in vals if x is not None]
    return min(vals) if vals else None
