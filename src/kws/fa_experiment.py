"""Experimental text-alignment route and safe-crop policy.

This module is deliberately separate from the frozen T0--T4 selector.  It
consumes model-neutral sidecars so MMS-FA and Qwen3-ForcedAligner can be run in
their own pinned GPU environments without becoming runtime dependencies here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .iojson import load_jsonl
from .select_l2 import l1_eligible
from .sidecar import SidecarError


@dataclass(frozen=True)
class AlignmentEvidence:
    coverage: float
    start_sec: float
    end_sec: float
    duration_sec: float
    mean_logp: float | None = None
    p10_logp: float | None = None
    star_fraction: float = 0.0
    edge_clipped: bool = False


@dataclass(frozen=True)
class RouteResult:
    chosen: str
    reason: str
    qkw_winner: str
    fa_winner: str
    agreed: bool
    eligible: tuple[str, ...]


@dataclass(frozen=True)
class CropPlan:
    apply: bool
    start_sec: float
    end_sec: float
    reason: str


def _finite(raw: Any, *, uid: str, stream: str, key: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SidecarError(f"uid={uid}: alignment[{stream}].{key} is not a float") from exc
    if not math.isfinite(value):
        raise SidecarError(f"uid={uid}: alignment[{stream}].{key} is not finite")
    return value


def parse_alignment_stream(raw: Mapping[str, Any], *, uid: str, stream: str) -> AlignmentEvidence:
    required = ("coverage", "start_sec", "end_sec", "duration_sec")
    missing = [key for key in required if key not in raw]
    if missing:
        raise SidecarError(f"uid={uid}: alignment[{stream}] missing {missing}")
    coverage = _finite(raw["coverage"], uid=uid, stream=stream, key="coverage")
    start = _finite(raw["start_sec"], uid=uid, stream=stream, key="start_sec")
    end = _finite(raw["end_sec"], uid=uid, stream=stream, key="end_sec")
    duration = _finite(raw["duration_sec"], uid=uid, stream=stream, key="duration_sec")
    star = _finite(raw.get("star_fraction", 0.0), uid=uid, stream=stream, key="star_fraction")
    if not 0.0 <= coverage <= 1.0:
        raise SidecarError(f"uid={uid}: alignment[{stream}].coverage={coverage} outside [0,1]")
    if not 0.0 <= star <= 1.0:
        raise SidecarError(f"uid={uid}: alignment[{stream}].star_fraction={star} outside [0,1]")
    if duration <= 0.0 or start < 0.0 or end <= start or end > duration + 1e-3:
        raise SidecarError(
            f"uid={uid}: invalid alignment span {start:.4f}..{end:.4f} / {duration:.4f}s for {stream}"
        )

    def optional_score(key: str) -> float | None:
        return None if raw.get(key) is None else _finite(raw[key], uid=uid, stream=stream, key=key)

    return AlignmentEvidence(
        coverage=coverage,
        start_sec=start,
        end_sec=end,
        duration_sec=duration,
        mean_logp=optional_score("mean_logp"),
        p10_logp=optional_score("p10_logp"),
        star_fraction=star,
        edge_clipped=bool(raw.get("edge_clipped", False)),
    )


def load_alignment_sidecar(path: Path) -> tuple[dict[str, dict[str, AlignmentEvidence]], dict[str, str]]:
    scores: dict[str, dict[str, AlignmentEvidence]] = {}
    models: dict[str, str] = {}
    for index, row in enumerate(load_jsonl(path)):
        uid = str(row.get("uid") or "").strip()
        if not uid:
            raise SidecarError(f"row {index}: missing uid")
        if uid in scores:
            raise SidecarError(f"duplicate uid={uid} in {path}")
        model = str(row.get("model") or "").strip()
        if not model:
            raise SidecarError(f"uid={uid}: missing alignment model")
        payload = row.get("streams")
        if not isinstance(payload, dict) or not payload:
            raise SidecarError(f"uid={uid}: streams must be a non-empty dict")
        parsed: dict[str, AlignmentEvidence] = {}
        for stream, raw in payload.items():
            if not isinstance(raw, dict):
                raise SidecarError(f"uid={uid}: alignment[{stream}] must be an object")
            parsed[str(stream)] = parse_alignment_stream(raw, uid=uid, stream=str(stream))
        scores[uid] = parsed
        models[uid] = model
    return scores, models


def _fa_valid(ev: AlignmentEvidence, *, min_coverage: float, max_star_fraction: float) -> bool:
    return (
        ev.coverage >= min_coverage
        and ev.star_fraction <= max_star_fraction
        and ev.mean_logp is not None
        and ev.p10_logp is not None
    )


def fa_winner(
    eligible: list[str],
    evidence: Mapping[str, AlignmentEvidence],
    *,
    min_coverage: float = 1.0,
    max_star_fraction: float = 0.25,
) -> str | None:
    missing = [name for name in eligible if name not in evidence]
    if missing:
        raise SidecarError(f"alignment sidecar missing eligible streams {missing}")
    valid = [name for name in eligible if _fa_valid(evidence[name], min_coverage=min_coverage, max_star_fraction=max_star_fraction)]
    if not valid:
        return None
    return max(
        valid,
        key=lambda name: (
            float(evidence[name].p10_logp),
            float(evidence[name].mean_logp),
            evidence[name].coverage,
            -evidence[name].star_fraction,
            1 if name == "original" else 0,
            name,
        ),
    )


def route_by_agreement(
    streams: Mapping[str, Mapping[str, Any]],
    *,
    t0: str,
    qkw: Mapping[str, float],
    evidence: Mapping[str, AlignmentEvidence],
    min_coverage: float = 1.0,
    max_star_fraction: float = 0.25,
) -> RouteResult:
    eligible, _ = l1_eligible(streams)
    missing_q = [name for name in eligible if name not in qkw]
    if missing_q:
        raise SidecarError(f"q_kw missing eligible streams {missing_q}")
    q_winner = max(eligible, key=lambda name: (float(qkw[name]), 1 if name == "original" else 0, name))
    f_winner = fa_winner(
        eligible,
        evidence,
        min_coverage=min_coverage,
        max_star_fraction=max_star_fraction,
    )
    if f_winner is None:
        return RouteResult(t0, "fa_no_valid_evidence_fallback_t0", q_winner, "", False, tuple(eligible))
    if q_winner != f_winner:
        return RouteResult(t0, "qkw_fa_disagree_fallback_t0", q_winner, f_winner, False, tuple(eligible))
    return RouteResult(q_winner, "qkw_fa_agree", q_winner, f_winner, True, tuple(eligible))


def safe_crop_plan(
    evidence: AlignmentEvidence,
    *,
    margin_sec: float = 0.24,
    min_output_sec: float = 1.50,
    min_coverage: float = 1.0,
    max_star_fraction: float = 0.25,
    reject_edge_clipped: bool = True,
) -> CropPlan:
    duration = evidence.duration_sec
    if evidence.coverage < min_coverage or evidence.star_fraction > max_star_fraction:
        return CropPlan(False, 0.0, duration, "incomplete_alignment_fallback_full")
    if reject_edge_clipped and evidence.edge_clipped:
        return CropPlan(False, 0.0, duration, "edge_clipped_fallback_full")
    if duration <= min_output_sec:
        return CropPlan(False, 0.0, duration, "source_not_longer_than_min_output")

    start = max(0.0, evidence.start_sec - margin_sec)
    end = min(duration, evidence.end_sec + margin_sec)
    if end - start < min_output_sec:
        need = min_output_sec - (end - start)
        add_left = min(start, need / 2.0)
        start -= add_left
        need -= add_left
        add_right = min(duration - end, need)
        end += add_right
        need -= add_right
        start = max(0.0, start - need)
    if end - start + 1e-6 < min_output_sec:
        return CropPlan(False, 0.0, duration, "cannot_meet_min_output_fallback_full")
    if start <= 1e-3 and end >= duration - 1e-3:
        return CropPlan(False, 0.0, duration, "crop_equals_full")
    return CropPlan(True, start, end, "safe_alignment_crop")
