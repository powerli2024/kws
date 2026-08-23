"""Strict sidecar parsers. Empty/ambiguous payloads must error, not silently score 0."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .iojson import load_jsonl

COS_PAYLOAD_KEYS = ("cos_to_raw", "scores", "cos")
PM_PAYLOAD_KEYS = ("p_music", "pmusic")
QKW_PAYLOAD_KEYS = ("q_kw", "nll")
PAIR_COS_KEY = "pair_cos"


class SidecarError(ValueError):
    pass


def _require_uid(row: Mapping[str, Any], *, index: int) -> str:
    uid = row.get("uid")
    if uid is None or str(uid).strip() == "":
        raise SidecarError(f"row {index}: missing uid")
    return str(uid)


def _exactly_one_payload(row: Mapping[str, Any], keys: tuple[str, ...], *, uid: str) -> tuple[str, Any]:
    present = [k for k in keys if k in row]
    if len(present) != 1:
        raise SidecarError(
            f"uid={uid}: need exactly one of {keys}, got {present or 'none'} "
            "(whole-row fallback is forbidden)"
        )
    key = present[0]
    val = row[key]
    if val is None:
        raise SidecarError(f"uid={uid}: {key} is null")
    if isinstance(val, dict) and not val:
        raise SidecarError(f"uid={uid}: {key} is an empty dict")
    return key, val


def parse_cos_payload(
    payload: Mapping[str, Any],
    *,
    uid: str,
    required: Iterable[str] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, raw in payload.items():
        if name in {"uid", "split", "streams", "hyp", "oracle_stream"}:
            raise SidecarError(f"uid={uid}: cos payload contains metadata key {name!r}")
        try:
            x = float(raw)
        except (TypeError, ValueError) as e:
            raise SidecarError(f"uid={uid}: cos[{name!r}]={raw!r} is not a float") from e
        if x < -1.000001 or x > 1.000001:
            raise SidecarError(f"uid={uid}: cos[{name!r}]={x} outside [-1, 1]")
        out[str(name)] = x
    if not out:
        raise SidecarError(f"uid={uid}: no cosine values")
    if required:
        missing = [k for k in required if k not in out]
        if missing:
            raise SidecarError(f"uid={uid}: missing cos for streams {missing}")
    return out


def parse_cos_row(
    row: Mapping[str, Any],
    *,
    index: int = 0,
    required: Iterable[str] | None = None,
) -> tuple[str, dict[str, float]]:
    uid = _require_uid(row, index=index)
    _, payload = _exactly_one_payload(row, COS_PAYLOAD_KEYS, uid=uid)
    if not isinstance(payload, dict):
        raise SidecarError(f"uid={uid}: cosine payload must be a dict of stream→float")
    return uid, parse_cos_payload(payload, uid=uid, required=required)


def parse_pmusic_row(row: Mapping[str, Any], *, index: int = 0) -> tuple[str, dict[str, float] | float]:
    uid = _require_uid(row, index=index)
    _, payload = _exactly_one_payload(row, PM_PAYLOAD_KEYS, uid=uid)
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        x = float(payload)
        if x < 0.0 or x > 1.0:
            raise SidecarError(f"uid={uid}: p_music={x} outside [0, 1]")
        return uid, x
    if isinstance(payload, dict):
        out: dict[str, float] = {}
        for name, raw in payload.items():
            try:
                x = float(raw)
            except (TypeError, ValueError) as e:
                raise SidecarError(f"uid={uid}: p_music[{name!r}]={raw!r} is not a float") from e
            if x < 0.0 or x > 1.0:
                raise SidecarError(f"uid={uid}: p_music[{name!r}]={x} outside [0, 1]")
            out[str(name)] = x
        if not out:
            raise SidecarError(f"uid={uid}: p_music dict is empty")
        return uid, out
    raise SidecarError(f"uid={uid}: p_music must be a float or dict of stream→float")


def parse_qkw_row_with_kind(
    row: Mapping[str, Any], *, index: int = 0
) -> tuple[str, dict[str, float], str]:
    """Higher is better; return the original score convention as well."""
    uid = _require_uid(row, index=index)
    key, payload = _exactly_one_payload(row, QKW_PAYLOAD_KEYS, uid=uid)
    if not isinstance(payload, dict):
        raise SidecarError(f"uid={uid}: {key} must be a dict of stream→float")
    out: dict[str, float] = {}
    for name, raw in payload.items():
        try:
            x = float(raw)
        except (TypeError, ValueError) as e:
            raise SidecarError(f"uid={uid}: {key}[{name!r}]={raw!r} is not a float") from e
        if not math.isfinite(x):
            raise SidecarError(f"uid={uid}: {key}[{name!r}]={x} is not finite")
        if key == "q_kw" and not 0.0 <= x <= 1.0:
            raise SidecarError(f"uid={uid}: q_kw[{name!r}]={x} outside [0, 1]")
        out[str(name)] = -x if key == "nll" else x
    if not out:
        raise SidecarError(f"uid={uid}: {key} is empty after parse")
    return uid, out, key


def parse_qkw_row(row: Mapping[str, Any], *, index: int = 0) -> tuple[str, dict[str, float]]:
    """Compatibility wrapper; higher score is better."""
    uid, payload, _ = parse_qkw_row_with_kind(row, index=index)
    return uid, payload


def load_qkw_sidecar(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, row in enumerate(load_jsonl(path)):
        uid, payload = parse_qkw_row(row, index=i)
        if uid in out:
            raise SidecarError(f"duplicate uid={uid} in {path}")
        out[uid] = payload
    return out


def load_qkw_sidecar_with_kind(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Load scores plus their per-uid convention (``q_kw`` or ``nll``)."""
    scores: dict[str, dict[str, float]] = {}
    kinds: dict[str, str] = {}
    for i, row in enumerate(load_jsonl(path)):
        uid, payload, kind = parse_qkw_row_with_kind(row, index=i)
        if uid in scores:
            raise SidecarError(f"duplicate uid={uid} in {path}")
        scores[uid] = payload
        kinds[uid] = kind
    return scores, kinds


def load_cos_sidecar(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, row in enumerate(load_jsonl(path)):
        uid, payload = parse_cos_row(row, index=i)
        if uid in out:
            raise SidecarError(f"duplicate uid={uid} in {path}")
        out[uid] = payload
    return out


def load_pmusic_sidecar(path: Path) -> dict[str, dict[str, float] | float]:
    out: dict[str, dict[str, float] | float] = {}
    for i, row in enumerate(load_jsonl(path)):
        uid, payload = parse_pmusic_row(row, index=i)
        if uid in out:
            raise SidecarError(f"duplicate uid={uid} in {path}")
        out[uid] = payload
    return out


def clip_p_music(value: dict[str, float] | float | None, stream: str | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        if not stream:
            raise SidecarError("p_music dict requires a stream name")
        if stream not in value:
            raise SidecarError(f"p_music dict has no entry for stream={stream!r}")
        return float(value[stream])
    raise SidecarError(f"bad p_music type {type(value)}")


def parse_paircos_row(row: Mapping[str, Any], *, index: int = 0) -> tuple[str, dict[str, float]]:
    """Parse pairwise stream cosine keyed as ``stream_a|stream_b``."""
    uid = _require_uid(row, index=index)
    payload = row.get(PAIR_COS_KEY)
    if not isinstance(payload, dict) or not payload:
        raise SidecarError(f"uid={uid}: {PAIR_COS_KEY} must be a non-empty dict")
    out: dict[str, float] = {}
    for key, raw in payload.items():
        name = str(key)
        left, sep, right = name.partition("|")
        if not sep or not left or not right or left == right or name.count("|") != 1:
            raise SidecarError(f"uid={uid}: pair_cos key {name!r} must be 'stream_a|stream_b'")
        try:
            value = float(raw)
        except (TypeError, ValueError) as e:
            raise SidecarError(f"uid={uid}: pair_cos[{name!r}]={raw!r} is not a float") from e
        if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
            raise SidecarError(f"uid={uid}: pair_cos[{name!r}]={value} outside [-1, 1]")
        out[name] = value
    return uid, out


def load_paircos_sidecar(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, row in enumerate(load_jsonl(path)):
        uid, payload = parse_paircos_row(row, index=i)
        if uid in out:
            raise SidecarError(f"duplicate uid={uid} in {path}")
        out[uid] = payload
    return out
