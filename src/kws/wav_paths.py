"""Resolve KWS / CMD / BSS stream wavs.

Stage indexes from AutoDL store `kws_path=/root/datasetA/...` which is dead on
Windows. Always join `--data-dir` with `kws_rel`. Original BSS stream is
`{uid}_peak.wav`, not `{uid}_original.wav`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

STREAM_TO_WAV_TAG = {"original": "peak"}
_UID_RE = re.compile(r"^(pos|neg)_(\d+)$", re.I)


def stream_wav_tag(stream: str) -> str:
    name = str(stream)
    return STREAM_TO_WAV_TAG.get(name, name)


def parse_uid(uid: str, rec: Mapping[str, Any] | None = None) -> tuple[str, str]:
    rec = rec or {}
    split = str(rec.get("split") or "")
    raw = str(uid)
    m = _UID_RE.match(raw)
    if m:
        return m.group(1).lower(), m.group(2)
    if split:
        idx = rec.get("id")
        if idx is not None:
            return split, str(idx)
        if raw.startswith(split + "_"):
            return split, raw[len(split) + 1 :]
    if "_" in raw:
        a, b = raw.split("_", 1)
        return a, b
    raise ValueError(f"cannot parse uid={uid!r}")


def resolve_kws_wav(data_dir: Path, rec: Mapping[str, Any]) -> Path | None:
    data_dir = Path(data_dir)
    rel = str(rec.get("kws_rel") or "").replace("\\", "/").lstrip("/")
    if rel:
        p = data_dir / rel
        if p.is_file():
            return p.resolve()
    uid = str(rec.get("uid") or "")
    if not uid:
        return None
    try:
        split, idx = parse_uid(uid, rec)
    except ValueError:
        return None
    p = data_dir / split / f"kws_{idx}.wav"
    return p.resolve() if p.is_file() else None


def resolve_cmd_wav(
    data_dir: Path,
    rec: Mapping[str, Any],
    *,
    dataset_row: Mapping[str, Any] | None = None,
) -> Path | None:
    data_dir = Path(data_dir)
    if dataset_row:
        rel = str(dataset_row.get("识别音频") or dataset_row.get("cmd_rel") or "")
        rel = rel.replace("\\", "/").lstrip("/")
        if rel:
            p = data_dir / rel
            if p.is_file():
                return p.resolve()
    uid = str(rec.get("uid") or "")
    if not uid:
        return None
    try:
        split, idx = parse_uid(uid, rec)
    except ValueError:
        return None
    p = data_dir / split / f"cmd_{idx}.wav"
    return p.resolve() if p.is_file() else None


def stage_wav_dir(pos_neg: Path, split: str, stage: str) -> Path:
    return Path(pos_neg) / split / stage / "wav"


def resolve_stream_wav(
    pos_neg: Path,
    rec: Mapping[str, Any],
    stream: str,
) -> Path | None:
    """Look up one BSS stream under the record's best_stage wav dir."""
    split = str(rec.get("split") or "")
    uid = str(rec.get("uid") or "")
    stage = str(rec.get("best_stage") or rec.get("stage") or "")
    if not (split and uid and stage):
        return None
    wav_dir = stage_wav_dir(pos_neg, split, stage)
    tag = stream_wav_tag(stream)
    names = [f"{uid}_{tag}.wav"]
    if tag != stream:
        names.append(f"{uid}_{stream}.wav")
    if stream == "original":
        names.extend([f"{uid}_peak.wav", f"{uid}_original.wav"])
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = wav_dir / name
        if p.is_file():
            return p.resolve()
    return None


def load_dataset_index(data_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Map (split, id) → pos.jsonl / neg.jsonl row."""
    from .iojson import load_jsonl

    out: dict[tuple[str, int], dict[str, Any]] = {}
    data_dir = Path(data_dir)
    for split in ("pos", "neg"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            try:
                idx = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            out[(split, idx)] = row
    return out


def dataset_row_for(rec: Mapping[str, Any], index: Mapping[tuple[str, int], dict[str, Any]]) -> dict[str, Any] | None:
    uid = str(rec.get("uid") or "")
    try:
        split, idx_s = parse_uid(uid, rec)
        return index.get((split, int(idx_s)))
    except (ValueError, TypeError):
        return None
