"""JSONL / stage-index helpers. No MMS-FA fields required."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_index_path(pos_neg_root: Path, split: str, best_stage: str) -> Path:
    """best_stage is e.g. s1_onnx_full or s7_cv_then_onnx_gate/thr_a."""
    return Path(pos_neg_root) / split / best_stage / "index.jsonl"


def index_by_uid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        uid = str(r.get("uid") or "")
        if uid:
            if uid in out:
                raise ValueError(f"duplicate uid in index: {uid}")
            out[uid] = r
    return out


def limit_rows_balanced(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Keep both splits so CMD-cosine EER is defined on a smoke subset."""
    if not n or n <= 0 or n >= len(rows):
        return rows
    pos = [r for r in rows if str(r.get("split") or "") == "pos"]
    neg = [r for r in rows if str(r.get("split") or "") == "neg"]
    if not pos or not neg:
        return rows[:n]
    n_neg = max(1, n // 2)
    n_pos = max(1, n - n_neg)
    return pos[:n_pos] + neg[:n_neg]
