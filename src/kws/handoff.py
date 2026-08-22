"""Validate extract@sep → kws output contract. No MMS-FA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "kws_sep_handoff/v1"
EXTRACT_REPO = "https://github.com/powerli2024/extract"
EXTRACT_BRANCH = "sep"


class HandoffError(ValueError):
    pass


def load_handoff(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise HandoffError(f"missing kws_handoff.json: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HandoffError(f"invalid JSON {p}: {e}") from e
    if not isinstance(obj, dict):
        raise HandoffError(f"{p} is not an object")
    schema = obj.get("schema")
    if schema != SCHEMA:
        raise HandoffError(f"{p} schema={schema!r} expected {SCHEMA}")
    if "mms_fa" not in obj or obj["mms_fa"] is not False:
        raise HandoffError(f"{p} mms_fa must be false, got {obj.get('mms_fa')!r}")
    within = obj.get("selector_within_stage")
    if within and within != "oracle_cer_prefer_original":
        raise HandoffError(f"{p} selector_within_stage={within!r}")
    return obj


def find_handoff(pos_neg: str | Path) -> Path | None:
    root = Path(pos_neg)
    for cand in (root / "kws_handoff.json", root / "best_sep" / "kws_handoff.json"):
        if cand.is_file():
            return cand
    return None
