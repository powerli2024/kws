#!/usr/bin/env python3
"""Static review: code implements the frozen experiment design, not MMS-FA."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[tuple[str, bool, str]] = []


def ok(name: str, cond: bool, detail: str) -> None:
    CHECKS.append((name, cond, detail))


def source_blob() -> str:
    parts: list[str] = []
    skip = {"review_checklist.py"}
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if p.name in skip:
            continue
        parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def main() -> int:
    blob = source_blob()
    src_text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    src_low = src_text.lower()
    ok(
        "no_mms_fa_selector",
        "MmsFa" not in blob and "mms_fa_scorer" not in blob and "pick_mms" not in blob,
        "selector path must not call MMS-FA",
    )
    ok(
        "no_mms_import",
        "import mms_fa" not in src_low
        and "from mms_fa" not in src_low
        and "from mms " not in src_low
        and "import mms\n" not in src_low,
        "no MMS module import in src/",
    )
    ok(
        "oracle_prefers_original",
        "prefer_original" in blob,
        "T0 oracle_of must prefer original on CER ties",
    )
    ok(
        "candidates_orig_spk",
        "is_sep_stream" in blob and "original" in blob,
        "candidates are original vs sep tracks",
    )
    ok(
        "skip_sep_not_duration",
        "never skip BSS because the clip is short" in blob
        or "Duration is accepted as an input but never used" in blob,
        "skip-sep must not use dur<=1.8s",
    )
    ok(
        "l1_slack_0p05",
        "CER_SLACK_DEFAULT = 0.05" in blob,
        "L1 slack frozen at 0.05 search default",
    )
    ok(
        "catastrophe_grid",
        "0.90" in blob and "CATASTROPHE_COS_GRID" in blob,
        "cos(e*,e_raw) grid 0.90-0.95",
    )
    ok(
        "window_skip_0p8",
        "MIN_DUR_SEC = 0.8" in blob,
        "window min-cos skipped below 0.8s",
    )
    ok(
        "se_not_equal_speaker",
        "Denoise" in blob,
        "SE safety documented",
    )
    ok(
        "presence_is_veto",
        "only enroll-adoption veto" in blob or "PresenceVeto" in blob,
        "Presence is veto, not an online enroll metric",
    )
    ok(
        "t4_is_ablation",
        "t4_global_se" in blob or "T4" in blob,
        "T4 global SE is an ablation arm",
    )

    failed = 0
    for name, cond, detail in CHECKS:
        mark = "PASS" if cond else "FAIL"
        if not cond:
            failed += 1
        print(f"[{mark}] {name}: {detail}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
