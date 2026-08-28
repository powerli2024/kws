#!/usr/bin/env python3
"""Fail-fast validation of an extract-sep tree before KWS analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.iojson import write_json  # noqa: E402
from kws.sep_audit import audit_sep_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-neg", type=Path, required=True)
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--expected-uids", type=int, default=1838)
    parser.add_argument("--check-duration", action="store_true")
    parser.add_argument("--require-handoff", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    report = audit_sep_root(
        args.pos_neg, splits, expected_uids=args.expected_uids,
        check_duration=args.check_duration, require_handoff=args.require_handoff,
    )
    out = args.out or (args.pos_neg / "reports" / "kws_input_audit.json")
    write_json(out, report)
    print(json.dumps({
        "ok": report["ok"], "n_best_uid": report["n_best_uid"],
        "n_failures": len(report["failures"]), "n_warnings": len(report["warnings"]),
        "out": str(out.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
