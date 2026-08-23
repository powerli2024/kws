#!/usr/bin/env python3
"""Compare two or more best_sep directories (KWS enroll purity bake-off).

CER is a *constraint*, not the ranking objective (most winners are already CER=0).
Acoustic proxies (SNR, p_music, duration) are diagnostics only.
KWS-local ranking is enroll↔CMD cosine (scripts/eval_cmd_cosine.py).
Frozen Presence on extract@main is a later contest veto, not this step.

Usage:
  python scripts/compare_best_sep.py \\
    --dir current=d:\\media\\pos_neg\\best_sep \\
    --dir kws_sep=/root/autodl-tmp/kws_sep/best_sep \\
    --baseline current \\
    --out reports/best_sep_bakeoff.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.best_sep_eval import (  # noqa: E402
    pairwise_disagreement,
    summarize_best_sep,
    verdict,
)
from kws.iojson import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare best_sep dumps for enroll purity")
    p.add_argument(
        "--dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="可重复。例: --dir current=d:\\media\\pos_neg\\best_sep",
    )
    p.add_argument("--baseline", default=None, help="对照名；默认第一个 --dir")
    p.add_argument("--max-audio", type=int, default=0, help="声学代理最多读多少条 wav；0=全量")
    p.add_argument(
        "--presence-json",
        type=Path,
        default=None,
        help="optional later contest veto; not used to rank KWS groups in this repo",
    )
    p.add_argument(
        "--cmd-eval-json",
        type=Path,
        default=None,
        help="eval_cmd_cosine.py output (summaries + rank)",
    )
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "best_sep_bakeoff.json")
    return p.parse_args()


def _parse_dirs(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for i, spec in enumerate(specs):
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = f"dir{i}", spec
        name = name.strip()
        p = Path(path.strip()).expanduser()
        if not name:
            raise SystemExit(f"非法 --dir: {spec}")
        out.append((name, p))
    if not out:
        raise SystemExit("至少给一个 --dir NAME=PATH")
    return out


def _md(payload: dict[str, Any]) -> str:
    lines = [
        "# best_sep bake-off",
        "",
        payload["verdict"].get("note", ""),
        "",
        "| name | n_ok | missing | CER mean | CER=0 | original% | dur_p10 | snr_med |",
        "|------|------|---------|----------|-------|-----------|---------|---------|",
    ]
    for name, s in payload["summaries"].items():
        def f(x, nd=4):
            return "—" if x is None else round(float(x), nd)

        orig = s.get("original_winner_rate")
        lines.append(
            f"| {name} | {s['n_ok']} | {s['n_missing_wav']} | {f(s.get('oracle_cer_mean'))} | "
            f"{f(s.get('cer0_rate'))} | {f(orig, 3)} | {f(s.get('dur_sec_p10'), 3)} | "
            f"{f(s.get('snr_med_db_mean'), 2)} |"
        )
    if payload.get("pairs"):
        lines += ["", "## pairwise disagreement", ""]
        for pair in payload["pairs"]:
            d = pair["disagreement"]
            lines.append(
                f"- `{pair['a']}` vs `{pair['b']}`: common={d['n_common']} "
                f"wav_diff={d['n_wav_fingerprint_diff']} stream_diff={d['n_oracle_stream_diff']} "
                f"stage_diff={d['n_best_stage_diff']}"
            )
    v = payload["verdict"]
    if v.get("adopt"):
        lines += ["", f"**Adopt:** `{v['adopt']}`"]
    else:
        lines += ["", f"**Keep baseline:** `{v['baseline']}`"]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    dirs = _parse_dirs(args.dir)
    names = [n for n, _ in dirs]
    baseline = args.baseline or names[0]
    if baseline not in names:
        raise SystemExit(f"--baseline={baseline} 不在 {names}")

    summaries: dict[str, dict[str, Any]] = {}
    slim: dict[str, dict[str, Any]] = {}
    for name, path in dirs:
        print(f"[INFO] summarize {name} ← {path}", flush=True)
        s = summarize_best_sep(path, max_audio=args.max_audio)
        summaries[name] = s
        slim[name] = {k: v for k, v in s.items() if k != "uids"}
        print(
            f"       n_ok={s['n_ok']} missing={s['n_missing_wav']} "
            f"cer_mean={s.get('oracle_cer_mean')} cer0={s.get('cer0_rate')}",
            flush=True,
        )

    pairs = []
    for i, (na, _) in enumerate(dirs):
        for nb, _ in dirs[i + 1 :]:
            d = pairwise_disagreement(summaries[na], summaries[nb])
            pairs.append({"a": na, "b": nb, "disagreement": d})
            print(
                f"[INFO] {na} vs {nb}: wav_diff={d['n_wav_fingerprint_diff']}/"
                f"{d['n_common']}",
                flush=True,
            )

    presence = None
    if args.presence_json and args.presence_json.is_file():
        raw = json.loads(args.presence_json.read_text(encoding="utf-8"))
        presence = raw.get("presence") or raw.get("arms") or raw
        if not isinstance(presence, dict):
            raise SystemExit("presence json 须含 {name: {frr, far}}")

    cmd = None
    if args.cmd_eval_json and args.cmd_eval_json.is_file():
        raw = json.loads(args.cmd_eval_json.read_text(encoding="utf-8"))
        cmd = raw.get("summaries")
        if not isinstance(cmd, dict):
            raise SystemExit("cmd-eval json 须含 summaries")

    v = verdict(names=names, summaries=summaries, presence=presence, baseline=baseline, cmd=cmd)
    payload = {
        "baseline": baseline,
        "summaries": slim,
        "pairs": pairs,
        "verdict": v,
        "protocol": "docs/BEST_SEP_EVAL.md",
    }
    write_json(args.out, payload)
    md_path = args.out.with_suffix(".md")
    md_path.write_text(_md(payload), encoding="utf-8")
    print(json.dumps(v, ensure_ascii=False, indent=2))
    print(f"[OK] {args.out}")
    print(f"[OK] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
