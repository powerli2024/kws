#!/usr/bin/env python3
"""KWS-local pipeline: sidecar → T0–T4 picks → multi best_sep → CMD cosine rank.

Does not run extract Presence / mix ASR. Those consume the exported groups later.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--data-dir", type=Path, default=Path(r"d:\media\datasetA"))
    p.add_argument("--backend", default="eres2netv2")
    p.add_argument("--device", default=None)
    p.add_argument("--qkw-jsonl", type=Path, default=None, help="real T2 q_kw/nll sidecar")
    p.add_argument("--paircos-jsonl", type=Path, default=None, help="optional pairwise ERes cosine sidecar")
    p.add_argument("--require-e2", action="store_true", help="fail unless a real q_kw sidecar is supplied")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-sidecar", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument(
        "--with-se-groups",
        action="store_true",
        help="disabled until post-SE ASR validation is wired",
    )
    p.add_argument(
        "--reuse-enriched",
        action="store_true",
        help="reuse reports/best_sep_enriched.jsonl only when it was built for this exact pos_neg tree",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(r"d:\media\pos_neg\best_sep_groups"),
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> int:
    args = parse_args()
    py = sys.executable
    enriched = ROOT / "reports" / "best_sep_enriched.jsonl"
    if args.with_se_groups:
        raise SystemExit(
            "--with-se-groups is disabled: this repository has no post-SE ASR safety score, "
            "so exporting an SE-labelled copy would not be a valid experiment"
        )
    if not args.reuse_enriched or not enriched.is_file():
        run(
            [
                py,
                "scripts/rebuild_best_sep.py",
                "--pos-neg",
                str(args.pos_neg),
                "--allow-legacy",
            ]
        )
    cos = ROOT / "reports" / "sidecars" / "cos_to_raw.jsonl"
    pm = ROOT / "reports" / "sidecars" / "p_music.jsonl"
    paircos = ROOT / "reports" / "sidecars" / "pair_cos.jsonl"
    extra_limit = ["--limit", str(args.limit)] if args.limit else []
    if not args.skip_sidecar:
        cmd = [
            py,
            "scripts/build_eres_sidecar.py",
            "--enriched",
            str(enriched),
            "--pos-neg",
            str(args.pos_neg),
            "--data-dir",
            str(args.data_dir),
            "--backend",
            args.backend,
            "--out-cos",
            str(cos),
            "--out-pmusic",
            str(pm),
            "--out-paircos",
            str(paircos),
            *extra_limit,
        ]
        if args.device:
            cmd += ["--device", args.device]
        run(cmd)
    if not cos.is_file():
        raise SystemExit(f"missing sidecar {cos}; run without --skip-sidecar")

    if args.require_e2 and args.qkw_jsonl is None:
        raise SystemExit("--require-e2 needs --qkw-jsonl from frozen forced decode")
    if args.qkw_jsonl is not None and not args.qkw_jsonl.is_file():
        raise SystemExit(f"q_kw sidecar not found: {args.qkw_jsonl}")
    if args.paircos_jsonl is not None and not args.paircos_jsonl.is_file():
        raise SystemExit(f"pair_cos sidecar not found: {args.paircos_jsonl}")
    picks = ROOT / "reports" / "t0_t4_picks.jsonl"
    pick_cmd = [
        py,
        "scripts/run_t0_t4.py",
        "--enriched",
        str(enriched),
        "--cos-jsonl",
        str(cos),
        "--pmusic-jsonl",
        str(pm),
        "--picks",
        str(picks),
        *extra_limit,
    ]
    if args.qkw_jsonl is not None:
        pick_cmd += ["--qkw-jsonl", str(args.qkw_jsonl), "--strict-text"]
        pair_arg = args.paircos_jsonl or paircos
        if pair_arg.is_file():
            pick_cmd += ["--paircos-jsonl", str(pair_arg)]
        groups = ["e0_raw", "e1_t0", "e2_qkw", "skip_then_t0", "skip_then_t2"]
    else:
        # Do not silently publish a degraded E2 result. This mode is E0/E1 only.
        pick_cmd += ["--arm", "T0"]
        groups = ["e0_raw", "e1_t0", "skip_then_t0"]
        print("[WARN] no --qkw-jsonl: running E0/E1 only; E2 is intentionally omitted.", flush=True)
    run(pick_cmd)
    if not args.skip_export:
        cmd = [
            py,
            "scripts/export_best_sep_groups.py",
            "--picks",
            str(picks),
            "--pos-neg",
            str(args.pos_neg),
            "--data-dir",
            str(args.data_dir),
            "--out-root",
            str(args.out_root),
            *extra_limit,
        ]
        for g in groups:
            cmd += ["--group", g]
        if args.with_se_groups:
            cmd += ["--se-backend", "spectral"]
        run(cmd)

    cmd = [
        py,
        "scripts/eval_cmd_cosine.py",
        "--data-dir",
        str(args.data_dir),
        "--backend",
        args.backend,
        "--baseline",
        "e1_t0",
        *extra_limit,
    ]
    if args.device:
        cmd += ["--device", args.device]
    for g in groups:
        cmd += ["--dir", f"{g}={args.out_root / g}"]
    run(cmd)
    print("[OK] kws eval pipeline done — ranking is CMD cosine, not Presence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
