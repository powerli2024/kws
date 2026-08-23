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
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-sidecar", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--with-se-groups", action="store_true", help="also export t1_spectral and t4_spectral")
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
    if not enriched.is_file():
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
            *extra_limit,
        ]
        if args.device:
            cmd += ["--device", args.device]
        run(cmd)
    if not cos.is_file():
        raise SystemExit(f"missing sidecar {cos}; run without --skip-sidecar")

    picks = ROOT / "reports" / "t0_t4_picks.jsonl"
    run(
        [
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
    )
    groups = ["e0_raw", "e1_t0", "e2_qkw", "skip_then_t0", "skip_then_t2"]
    if args.with_se_groups:
        groups += ["t1_spectral", "t4_spectral"]
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
        "t0",
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
