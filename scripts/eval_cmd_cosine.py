#!/usr/bin/env python3
"""Rank best_sep groups by enroll↔CMD cosine (pos vs neg).

This is the KWS-local quality check: can the current encoder tell pos CMD from
neg CMD more cleanly after we swap enroll? Contest Presence is a later step
on extract@main using these exported trees.

Higher pos cosine and lower neg cosine → cleaner enroll.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.audio import cosine_sim, load_wav_mono  # noqa: E402
from kws.best_sep_eval import resolve_wav  # noqa: E402
from kws.cmd_eval import collect_split_scores, rank_groups, summarize_cmd_scores  # noqa: E402
from kws.eres import load_embedder  # noqa: E402
from kws.iojson import limit_rows_balanced, load_jsonl, write_json, write_jsonl  # noqa: E402
from kws.wav_paths import (  # noqa: E402
    dataset_row_for,
    load_dataset_index,
    parse_uid,
    resolve_cmd_wav,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeatable best_sep dir, e.g. --dir t0=d:\\media\\pos_neg\\best_sep_groups\\t0",
    )
    p.add_argument("--data-dir", type=Path, default=Path(r"d:\media\datasetA"))
    p.add_argument("--backend", default="eres2netv2")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--extract-ve", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--baseline", default=None)
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "eval" / "cmd_cosine.json")
    p.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "reports" / "eval" / "cmd_cosine_scores.jsonl",
    )
    return p.parse_args()


def _parse_dirs(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for i, spec in enumerate(specs):
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = f"dir{i}", spec
        out.append((name.strip(), Path(path.strip()).expanduser()))
    if not out:
        raise SystemExit("need at least one --dir NAME=PATH")
    return out


def _index_rows(best_sep: Path) -> list[dict]:
    idx = best_sep / "index.jsonl"
    if idx.is_file():
        return load_jsonl(idx)
    rows = []
    for split in ("pos", "neg"):
        d = best_sep / split
        if not d.is_dir():
            continue
        for wav in sorted(d.glob("*.wav")):
            rows.append({"uid": wav.stem, "split": split, "dest_rel": f"{split}/{wav.name}", "ok": True})
    return rows


def main() -> int:
    args = parse_args()
    dirs = _parse_dirs(args.dir)
    names = [n for n, _ in dirs]
    baseline = args.baseline or names[0]
    ds = load_dataset_index(args.data_dir)
    print(f"[INFO] load embedder backend={args.backend}", flush=True)
    enc = load_embedder(
        args.backend,
        model_dir=args.model_dir,
        device=args.device,
        extract_ve=args.extract_ve,
    )
    print(f"[INFO] encoder={enc.name}", flush=True)

    cmd_cache: dict[str, object] = {}
    all_scores: list[dict] = []
    summaries: dict[str, dict] = {}

    for name, path in dirs:
        rows = _index_rows(path)
        if args.limit:
            rows = limit_rows_balanced(rows, args.limit)
        scored = []
        n_miss = 0
        for rec in rows:
            if rec.get("ok") is False:
                continue
            uid = str(rec.get("uid") or "")
            split = str(rec.get("split") or "")
            if not split and uid:
                split = parse_uid(uid, rec)[0]
                rec = {**rec, "split": split}
            enroll = resolve_wav(path, rec)
            drow = dataset_row_for(rec, ds)
            cmd = resolve_cmd_wav(args.data_dir, rec, dataset_row=drow)
            if enroll is None or cmd is None:
                n_miss += 1
                continue
            ew, esr = load_wav_mono(enroll)
            e_enroll = enc.embed(ew, esr)
            ck = str(cmd)
            if ck not in cmd_cache:
                cw, csr = load_wav_mono(cmd)
                cmd_cache[ck] = enc.embed(cw, csr)
            e_cmd = cmd_cache[ck]
            cos = float(cosine_sim(e_enroll, e_cmd))
            wake = str(rec.get("wake_text") or (drow or {}).get("唤醒文本") or "")
            lang = rec.get("lang") or (drow or {}).get("lang")
            if not lang:
                lang = "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in wake) else "en"
            row = {
                "group": name,
                "uid": uid,
                "split": split,
                "lang": lang,
                "cos_enroll_cmd": max(-1.0, min(1.0, cos)),
                "enroll": str(enroll),
                "cmd": str(cmd),
            }
            scored.append(row)
            all_scores.append(row)
        pos, neg = collect_split_scores(scored)
        summaries[name] = {
            **summarize_cmd_scores(pos, neg),
            "path": str(path.resolve()),
            "n_scored": len(scored),
            "n_missing": n_miss,
            "encoder": enc.name,
        }
        print(
            f"[INFO] {name} n={len(scored)} miss={n_miss} "
            f"gap={summaries[name].get('mean_gap')} eer={summaries[name].get('eer')}",
            flush=True,
        )

    ranking = rank_groups(summaries, baseline=baseline)
    payload = {
        "baseline": baseline,
        "encoder": enc.name,
        "summaries": summaries,
        "rank": ranking,
        "protocol": "docs/BEST_SEP_EVAL.md",
        "note": ranking["note"],
    }
    write_json(args.out, payload)
    write_jsonl(args.scores, all_scores)
    md = args.out.with_suffix(".md")
    md.write_text(_md(payload), encoding="utf-8")
    print(json.dumps(ranking, ensure_ascii=False, indent=2))
    print(f"[OK] {args.out}")
    print(f"[OK] {md}")
    return 0


def _md(payload: dict) -> str:
    lines = [
        "# CMD cosine reject eval",
        "",
        payload.get("note") or "",
        "",
        f"encoder: `{payload.get('encoder')}`",
        "",
        "| group | n_pos | n_neg | pos_mean | neg_mean | gap | AUC | EER | EER-thr |",
        "|-------|-------|-------|----------|----------|-----|-----|-----|---------|",
    ]

    def f(x, nd=4):
        return "—" if x is None else round(float(x), nd)

    for name, s in payload["summaries"].items():
        eer = s.get("eer") or {}
        lines.append(
            f"| {name} | {s.get('n_pos')} | {s.get('n_neg')} | {f(s.get('pos_mean'))} | "
            f"{f(s.get('neg_mean'))} | {f(s.get('mean_gap'))} | {f(s.get('auc'))} | "
            f"{f(eer.get('eer'))} | {f(eer.get('threshold'))} |"
        )
    rank = payload.get("rank") or {}
    lines += ["", f"**Rank:** {' > '.join(rank.get('order') or [])}", ""]
    lines.append("Locked VE τ is a probe only (`locked_tau_probe_not_adopt`); do not adopt from it.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
