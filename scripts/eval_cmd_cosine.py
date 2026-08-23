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
from kws.cmd_eval import aggregate_lang_thresholds, paired_deltas, summarize_by_lang  # noqa: E402
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


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _locked_tau_compatible(backend: str, encoder_name: str) -> bool:
    """Frozen thresholds are tied to the ERes2NetV2 scoring space."""
    normalized = (backend or "").lower().strip()
    return normalized in {"eres2netv2", "eres2net", "eres"} and "eres2netv2" in encoder_name.lower()


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
    group_rows: dict[str, list[dict]] = {}
    group_meta: dict[str, dict] = {}

    for name, path in dirs:
        rows = _index_rows(path)
        if args.limit:
            rows = limit_rows_balanced(rows, args.limit)
        scored = []
        n_miss = 0
        n_rejected = 0
        rejected_by_split: dict[str, int] = {}
        missing_by_split: dict[str, int] = {}
        for rec in rows:
            if rec.get("ok") is False:
                if rec.get("error") == "rejected_enroll":
                    n_rejected += 1
                    _increment(rejected_by_split, str(rec.get("split") or "unknown"))
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
                _increment(missing_by_split, split or "unknown")
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
        group_rows[name] = scored
        group_meta[name] = {
            "path": str(path.resolve()),
            "n_missing": n_miss,
            "n_index": len(rows),
            "n_rejected": n_rejected,
            "rejected_by_split": rejected_by_split,
            "missing_by_split": missing_by_split,
        }
        print(
            f"[INFO] {name} raw_scored={len(scored)} reject={n_rejected} missing={n_miss}",
            flush=True,
        )
    if baseline not in group_rows:
        raise SystemExit(f"baseline {baseline!r} was not scored")

    baseline_uids = {str(row["uid"]) for row in group_rows[baseline]}
    if not baseline_uids:
        raise SystemExit(f"baseline {baseline!r} has no scored UIDs")
    common_uids = set(baseline_uids)
    for rows in group_rows.values():
        common_uids &= {str(row["uid"]) for row in rows}

    summaries: dict[str, dict] = {}
    comparable_rows: dict[str, list[dict]] = {}
    for name, rows in group_rows.items():
        scored_uids = {str(row["uid"]) for row in rows}
        comparable = [row for row in rows if str(row["uid"]) in common_uids]
        comparable_rows[name] = comparable
        pos, neg = collect_split_scores(comparable)
        by_lang = summarize_by_lang(comparable)
        coverage_vs_baseline = len(scored_uids & baseline_uids) / len(baseline_uids)
        summaries[name] = {
            **summarize_cmd_scores(pos, neg),
            **group_meta[name],
            "encoder": enc.name,
            "n_scored_raw": len(rows),
            "n_scored_common": len(comparable),
            "n_baseline_scored": len(baseline_uids),
            "n_common": len(common_uids),
            "coverage_vs_baseline": coverage_vs_baseline,
            "coverage_complete": scored_uids == baseline_uids,
            "by_lang": by_lang,
            "locked_tau_by_lang_aggregate": aggregate_lang_thresholds(by_lang),
        }
        print(
            f"[INFO] {name} common={len(comparable)}/{len(baseline_uids)} "
            f"gap={summaries[name].get('mean_gap')} eer={summaries[name].get('eer')}",
            flush=True,
        )
    paired_vs_baseline = {
        name: paired_deltas(comparable_rows[baseline], rows)
        for name, rows in comparable_rows.items()
        if name != baseline
    }

    locked_tau_compatible = _locked_tau_compatible(args.backend, enc.name)
    ranking = rank_groups(
        summaries,
        baseline=baseline,
        locked_tau_compatible=locked_tau_compatible,
        require_full_coverage=True,
    )
    payload = {
        "baseline": baseline,
        "encoder": enc.name,
        "locked_tau_compatible": locked_tau_compatible,
        "comparison": {
            "basis": "common_scored_uid_intersection",
            "n_baseline_scored": len(baseline_uids),
            "n_common": len(common_uids),
            "coverage_required_for_rank": True,
        },
        "summaries": summaries,
        "rank": ranking,
        "paired_vs_baseline": paired_vs_baseline,
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
        "| group | common / baseline | rejected | missing | pos_mean | neg_mean | gap | AUC | EER |",
        "|-------|-------------------|----------|---------|----------|----------|-----|-----|-----|",
    ]

    def f(x, nd=4):
        return "—" if x is None else round(float(x), nd)

    for name, s in payload["summaries"].items():
        eer = s.get("eer") or {}
        lines.append(
            f"| {name} | {s.get('n_scored_common')} / {s.get('n_baseline_scored')} | "
            f"{s.get('n_rejected')} | {s.get('n_missing')} | {f(s.get('pos_mean'))} | "
            f"{f(s.get('neg_mean'))} | {f(s.get('mean_gap'))} | {f(s.get('auc'))} | "
            f"{f(eer.get('eer'))} |"
        )
    rank = payload.get("rank") or {}
    lines += ["", f"**Rank:** {' > '.join(rank.get('order') or [])}", ""]
    lines.append(
        "Rank is blocked on incomplete coverage and on non-ERes2NetV2 backends. "
        "Frozen VE τ is a probe only; do not adopt from it."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
