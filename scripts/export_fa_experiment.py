#!/usr/bin/env python3
"""Materialize one FA experiment arm, applying only prevalidated crop plans."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.audio import load_wav_mono, save_wav_mono  # noqa: E402
from kws.iojson import load_jsonl, write_json, write_jsonl  # noqa: E402
from kws.wav_paths import resolve_kws_wav, resolve_stream_wav  # noqa: E402

ARMS = ("F0_t0_full", "F1_qkw_full", "F2_fa_full", "F3_agree_full", "F4_agree_safe_crop")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--picks", type=Path, default=ROOT / "reports" / "fa_experiment_picks.jsonl")
    p.add_argument("--enriched", type=Path, default=ROOT / "reports" / "best_sep_enriched.jsonl")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--data-dir", type=Path, default=Path(r"d:\media\datasetA"))
    p.add_argument("--out-root", type=Path, default=Path(r"d:\media\pos_neg\fa_experiment"))
    p.add_argument("--summary", type=Path, default=ROOT / "reports" / "fa_experiment_export.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    picks = {str(row["uid"]): row for row in load_jsonl(args.picks)}
    enriched = {str(row["uid"]): row for row in load_jsonl(args.enriched)}
    if set(picks) != set(enriched):
        raise SystemExit(f"strict UID coverage failed: picks={len(picks)} enriched={len(enriched)}")
    dest_root = args.out_root / args.arm
    index: list[dict] = []
    n_crop = 0
    for uid, pick in picks.items():
        rec = enriched[uid]
        spec = (pick.get("arms") or {}).get(args.arm)
        if not isinstance(spec, dict):
            raise SystemExit(f"uid={uid}: missing arm {args.arm}")
        stream = str(spec.get("chosen") or "")
        src = resolve_stream_wav(args.pos_neg, rec, stream)
        if src is None and stream == "original":
            src = resolve_kws_wav(args.data_dir, rec)
        if src is None:
            raise SystemExit(f"uid={uid}: missing wav for {stream}")
        split = str(rec.get("split") or pick.get("split") or "")
        dest = dest_root / split / f"{uid}.wav"
        crop = spec.get("crop")
        if crop:
            wav, sr = load_wav_mono(src)
            a = max(0, int(round(float(crop["start_sec"]) * sr)))
            b = min(len(wav), int(round(float(crop["end_sec"]) * sr)))
            if b <= a:
                raise SystemExit(f"uid={uid}: invalid crop samples {a}:{b}")
            save_wav_mono(dest, wav[a:b], sr)
            n_crop += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        index.append({
            "uid": uid, "split": split, "arm": args.arm, "chosen": stream,
            "crop": crop, "src_wav": str(src), "dest_rel": f"{split}/{uid}.wav", "ok": dest.is_file(),
        })
    write_jsonl(dest_root / "index.jsonl", index)
    summary = {"arm": args.arm, "n": len(index), "n_crop": n_crop, "path": str(dest_root.resolve())}
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
