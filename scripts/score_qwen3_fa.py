#!/usr/bin/env python3
"""Generate crop-boundary sidecar with Qwen3-ForcedAligner-0.6B.

The official aligner exposes timestamps but no calibrated confidence.  This
script therefore writes crop evidence only; it never fabricates MMS route
scores or q_kw from timestamps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.audio import load_wav_mono  # noqa: E402
from kws.iojson import limit_rows_balanced, load_jsonl  # noqa: E402
from kws.wav_paths import resolve_stream_wav  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3 forced-alignment crop sidecar")
    p.add_argument("--enriched", type=Path, default=ROOT / "reports" / "best_sep_enriched.jsonl")
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "sidecars" / "qwen3_fa.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _items(result):
    # ForcedAlignResult supports sequence access in the official package; keep
    # compatibility with revisions exposing an explicit .items field.
    value = getattr(result, "items", result)
    return list(value)


def main() -> int:
    args = parse_args()
    if not args.enriched.is_file() or not args.pos_neg.is_dir() or not args.model_dir.is_dir():
        raise SystemExit("[ERR] --enriched, --pos-neg and local --model-dir must exist")
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"[ERR] output exists: {args.out}; pass --overwrite")
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as exc:
        raise SystemExit("[ERR] install the official qwen-asr package and torch") from exc

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    aligner = Qwen3ForcedAligner.from_pretrained(
        str(args.model_dir.resolve()), dtype=dtype, device_map=args.device,
    )
    rows = load_jsonl(args.enriched)
    if args.limit:
        rows = limit_rows_balanced(rows, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row_i, rec in enumerate(rows, start=1):
            uid = str(rec.get("uid") or "")
            text = str(rec.get("wake_text") or "").strip()
            lang = {"zh": "Chinese", "en": "English"}.get(str(rec.get("lang") or ""))
            streams = rec.get("streams") or {}
            if not uid or not text or lang is None or not streams:
                raise SystemExit(f"[ERR] uid={uid}: invalid uid/text/lang/streams")
            names: list[str] = []
            audios: list[tuple[object, int]] = []
            durations: list[float] = []
            for stream in streams:
                path = resolve_stream_wav(args.pos_neg, rec, stream)
                if path is None:
                    raise SystemExit(f"[ERR] uid={uid}: missing stream wav {stream}")
                wav, sr = load_wav_mono(path)
                if sr != 16000 or wav.size == 0:
                    raise SystemExit(f"[ERR] uid={uid}: invalid wav {stream}")
                names.append(str(stream))
                audios.append((wav, sr))
                durations.append(float(len(wav) / sr))
            results = aligner.align(audio=audios, text=[text] * len(audios), language=[lang] * len(audios))
            if len(results) != len(names):
                raise SystemExit(f"[ERR] uid={uid}: aligner batch size mismatch")
            payload: dict[str, dict] = {}
            for name, result, duration in zip(names, results, durations):
                items = _items(result)
                valid = [it for it in items if float(it.end_time) > float(it.start_time)]
                if not items or not valid:
                    # Keep a structurally valid full-span row. coverage=0 makes
                    # the downstream safe-crop policy fail closed.
                    start, end, coverage = 0.0, duration, 0.0
                else:
                    start = min(float(it.start_time) for it in valid)
                    end = max(float(it.end_time) for it in valid)
                    start = max(0.0, min(start, duration))
                    end = max(start + 1e-6, min(end, duration))
                    coverage = len(valid) / len(items)
                payload[name] = {
                    "coverage": round(float(coverage), 8),
                    "star_fraction": 0.0,
                    "start_sec": round(start, 6),
                    "end_sec": round(end, 6),
                    "duration_sec": round(duration, 6),
                    "edge_clipped": bool(start <= 0.02 or end >= duration - 0.02),
                    "n_items": len(items),
                    "n_positive_span": len(valid),
                }
            handle.write(json.dumps({"uid": uid, "model": "qwen3_forced_aligner_0.6b", "streams": payload}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"\r[INFO] uid {row_i}/{len(rows)} current={uid}", end="", flush=True)
    print()
    check = load_jsonl(args.out)
    if len(check) != len(rows) or {str(x["uid"]) for x in check} != {str(x["uid"]) for x in rows}:
        raise SystemExit("[ERR] final Qwen3-FA coverage mismatch")
    print(f"[OK] {args.out} uid={len(check)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
