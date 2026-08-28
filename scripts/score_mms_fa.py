#!/usr/bin/env python3
"""Generate MMS-FA route scores and boundaries in a pinned torchaudio 2.7 env."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.audio import load_wav_mono  # noqa: E402
from kws.iojson import limit_rows_balanced, load_jsonl  # noqa: E402
from kws.wav_paths import resolve_stream_wav  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMS-FA phonetic evidence sidecar")
    p.add_argument("--enriched", type=Path, default=ROOT / "reports" / "best_sep_enriched.jsonl")
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--uroman-dir", type=Path, required=True, help="local uroman repository containing bin/uroman.pl")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "sidecars" / "mms_fa.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _normalize_uroman(text: str, script: Path) -> str:
    try:
        proc = subprocess.run(
            ["perl", str(script)], input=text + "\n", text=True,
            capture_output=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"[ERR] uroman failed: {exc}") from exc
    value = proc.stdout.lower().replace("’", "'")
    value = re.sub(r"([^a-z' ])", " ", value)
    value = re.sub(r" +", " ", value).strip()
    if not value:
        raise SystemExit(f"[ERR] uroman produced empty text for {text!r}")
    return value


def _weighted_mean_log(spans) -> float:
    weights = [max(1, int(span.end) - int(span.start)) for span in spans]
    values = [math.log(max(float(span.score), 1e-8)) for span in spans]
    return float(np.average(values, weights=weights))


def main() -> int:
    args = parse_args()
    uroman = args.uroman_dir / "bin" / "uroman.pl"
    if not args.enriched.is_file() or not args.pos_neg.is_dir() or not uroman.is_file():
        raise SystemExit("[ERR] --enriched, --pos-neg and --uroman-dir/bin/uroman.pl must exist")
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"[ERR] output exists: {args.out}; pass --overwrite")
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle
    except ImportError as exc:
        raise SystemExit("[ERR] install matching torch==2.7.* and torchaudio==2.7.*") from exc
    if not str(torchaudio.__version__).startswith("2.7."):
        raise SystemExit(f"[ERR] require torchaudio 2.7.*, got {torchaudio.__version__}")
    device = torch.device(args.device)
    model = bundle.get_model(with_star=True).to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    star_id = bundle.get_dict().get("*")

    rows = load_jsonl(args.enriched)
    if args.limit:
        rows = limit_rows_balanced(rows, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row_i, rec in enumerate(rows, start=1):
            uid = str(rec.get("uid") or "")
            wake_text = str(rec.get("wake_text") or "").strip()
            streams = rec.get("streams") or {}
            if not uid or not wake_text or not streams:
                raise SystemExit(f"[ERR] uid={uid}: invalid uid/wake_text/streams")
            normalized = _normalize_uroman(wake_text, uroman)
            transcript = normalized.split()
            tokens = tokenizer(transcript)
            n_target = sum(len(word) for word in tokens)
            if n_target <= 0:
                raise SystemExit(f"[ERR] uid={uid}: tokenizer produced no target tokens")
            payload: dict[str, dict] = {}
            for stream in streams:
                path = resolve_stream_wav(args.pos_neg, rec, stream)
                if path is None:
                    raise SystemExit(f"[ERR] uid={uid}: missing stream wav {stream}")
                wav, sr = load_wav_mono(path)
                if sr != bundle.sample_rate or wav.size == 0:
                    raise SystemExit(f"[ERR] uid={uid}: invalid wav {stream} sr={sr}")
                waveform = torch.from_numpy(wav).unsqueeze(0).to(device)
                with torch.inference_mode():
                    emission, _ = model(waveform)
                    word_spans = aligner(emission[0], tokens)
                spans = [span for word in word_spans for span in word]
                if not spans:
                    # A malformed result is recorded as zero coverage and will
                    # fail closed in the route/crop policy.
                    payload[str(stream)] = {
                        "coverage": 0.0, "mean_logp": -18.42068074, "p10_logp": -18.42068074,
                        "star_fraction": 1.0, "start_sec": 0.0,
                        "end_sec": round(float(len(wav) / sr), 6),
                        "duration_sec": round(float(len(wav) / sr), 6), "edge_clipped": True,
                        "normalized_text": normalized,
                    }
                    continue
                ratio = len(wav) / int(emission.shape[1]) / sr
                logs = [math.log(max(float(span.score), 1e-8)) for span in spans]
                start = min(int(span.start) for span in spans) * ratio
                end = max(int(span.end) for span in spans) * ratio
                duration = len(wav) / sr
                star_fraction = 0.0
                if star_id is not None:
                    frame_a = min(int(span.start) for span in spans)
                    frame_b = max(int(span.end) for span in spans)
                    aligned_argmax = emission[0, frame_a:frame_b].argmax(dim=-1)
                    if aligned_argmax.numel():
                        star_fraction = float((aligned_argmax == int(star_id)).float().mean().item())
                payload[str(stream)] = {
                    "coverage": round(min(1.0, len(spans) / n_target), 8),
                    "mean_logp": round(_weighted_mean_log(spans), 8),
                    "p10_logp": round(float(np.quantile(logs, 0.10)), 8),
                    "star_fraction": round(star_fraction, 8),
                    "start_sec": round(max(0.0, start), 6),
                    "end_sec": round(min(duration, end), 6),
                    "duration_sec": round(duration, 6),
                    "edge_clipped": bool(start <= 0.02 or end >= duration - 0.02),
                    "normalized_text": normalized,
                    "n_target_tokens": n_target,
                    "n_aligned_tokens": len(spans),
                }
            out_row = {
                "uid": uid,
                "model": "torchaudio_mms_fa",
                "streams": payload,
                "versions": {"torch": torch.__version__, "torchaudio": torchaudio.__version__, "normalizer": "uroman"},
            }
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"\r[INFO] uid {row_i}/{len(rows)} current={uid}", end="", flush=True)
    print()
    check = load_jsonl(args.out)
    if len(check) != len(rows) or {str(x["uid"]) for x in check} != {str(x["uid"]) for x in rows}:
        raise SystemExit("[ERR] final MMS-FA coverage mismatch")
    print(f"[OK] {args.out} uid={len(check)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
