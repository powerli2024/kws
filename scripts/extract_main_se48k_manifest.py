#!/usr/bin/env python3
"""Batch adapter for extract-main's MossFormer2_SE_48K.

The manifest is produced by ``run_se_route_eval.py``.  This adapter loads the
ClearVoice model once, keeps the full waveform by default, and writes canonical
16 kHz mono output for KWS/ASR.  It is intentionally a CLI boundary: KWS does
not import extract-main at module import time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MossFormer2_SE_48K manifest adapter")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--extract-main", type=Path, default=Path("/root/extract-main"))
    p.add_argument("--clearvoice-root", type=Path, default=Path("/root/autodl-tmp/ClearerVoice-Studio"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk-sec", type=float, default=0.0, help="0=full waveform; only fallback on OOM")
    p.add_argument("--overlap-sec", type=float, default=0.08)
    return p.parse_args()


def _chunked(se, wav, *, chunk_sec: float, overlap_sec: float):
    import numpy as np

    chunk = max(1, int(float(chunk_sec) * se.sample_rate))
    overlap = max(0, min(int(float(overlap_sec) * se.sample_rate), chunk // 4))
    if len(wav) <= chunk:
        return se.enhance_48k(wav)
    hop = max(1, chunk - overlap)
    accum = np.zeros(len(wav), dtype=np.float64)
    weight = np.zeros(len(wav), dtype=np.float64)
    for start in range(0, len(wav), hop):
        end = min(len(wav), start + chunk)
        y = se.enhance_48k(wav[start:end])[: end - start]
        w = np.ones(end - start, dtype=np.float64)
        if overlap and start:
            n = min(overlap, len(w))
            w[:n] = np.linspace(0.0, 1.0, n, endpoint=False)
        if overlap and end < len(wav):
            n = min(overlap, len(w))
            w[-n:] = np.minimum(w[-n:], np.linspace(1.0, 0.0, n, endpoint=False))
        accum[start:end] += y * w
        weight[start:end] += w
        if end == len(wav):
            break
    return np.asarray(accum / np.maximum(weight, 1e-8), dtype=np.float32)


def main() -> int:
    import json
    import os

    args = parse_args()
    if args.chunk_sec < 0 or args.overlap_sec < 0:
        raise SystemExit("[ERR] chunk/overlap seconds must be non-negative")
    root = args.extract_main.resolve()
    ve_scripts = root / "ve" / "scripts"
    if not ve_scripts.is_dir():
        raise SystemExit(f"[ERR] extract-main ve/scripts missing: {ve_scripts}")
    sys.path.insert(0, str(ve_scripts))
    from audio_io import load_audio, resample_wav, save_audio  # type: ignore
    from moss_se48k import MossFormer2SE48K  # type: ignore

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("[OK] empty manifest")
        return 0
    se = MossFormer2SE48K(clearvoice_root=args.clearvoice_root, device=args.device)
    print(f"[INFO] MossFormer2_SE_48K loaded; files={len(rows)} device={args.device}", flush=True)
    for index, row in enumerate(rows, start=1):
        src, dst = Path(row["input"]), Path(row["output"])
        if not src.is_file():
            raise SystemExit(f"[ERR] input missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and dst.stat().st_size > 44:
            print(f"\r[SE48K] {index}/{len(rows)} reuse={dst.name}", end="", flush=True)
            continue
        wav16, sr = load_audio(src, sr=16000)
        wav48 = resample_wav(wav16, sr, 48000, method="poly")
        try:
            out48 = se.enhance_48k(wav48) if not args.chunk_sec else _chunked(
                se, wav48, chunk_sec=args.chunk_sec, overlap_sec=args.overlap_sec
            )
        except Exception as exc:
            # Do not hide model errors.  Only retry likely GPU OOM with bounded
            # chunks; all other errors remain actionable and fail the run.
            message = str(exc).lower()
            if "out of memory" not in message and "cuda error" not in message:
                raise
            last = exc
            for seconds in (8.0, 4.0, 2.0):
                try:
                    out48 = _chunked(se, wav48, chunk_sec=seconds, overlap_sec=args.overlap_sec)
                    break
                except Exception as retry_exc:
                    last = retry_exc
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            else:
                raise last
        out16 = resample_wav(out48, 48000, 16000, method="poly")
        save_audio(dst, out16, 16000)
        print(f"\r[SE48K] {index}/{len(rows)} wrote={dst.name}", end="", flush=True)
    print()
    print(f"[OK] SE48K manifest complete: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
