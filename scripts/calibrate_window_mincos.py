#!/usr/bin/env python3
"""Window min-cos over enroll wavs. Encoder is optional.

Without eres, writes duration / skip reasons only. With --encoder eres
(requires modelscope on PATH), computes pairwise min cos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.iojson import load_jsonl, write_json, write_jsonl  # noqa: E402
from kws.window_mincos import (  # noqa: E402
    PERCENTILE_GRID,
    anomaly_by_percentile,
    pairwise_min_cos,
    slice_windows,
    window_starts,
)


def load_wav(path: Path, sr: int = 16000) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as e:
        raise SystemExit("pip install soundfile") from e
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        try:
            import librosa

            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        except Exception as e:
            raise SystemExit(f"resample {path} {file_sr}->{sr} failed: {e}") from e
    return np.asarray(wav, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--best-sep", type=Path, default=Path(r"d:\media\pos_neg\best_sep"))
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "window_mincos.jsonl")
    p.add_argument("--summary", type=Path, default=ROOT / "reports" / "window_mincos_summary.json")
    p.add_argument("--percentile", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    idx = args.index or (args.best_sep / "index.jsonl")
    rows = load_jsonl(idx)
    if args.limit:
        rows = rows[: args.limit]
    out = []
    mincos: list[float] = []
    n_skip = 0
    for rec in rows:
        rel = rec.get("dest_rel") or rec.get("dest_wav")
        wav_path = Path(rel) if rel and Path(rel).is_file() else args.best_sep / rec.get("dest_rel", "")
        if not wav_path.is_file():
            out.append({"uid": rec["uid"], "error": f"missing {wav_path}"})
            continue
        wav = load_wav(wav_path)
        spec = window_starts(len(wav), 16000)
        if spec.skipped:
            n_skip += 1
            out.append({"uid": rec["uid"], "skipped": True, "reason": spec.reason, "dur": len(wav) / 16000})
            continue
        # energy proxy embedding when eres is absent: normalized log-mel-ish FFT bins
        windows = slice_windows(wav, spec)
        embs = []
        for w in windows:
            n = 1 << int(np.floor(np.log2(max(256, min(len(w), 1024)))))
            spec_mag = np.abs(np.fft.rfft(w[:n] * np.hanning(n)))
            embs.append(np.log(spec_mag + 1e-8))
        mc = pairwise_min_cos(embs)
        row = {
            "uid": rec["uid"],
            "skipped": False,
            "n_windows": len(windows),
            "min_cos": mc,
            "embedding": "log_rfft_placeholder_not_eres",
            "dur": len(wav) / 16000,
        }
        out.append(row)
        if mc is not None:
            mincos.append(mc)
    n_anom = 0
    if mincos:
        for row in out:
            if row.get("min_cos") is None or row.get("skipped"):
                continue
            row["anomaly_p10"] = anomaly_by_percentile(row["min_cos"], mincos, percentile=args.percentile)
            n_anom += int(row["anomaly_p10"])
    write_jsonl(args.out, out)
    write_json(
        args.summary,
        {
            "n": len(out),
            "n_skip_short": n_skip,
            "n_with_mincos": len(mincos),
            "mincos_p5": float(np.percentile(mincos, 5)) if mincos else None,
            "mincos_p10": float(np.percentile(mincos, 10)) if mincos else None,
            "mincos_p50": float(np.percentile(mincos, 50)) if mincos else None,
            "n_anomaly_at_requested_p": n_anom,
            "percentile": args.percentile,
            "note": (
                "FFT-proxy embeddings are for plumbing only. Lock the percentile "
                "on eres min-cos vs 100 listen labels before using as a gate."
            ),
            "kind": "hypothesis_until_eres_and_labels",
        },
    )
    print(f"wrote {args.out} skip={n_skip} mincos_n={len(mincos)}")


if __name__ == "__main__":
    main()
