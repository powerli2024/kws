#!/usr/bin/env python3
"""Fill the ERes cosine sidecar from wavs that already exist.

Why this was empty: T2 only *reads* a jsonl. Nothing in kws embedded tracks.
Raw KWS is datasetA/{pos,neg}/kws_*.wav (via kws_rel). BSS streams are
pos_neg/{split}/{stage}/wav/{uid}_{tag}.wav (original → peak).

Writes:
  reports/sidecars/cos_to_raw.jsonl   uid + cos_to_raw dict
  reports/sidecars/p_music.jsonl      uid + p_music dict (heuristic B-fallback)

cos(track, raw) is catastrophe, not purity. CMD ranking is eval_cmd_cosine.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.audio import cosine_sim, load_wav_mono  # noqa: E402
from kws.eres import load_embedder  # noqa: E402
from kws.iojson import load_jsonl, write_json, write_jsonl, limit_rows_balanced  # noqa: E402
from kws.residual import p_music_heuristic, snr_med_db  # noqa: E402
from kws.wav_paths import resolve_kws_wav, resolve_stream_wav  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed KWS + BSS streams; write cos sidecar")
    p.add_argument("--enriched", type=Path, default=ROOT / "reports" / "best_sep_enriched.jsonl")
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--data-dir", type=Path, default=Path(r"d:\media\datasetA"))
    p.add_argument("--out-cos", type=Path, default=ROOT / "reports" / "sidecars" / "cos_to_raw.jsonl")
    p.add_argument("--out-pmusic", type=Path, default=ROOT / "reports" / "sidecars" / "p_music.jsonl")
    p.add_argument("--out-meta", type=Path, default=ROOT / "reports" / "sidecars" / "build_meta.json")
    p.add_argument("--backend", default="eres2netv2", help="eres2netv2 | campplus | fft")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--extract-ve", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--allow-missing", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.enriched)
    if args.limit:
        rows = limit_rows_balanced(rows, args.limit)
    print(f"[INFO] load embedder backend={args.backend}", flush=True)
    enc = load_embedder(
        args.backend,
        model_dir=args.model_dir,
        device=args.device,
        extract_ve=args.extract_ve,
    )
    print(f"[INFO] encoder={enc.name} n={len(rows)}", flush=True)

    cos_rows: list[dict] = []
    pm_rows: list[dict] = []
    n_miss_kws = n_miss_stream = n_ok = 0
    missing: list[dict] = []

    for i, rec in enumerate(rows):
        uid = str(rec["uid"])
        streams = rec.get("streams") or {}
        kws_path = resolve_kws_wav(args.data_dir, rec)
        if kws_path is None:
            n_miss_kws += 1
            missing.append({"uid": uid, "error": "missing_kws", "kws_rel": rec.get("kws_rel")})
            if not args.allow_missing:
                raise SystemExit(
                    f"[ERR] raw KWS missing for {uid}. "
                    f"Expected {{data_dir}}/{rec.get('kws_rel')} under {args.data_dir}. "
                    "Pass --data-dir at the datasetA root (not the AutoDL kws_path)."
                )
            continue
        kws, sr = load_wav_mono(kws_path)
        e_raw = enc.embed(kws, sr)
        cos: dict[str, float] = {}
        pm: dict[str, float] = {}
        snrs: dict[str, float] = {}
        row_miss = False
        for name in streams:
            sp = resolve_stream_wav(args.pos_neg, rec, name)
            if sp is None:
                n_miss_stream += 1
                row_miss = True
                missing.append(
                    {
                        "uid": uid,
                        "error": "missing_stream",
                        "stream": name,
                        "stage": rec.get("best_stage"),
                    }
                )
                if not args.allow_missing:
                    raise SystemExit(
                        f"[ERR] stream wav missing uid={uid} stream={name} "
                        f"stage={rec.get('best_stage')} under {args.pos_neg}"
                    )
                continue
            wav, ssr = load_wav_mono(sp)
            emb = enc.embed(wav, ssr)
            c = float(cosine_sim(emb, e_raw))
            cos[name] = max(-1.0, min(1.0, c))
            pm[name] = float(p_music_heuristic(wav, ssr))
            snrs[name] = float(snr_med_db(wav, ssr))
        if row_miss and not args.allow_missing:
            continue
        if not cos:
            continue
        cos_rows.append({"uid": uid, "cos_to_raw": cos})
        pm_rows.append({"uid": uid, "p_music": pm, "snr_med_db": snrs})
        n_ok += 1
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[INFO] {i + 1}/{len(rows)} ok={n_ok}", flush=True)

    write_jsonl(args.out_cos, cos_rows)
    write_jsonl(args.out_pmusic, [{"uid": r["uid"], "p_music": r["p_music"]} for r in pm_rows])
    write_json(
        args.out_meta,
        {
            "encoder": enc.name,
            "backend": args.backend,
            "n_input": len(rows),
            "n_ok": n_ok,
            "n_missing_kws": n_miss_kws,
            "n_missing_stream": n_miss_stream,
            "out_cos": str(args.out_cos),
            "out_pmusic": str(args.out_pmusic),
            "note": (
                "cos_to_raw is catastrophe vs raw KWS, not a purity score. "
                "p_music is residual.p_music_heuristic until YAMNet is calibrated."
            ),
            "missing_head": missing[:20],
        },
    )
    print(
        json.dumps(
            {"n_ok": n_ok, "n_missing_kws": n_miss_kws, "n_missing_stream": n_miss_stream},
            ensure_ascii=False,
        )
    )
    print(f"[OK] {args.out_cos}")
    print(f"[OK] {args.out_pmusic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
