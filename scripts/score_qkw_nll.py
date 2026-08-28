#!/usr/bin/env python3
"""Compute frozen Qwen3-ASR target-token NLL for every KWS candidate stream.

Output rows are directly consumable by ``run_kws_eval.py --qkw-jsonl``:
  {"uid": "pos_0", "nll": {"original": 1.23, "spk1": 0.84}}

Raw NLL may rank same-CER streams but is not calibrated q_kw. This script
never enables the absolute-confidence two-speaker reject gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.audio import load_wav_mono  # noqa: E402
from kws.iojson import limit_rows_balanced, load_jsonl, write_json  # noqa: E402
from kws.qkw_nll import Qwen3ASRNLLScorer  # noqa: E402
from kws.wav_paths import resolve_stream_wav  # noqa: E402

SCORER_SCHEMA = "qkw_nll/v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen Qwen3-ASR forced-decode NLL sidecar")
    p.add_argument("--enriched", type=Path, default=ROOT / "reports" / "best_sep_enriched.jsonl")
    p.add_argument("--pos-neg", type=Path, default=Path(r"d:\media\pos_neg"))
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "sidecars" / "q_kw_nll.jsonl")
    p.add_argument("--meta", type=Path, default=ROOT / "reports" / "sidecars" / "q_kw_nll_meta.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=4, help="candidate streams per forward batch")
    p.add_argument("--limit", type=int, default=0, help="balanced UID smoke subset; 0=all")
    p.add_argument("--resume", action="store_true", help="resume an output with the exact same metadata signature")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _model_dir(value: Path | None) -> Path:
    candidates: list[Path] = []
    if value is not None:
        candidates.append(value)
    for key in ("ASR_MODEL_DIR", "QWEN3_ASR_DIR", "VE_ASR_MODEL_DIR"):
        raw = os.environ.get(key, "").strip()
        if raw:
            candidates.append(Path(raw))
    candidates.extend(
        [
            Path("/root/autodl-tmp/Qwen3-ASR-1.7B"),
            Path("/root/Qwen3-ASR-1.7B"),
        ]
    )
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise SystemExit(
        "[ERR] Qwen3-ASR model directory not found. Pass --model-dir or set ASR_MODEL_DIR."
    )


def _signature(args: argparse.Namespace, model_dir: Path, n_input: int) -> dict[str, Any]:
    model_config = model_dir / "config.json"
    return {
        "schema": SCORER_SCHEMA,
        "enriched": str(args.enriched.resolve()),
        "enriched_sha256": _sha256(args.enriched),
        "pos_neg": str(args.pos_neg.resolve()),
        "model_dir": str(model_dir),
        "model_config_sha256": _sha256(model_config) if model_config.is_file() else None,
        "device": str(args.device),
        "dtype": str(args.dtype),
        "limit": int(args.limit),
        "n_input_uid": int(n_input),
        "context": "",
        "language_mode": "wake_lang_forced_text_only",
        "target_mask": "prefix+pad+eos_excluded",
    }


def _load_done(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows = load_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        uid = str(row.get("uid") or "")
        payload = row.get("nll")
        if not uid or not isinstance(payload, dict) or not payload:
            raise SystemExit(f"[ERR] invalid resume row in {path}: uid={uid!r}")
        if uid in seen:
            raise SystemExit(f"[ERR] duplicate uid={uid} in resume output {path}")
        seen.add(uid)
    return rows, seen


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("[ERR] choose only one of --resume or --overwrite")
    if args.batch_size <= 0:
        raise SystemExit("[ERR] --batch-size must be positive")
    if not args.enriched.is_file():
        raise SystemExit(f"[ERR] enriched index not found: {args.enriched}")
    if not args.pos_neg.is_dir():
        raise SystemExit(f"[ERR] pos-neg root not found: {args.pos_neg}")

    rows = load_jsonl(args.enriched)
    if args.limit:
        rows = limit_rows_balanced(rows, args.limit)
    expected = {str(r.get("uid") or "") for r in rows}
    expected_candidates = sum(len(r.get("streams") or {}) for r in rows)
    if "" in expected or len(expected) != len(rows):
        raise SystemExit("[ERR] enriched input has missing or duplicate UIDs")
    model_dir = _model_dir(args.model_dir)
    signature = _signature(args, model_dir, len(rows))

    done_rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if args.out.exists():
        if args.overwrite:
            args.out.unlink()
        elif not args.resume:
            raise SystemExit(f"[ERR] output exists: {args.out}; use --resume or --overwrite")
        else:
            if not args.meta.is_file():
                raise SystemExit(f"[ERR] resume metadata missing: {args.meta}")
            old_meta = json.loads(args.meta.read_text(encoding="utf-8"))
            old_signature = old_meta.get("signature")
            if old_signature != signature:
                raise SystemExit("[ERR] resume signature mismatch; use a new output or --overwrite")
            done_rows, done = _load_done(args.out)
            if not done <= expected:
                raise SystemExit("[ERR] resume output contains UIDs outside the current input scope")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.meta,
        {
            "signature": signature,
            "status": "running",
            "n_done": len(done),
            "note": "Raw NLL ranks same-CER streams only; it is not calibrated q_kw.",
        },
    )
    pending = [r for r in rows if str(r["uid"]) not in done]
    if not pending:
        print(f"[OK] already complete: {args.out} n={len(done)}")
        return 0

    print(
        f"[INFO] load Qwen3-ASR model={model_dir} device={args.device} "
        f"dtype={args.dtype} pending_uid={len(pending)}",
        flush=True,
    )
    scorer = Qwen3ASRNLLScorer(
        str(model_dir),
        device=args.device,
        dtype=args.dtype,
        max_batch_size=args.batch_size,
    )

    mode = "a" if args.resume and args.out.exists() else "w"
    n_candidates = sum(len(r.get("nll") or {}) for r in done_rows)
    with args.out.open(mode, encoding="utf-8") as out_f:
        for row_i, rec in enumerate(pending, start=1):
            uid = str(rec["uid"])
            target = str(rec.get("wake_text") or "").strip()
            lang = str(rec.get("lang") or "").strip()
            streams = rec.get("streams") or {}
            if not target or lang not in {"zh", "en"} or not streams:
                raise SystemExit(
                    f"[ERR] uid={uid}: need non-empty wake_text, lang=zh|en and non-empty streams"
                )

            candidates: list[tuple[str, Any]] = []
            for stream in streams:
                wav_path = resolve_stream_wav(args.pos_neg, rec, stream)
                if wav_path is None:
                    raise SystemExit(
                        f"[ERR] stream wav missing uid={uid} stream={stream} "
                        f"stage={rec.get('best_stage')} under {args.pos_neg}"
                    )
                wav, sr = load_wav_mono(wav_path)
                if sr != 16000 or wav.size == 0:
                    raise SystemExit(f"[ERR] invalid wav uid={uid} stream={stream} sr={sr} n={wav.size}")
                candidates.append((str(stream), wav))

            nll: dict[str, float] = {}
            token_count: dict[str, int] = {}
            for chunk in _chunks(candidates, args.batch_size):
                results = scorer.score_batch(
                    [wav for _, wav in chunk],
                    [target] * len(chunk),
                    [lang] * len(chunk),
                )
                for (stream, _), result in zip(chunk, results):
                    nll[stream] = round(float(result.nll), 8)
                    token_count[stream] = int(result.token_count)
            if set(nll) != set(streams):
                raise SystemExit(f"[ERR] uid={uid}: incomplete stream scores")
            out_row = {
                "uid": uid,
                "nll": nll,
                "token_count": token_count,
                "lang": lang,
            }
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_f.flush()
            done.add(uid)
            n_candidates += len(nll)
            print(
                f"\r[INFO] uid {len(done)}/{len(rows)} candidates={n_candidates} current={uid}",
                end="",
                flush=True,
            )
    print()

    if done != expected:
        missing = sorted(expected - done)
        raise SystemExit(f"[ERR] incomplete UID coverage: missing={len(missing)} head={missing[:10]}")
    final_rows, final_uids = _load_done(args.out)
    if final_uids != expected:
        raise SystemExit("[ERR] final output coverage differs from enriched input")
    total_candidates = sum(len(r["nll"]) for r in final_rows)
    if total_candidates != expected_candidates:
        raise SystemExit(
            f"[ERR] candidate coverage mismatch: scored={total_candidates} "
            f"expected={expected_candidates}"
        )
    write_json(
        args.meta,
        {
            "signature": signature,
            "status": "complete",
            "n_done": len(final_rows),
            "n_candidates": total_candidates,
            "n_expected_candidates": expected_candidates,
            "out": str(args.out.resolve()),
            "note": "Raw NLL ranks same-CER streams only; it is not calibrated q_kw.",
        },
    )
    print(f"[OK] {args.out} uid={len(final_rows)} candidates={total_candidates}")
    print(f"[OK] {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
