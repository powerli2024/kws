#!/usr/bin/env python3
"""Recompute and evaluate an independent s1 -> s7 route with optional SE views.

The script never overwrites extract-sep audio.  Every raw/SE waveform is scored
once per (audio SHA256, registration text, language), so copied stage audio
cannot acquire different CER merely because it appeared in another directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kws.asr_transcribe import Qwen3ASRTranscriber  # noqa: E402
from kws.audio import cosine_sim, load_wav_mono, save_wav_mono  # noqa: E402
from kws.cer_metric import cer_detail  # noqa: E402
from kws.eres import load_embedder  # noqa: E402
from kws.iojson import load_jsonl, write_json, write_jsonl  # noqa: E402
from kws.need_se import se_safety_ok  # noqa: E402
from kws.qkw_nll import Qwen3ASRNLLScorer  # noqa: E402
from kws.se_backend import spectral_subtract  # noqa: E402
from kws.se_route import choose, paired, route_one, summarize  # noqa: E402
from kws.stage_compare import StageArm, discover_stage_arms  # noqa: E402

SCHEMA = "kws_se_route/v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full raw/SE ASR recompute and s1->s7 evaluation")
    p.add_argument("--pos-neg", type=Path, required=True, help="extract-sep root containing pos/ and neg/")
    p.add_argument("--work-dir", type=Path, default=ROOT / "reports" / "se_route")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context-mode", choices=("wake", "none"), default="wake")
    p.add_argument("--s1-arm", default="s1_onnx_full")
    p.add_argument("--s7-arm", default="auto", help="exact discovered label; auto is development-only")
    p.add_argument("--splits", default="pos,neg")
    p.add_argument("--expected-uids", type=int, default=1838, help="0 disables exact assertion")
    p.add_argument("--trigger-cer", type=float, default=0.0)
    p.add_argument("--se-backend", choices=("spectral", "command", "precomputed"), default="spectral")
    p.add_argument(
        "--se-command",
        default=None,
        help="command template containing {input} and {output}; argv parsing, no shell expansion",
    )
    p.add_argument(
        "--se-batch-command",
        default=None,
        help="faster command template containing {manifest}; wrapper loads the model once",
    )
    p.add_argument("--precomputed-se-dir", type=Path, default=None)
    p.add_argument("--se-cos-thr", type=float, default=0.92)
    p.add_argument("--se-cer-slack", type=float, default=0.05)
    p.add_argument("--speaker-backend", default="eres2netv2", help="eres2netv2|cam++|fft_proxy|none")
    p.add_argument("--speaker-model-dir", type=Path, default=None)
    p.add_argument("--with-nll", action="store_true", help="add frozen target NLL as same-CER L2 rank")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite-scores", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug only; forbidden with expected-uids")
    p.add_argument("--allow-no-speaker-gate", action="store_true", help="debug only; never production")
    p.add_argument("--export-best", action="store_true", help="materialize proposed s1_s7_se tree")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def wav_path(arm: StageArm, uid: str, stream: str) -> Path:
    tag = "peak" if stream == "original" else stream
    return (arm.index_path.parent / "wav" / f"{uid}_{tag}.wav").resolve()


def select_arm(arms: list[StageArm], label: str, prefix: str) -> StageArm:
    exact = [arm for arm in arms if arm.label == label]
    if exact:
        return exact[0]
    if label != "auto":
        known = [arm.label for arm in arms if arm.label.startswith(prefix)]
        raise SystemExit(f"[ERR] arm {label!r} not found; matching labels={known}")
    matches = [arm for arm in arms if arm.label.startswith(prefix)]
    if not matches:
        raise SystemExit(f"[ERR] no {prefix} arm discovered")

    def key(arm: StageArm) -> tuple[Any, ...]:
        values = [float(row["oracle_cer"]) for row in arm.rows]
        mean = sum(values) / len(values) if values else float("inf")
        return (-len(values), mean, arm.label)

    return min(matches, key=key)


def inventory(args: argparse.Namespace, splits: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    selected: dict[str, Any] = {"splits": {}}
    arms_by_split = {split: discover_stage_arms(args.pos_neg, split) for split in splits}
    resolved_s7 = args.s7_arm
    if resolved_s7 == "auto":
        common: set[str] | None = None
        for split in splits:
            labels = {arm.label for arm in arms_by_split[split] if arm.label.startswith("s7")}
            common = labels if common is None else common & labels
        if not common:
            raise SystemExit("[ERR] no common s7 arm label exists across requested splits")

        def global_key(label: str) -> tuple[Any, ...]:
            chosen = [select_arm(arms_by_split[split], label, "s7") for split in splits]
            values = [float(row["oracle_cer"]) for arm in chosen for row in arm.rows]
            return (-len(values), sum(values) / len(values) if values else float("inf"), label)

        resolved_s7 = min(common, key=global_key)
    selected["s7_requested"] = args.s7_arm
    selected["s7_locked_label"] = resolved_s7
    for split in splits:
        arms = arms_by_split[split]
        s1 = select_arm(arms, args.s1_arm, "s1")
        s7 = select_arm(arms, resolved_s7, "s7")
        selected["splits"][split] = {
            "s1": s1.label,
            "s1_index": str(s1.index_path),
            "s7": s7.label,
            "s7_index": str(s7.index_path),
            "s7_auto_selected": args.s7_arm == "auto",
        }
        for role, arm in (("s1", s1), ("s7", s7)):
            for row in arm.rows:
                uid = str(row["uid"])
                for stream in sorted((row.get("streams") or {}).keys()):
                    path = wav_path(arm, uid, str(stream))
                    if not path.is_file():
                        raise SystemExit(f"[ERR] missing WAV uid={uid} arm={arm.label} stream={stream}: {path}")
                    out.append({
                        "uid": uid,
                        "split": split,
                        "wake_text": str(row.get("wake_text") or "").strip(),
                        "lang": str(row.get("lang") or "").strip(),
                        "role": role,
                        "arm": arm.label,
                        "stream": str(stream),
                        "view": "raw",
                        "wav": str(path),
                        "audio_sha256": sha256_file(path),
                    })
    if any(not row["wake_text"] or row["lang"] not in {"zh", "en"} for row in out):
        raise SystemExit("[ERR] every candidate needs wake_text and lang=zh|en")
    return out, selected


def run_se(args: argparse.Namespace, raw: dict[str, Any], dest: Path) -> None:
    src = Path(raw["wav"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if args.se_backend == "spectral":
        wav, sr = load_wav_mono(src)
        save_wav_mono(dest, spectral_subtract(wav, sr), sr)
    elif args.se_backend == "command":
        if args.se_batch_command:
            raise RuntimeError("batch command output was missing after batch execution")
        if not args.se_command or "{input}" not in args.se_command or "{output}" not in args.se_command:
            raise SystemExit("[ERR] command backend needs --se-command with {input} and {output}")
        rendered = args.se_command.format(input=str(src), output=str(dest))
        command = shlex.split(rendered, posix=os.name != "nt")
        subprocess.run(command, check=True)
    else:
        if args.precomputed_se_dir is None:
            raise SystemExit("[ERR] precomputed backend needs --precomputed-se-dir")
        name = f"{raw['uid']}__{raw['role']}__{safe_name(raw['arm'])}__{raw['stream']}.wav"
        prepared = args.precomputed_se_dir / raw["split"] / name
        if not prepared.is_file():
            raise SystemExit(f"[ERR] missing precomputed SE WAV: {prepared}")
        shutil.copy2(prepared, dest)
    validate_se_pair(args, src, dest)


def validate_se_pair(args: argparse.Namespace, src: Path, dest: Path) -> None:
    if not dest.is_file() or dest.stat().st_size <= 44:
        raise SystemExit(f"[ERR] SE backend did not create a valid WAV: {dest}")
    # Canonicalize every external backend to the KWS contract before hashing.
    if args.se_backend != "spectral":
        normalized, _ = load_wav_mono(dest, sr=16000)
        save_wav_mono(dest, normalized, 16000)
    before, sr0 = load_wav_mono(src)
    after, sr1 = load_wav_mono(dest)
    tolerance = max(1600, int(round(before.size * 0.02)))
    if sr0 != sr1 or after.size == 0 or abs(after.size - before.size) > tolerance:
        raise SystemExit(
            f"[ERR] SE changed duration/sample contract: {src} n={before.size}, {dest} n={after.size}"
        )


def add_se_views(args: argparse.Namespace, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    se_root = args.work_dir / "se_wav"
    meta_path = args.work_dir / "se_wav.meta.json"
    se_signature = {
        "schema": SCHEMA,
        "backend": args.se_backend,
        "command": args.se_command,
        "batch_command": args.se_batch_command,
        "precomputed_se_dir": str(args.precomputed_se_dir.resolve()) if args.precomputed_se_dir else None,
        "raw_inventory_sha256": sha256_json(sorted(
            (row["audio_sha256"], row["wake_text"], row["lang"]) for row in raw_rows
        )),
    }
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("signature") != se_signature:
            raise SystemExit(
                "[ERR] SE cache signature mismatch; choose a new --work-dir so different backends are not mixed"
            )
    write_json(meta_path, {"status": "running", "signature": se_signature})
    unique: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        unique.setdefault(str(row["audio_sha256"]), row)
    if args.se_backend == "command" and args.se_batch_command:
        if "{manifest}" not in args.se_batch_command:
            raise SystemExit("[ERR] --se-batch-command must contain {manifest}")
        manifest_rows = []
        for raw_hash, row in sorted(unique.items()):
            dest = se_root / raw_hash[:2] / f"{raw_hash}.wav"
            if not dest.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                manifest_rows.append({
                    "audio_sha256": raw_hash, "input": row["wav"], "output": str(dest),
                    "sample_rate": 16000, "length_policy": "full_waveform",
                })
        if manifest_rows:
            manifest = args.work_dir / "se_batch_manifest.jsonl"
            write_jsonl(manifest, manifest_rows)
            command = shlex.split(
                args.se_batch_command.format(manifest=str(manifest)), posix=os.name != "nt"
            )
            print(f"[SE] batch command files={len(manifest_rows)}", flush=True)
            subprocess.run(command, check=True)
    se_by_raw_hash: dict[str, tuple[Path, str]] = {}
    for index, (raw_hash, row) in enumerate(sorted(unique.items()), start=1):
        dest = se_root / raw_hash[:2] / f"{raw_hash}.wav"
        if not dest.is_file():
            run_se(args, row, dest)
        else:
            validate_se_pair(args, Path(row["wav"]), dest)
        se_by_raw_hash[raw_hash] = (dest.resolve(), sha256_file(dest))
        print(f"\r[SE] {index}/{len(unique)} hit_or_fresh={dest.name}", end="", flush=True)
    print()
    rows = list(raw_rows)
    for raw in raw_rows:
        path, digest = se_by_raw_hash[str(raw["audio_sha256"])]
        se = dict(raw)
        se.update({
            "view": "se",
            "wav": str(path),
            "audio_sha256": digest,
            "pre_audio_sha256": raw["audio_sha256"],
            "se_backend": args.se_backend,
        })
        rows.append(se)
    write_json(meta_path, {
        "status": "complete", "signature": se_signature,
        "n_raw_refs": len(raw_rows), "n_unique_raw_audio": len(unique),
    })
    return rows


def score_key(row: dict[str, Any]) -> str:
    return sha256_json([row["audio_sha256"], row["wake_text"], row["lang"]])


def load_score_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = str(row.get("score_key") or "")
        if not key or key in out:
            raise SystemExit(f"[ERR] invalid/duplicate score cache key in {path}: {key!r}")
        out[key] = row
    return out


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def recompute_scores(args: argparse.Namespace, rows: list[dict[str, Any]], signature: dict[str, Any]) -> dict[str, dict[str, Any]]:
    score_path = args.work_dir / "audio_scores.jsonl"
    meta_path = args.work_dir / "audio_scores.meta.json"
    if args.overwrite_scores and score_path.exists():
        score_path.unlink()
    cache = load_score_cache(score_path) if args.resume else {}
    if cache:
        if not meta_path.is_file() or json.loads(meta_path.read_text(encoding="utf-8")).get("signature") != signature:
            raise SystemExit("[ERR] score resume signature mismatch; use --overwrite-scores")
    elif score_path.exists() and not args.overwrite_scores:
        raise SystemExit(f"[ERR] score cache exists: {score_path}; use --resume or --overwrite-scores")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    write_json(meta_path, {"schema": SCHEMA, "status": "running", "signature": signature})

    unique = {}
    for row in rows:
        row["score_key"] = score_key(row)
        unique.setdefault(row["score_key"], row)
    pending_asr = [row for key, row in unique.items() if key not in cache or "cer" not in cache[key]]
    if pending_asr:
        print(f"[ASR] load model={args.model_dir} pending_unique={len(pending_asr)}", flush=True)
        asr = Qwen3ASRTranscriber(
            args.model_dir, device=args.device, dtype=args.dtype, max_batch_size=args.batch_size
        )
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in pending_asr:
            groups[(row["wake_text"], row["lang"])].append(row)
        done = 0
        with score_path.open("a", encoding="utf-8") as handle:
            for (wake, lang), items in groups.items():
                for start in range(0, len(items), args.batch_size):
                    chunk = items[start : start + args.batch_size]
                    wavs = [load_wav_mono(Path(row["wav"]))[0] for row in chunk]
                    hyps = asr.transcribe_many(
                        wavs, language=lang, wake_text=wake, context_mode=args.context_mode
                    )
                    for row, hyp in zip(chunk, hyps):
                        detail = cer_detail(hyp, wake)
                        scored = {
                            "score_key": row["score_key"],
                            "audio_sha256": row["audio_sha256"],
                            "wake_text": wake,
                            "lang": lang,
                            **detail,
                        }
                        cache[row["score_key"]] = scored
                        handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
                    handle.flush()
                    done += len(chunk)
                    print(f"\r[ASR] {done}/{len(pending_asr)}", end="", flush=True)
        print()
        del asr
        release_cuda()

    if args.with_nll:
        pending_nll = [row for key, row in unique.items() if cache.get(key, {}).get("nll") is None]
        if pending_nll:
            print(f"[NLL] load model={args.model_dir} pending_unique={len(pending_nll)}", flush=True)
            scorer = Qwen3ASRNLLScorer(
                str(args.model_dir), device=args.device, dtype=args.dtype, max_batch_size=args.batch_size
            )
            updates: dict[str, tuple[float, int]] = {}
            for start in range(0, len(pending_nll), args.batch_size):
                chunk = pending_nll[start : start + args.batch_size]
                results = scorer.score_batch(
                    [load_wav_mono(Path(row["wav"]))[0] for row in chunk],
                    [row["wake_text"] for row in chunk],
                    [row["lang"] for row in chunk],
                )
                for row, result in zip(chunk, results):
                    updates[row["score_key"]] = (float(result.nll), int(result.token_count))
                print(f"\r[NLL] {min(start + len(chunk), len(pending_nll))}/{len(pending_nll)}", end="", flush=True)
            print()
            del scorer
            release_cuda()
            # Rewrite atomically so resume has one row per score key.
            for key, (nll, token_count) in updates.items():
                cache[key]["nll"] = nll
                cache[key]["token_count"] = token_count
            tmp = score_path.with_suffix(".jsonl.tmp")
            write_jsonl(tmp, [cache[key] for key in sorted(cache)])
            os.replace(tmp, score_path)

    expected = set(unique)
    if set(cache) != expected:
        raise SystemExit(f"[ERR] score coverage mismatch cache={len(cache)} expected={len(expected)}")
    write_json(meta_path, {
        "schema": SCHEMA,
        "status": "complete",
        "signature": signature,
        "n_unique_score_key": len(cache),
        "n_candidate_refs": len(rows),
        "cache_reuse_refs": len(rows) - len(cache),
    })
    return cache


def add_speaker_safety(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    raw_by_identity = {
        (row["uid"], row["role"], row["arm"], row["stream"]): row
        for row in rows if row["view"] == "raw"
    }
    if args.speaker_backend == "none":
        if not args.allow_no_speaker_gate:
            raise SystemExit("[ERR] speaker gate is mandatory; configure ERes or use debug-only --allow-no-speaker-gate")
        for row in rows:
            if row["view"] == "se":
                row["cos_se_pre"] = None
                row["se_eligible"] = True
                row["se_safety_reason"] = "speaker_gate_bypassed_debug_only"
        return
    encoder = load_embedder(args.speaker_backend, model_dir=args.speaker_model_dir, device=args.device)
    embeddings: dict[str, Any] = {}
    audio_rows = {str(row["audio_sha256"]): row for row in rows}
    for index, (digest, row) in enumerate(sorted(audio_rows.items()), start=1):
        wav, sr = load_wav_mono(Path(row["wav"]))
        embeddings[digest] = encoder.embed(wav, sr)
        print(f"\r[SPK] {index}/{len(audio_rows)}", end="", flush=True)
    print()
    for row in rows:
        if row["view"] != "se":
            continue
        identity = (row["uid"], row["role"], row["arm"], row["stream"])
        raw = raw_by_identity[identity]
        cos = float(cosine_sim(embeddings[row["audio_sha256"]], embeddings[raw["audio_sha256"]]))
        ok, reason = se_safety_ok(
            cos_se_pre=cos,
            cer_se=float(row["cer"]),
            cer_pre=float(raw["cer"]),
            cos_thr=args.se_cos_thr,
            cer_slack=args.se_cer_slack,
        )
        row["cos_se_pre"] = cos
        row["se_eligible"] = bool(ok)
        row["se_safety_reason"] = reason
    del encoder
    release_cuda()


def one_stage(uid: str, candidates: list[dict[str, Any]], *, allow_se: bool) -> dict[str, Any]:
    rows = [row for row in candidates if row["role"] == "s1" and (allow_se or row["view"] == "raw")]
    rows = [row for row in rows if row["view"] == "raw" or row.get("se_eligible")]
    selected = choose(rows)
    return {"uid": uid, "ok": selected is not None, "selected": selected, "reason": "s1_only"}


def build_report(args: argparse.Namespace, rows: list[dict[str, Any]], selected_arms: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[row["uid"]].append(row)
    decisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for uid in sorted(by_uid):
        candidates = by_uid[uid]
        decisions["s1_raw"].append(one_stage(uid, candidates, allow_se=False))
        decisions["s1_raw_plus_safe_se"].append(one_stage(uid, candidates, allow_se=True))
        raw_route = route_one(candidates, allow_se=False, trigger_cer=args.trigger_cer)
        raw_route["uid"] = uid
        decisions["s1_to_s7_raw"].append(raw_route)
        se_route = route_one(candidates, allow_se=True, trigger_cer=args.trigger_cer)
        se_route["uid"] = uid
        decisions["s1_to_s7_safe_se"].append(se_route)
        global_se = choose([row for row in candidates if row["role"] == "s1" and row["view"] == "se"])
        decisions["always_se_s1_negative_control"].append({
            "uid": uid, "ok": global_se is not None, "selected": global_se, "reason": "always_se_no_safety"
        })
    metrics = {name: summarize(values) for name, values in decisions.items()}
    baseline, proposed = decisions["s1_to_s7_raw"], decisions["s1_to_s7_safe_se"]
    comparison = paired(baseline, proposed)
    expected = args.expected_uids or len(by_uid)
    local_pass = (
        metrics["s1_to_s7_safe_se"]["n_ok"] == expected
        and comparison["n_worsened"] == 0
        and metrics["s1_to_s7_safe_se"]["mean_cer"] is not None
        and metrics["s1_to_s7_safe_se"]["mean_cer"] <= 0.03
        and metrics["s1_to_s7_safe_se"]["cer0_rate"]
        >= metrics["s1_to_s7_raw"]["cer0_rate"] - 0.02
    )
    se_rows = [row for row in rows if row["view"] == "se"]
    report = {
        "schema": SCHEMA,
        "status": "LOCAL_PASS_NEEDS_CMD_PRESENCE" if local_pass else "NO_GO",
        "production_approved": False,
        "selected_arms": selected_arms,
        "policy": {
            "priority": "s1",
            "s7_trigger": f"selected s1 CER > {args.trigger_cer}",
            "s7_switch": "strict CER improvement; target NLL may break exact-CER ties",
            "se_gate": f"cos(se,pre)>={args.se_cos_thr} and CER_se<=CER_pre+{args.se_cer_slack}",
            "zh_metric": "toneless_pinyin_cer",
            "en_metric": "normalized_character_cer",
            "same_audio_score": "once_per_sha256+wake_text+lang",
        },
        "coverage": {
            "n_uid": len(by_uid),
            "expected_uid": expected,
            "n_candidate_refs": len(rows),
            "n_unique_audio": len({row["audio_sha256"] for row in rows}),
        },
        "se": {
            "backend": args.se_backend,
            "n_views": len(se_rows),
            "n_eligible": sum(bool(row.get("se_eligible")) for row in se_rows),
            "n_cos_reject": sum(row.get("se_safety_reason") == "cos_collapse" for row in se_rows),
            "n_cer_reject": sum(row.get("se_safety_reason") == "cer_regression" for row in se_rows),
        },
        "arms": metrics,
        "proposed_vs_raw_route": comparison,
        "go_no_go": {
            "local_pass": local_pass,
            "requirements": [
                "full UID coverage",
                "no paired CER regression against s1->s7 raw",
                "mean pinyin/character CER <= 0.03",
                "CER0 drop <= 2 percentage points",
                "then frozen CMD FRR/FAR and extract Presence/contest validation",
            ],
        },
    }
    return report, decisions


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# s1 → s7 → SE recompute report", "",
        f"- status: **{report['status']}**",
        f"- production approved: `{report['production_approved']}`",
        f"- UID coverage: `{report['coverage']['n_uid']}/{report['coverage']['expected_uid']}`",
        f"- unique audio / refs: `{report['coverage']['n_unique_audio']}/{report['coverage']['n_candidate_refs']}`",
        f"- SE eligible: `{report['se']['n_eligible']}/{report['se']['n_views']}`", "",
        "| arm | n | mean CER | CER0 | s7 trigger | s7 switch |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["arms"].items():
        lines.append(
            f"| {name} | {row['n_ok']} | {row['mean_cer']} | {row['cer0_rate']} | "
            f"{row['n_triggered_s7']} | {row['n_switched_s7']} |"
        )
    pair = report["proposed_vs_raw_route"]
    lines += [
        "", "## Paired proposed-vs-raw result", "",
        f"- improved / worsened / same: `{pair['n_improved']} / {pair['n_worsened']} / {pair['n_same']}`",
        f"- mean CER delta: `{pair['mean_delta']}`", "",
        "`LOCAL_PASS_NEEDS_CMD_PRESENCE` is only KWS-local evidence. Production adoption still",
        "requires the frozen CMD FRR/FAR evaluation and downstream extract Presence/contest veto.", "",
    ]
    return "\n".join(lines)


def export_route(args: argparse.Namespace, decisions: list[dict[str, Any]], dirname: str) -> None:
    root = args.work_dir / dirname
    index = []
    for row in decisions:
        if not row.get("ok"):
            raise SystemExit(f"[ERR] cannot export missing decision uid={row.get('uid')}")
        selected = row["selected"]
        dest = root / selected["split"] / f"{row['uid']}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected["wav"], dest)
        index.append({
            "uid": row["uid"], "split": selected["split"], "dest_rel": str(dest.relative_to(root)),
            "ok": True, "chosen_role": selected["role"], "chosen_arm": selected["arm"],
            "chosen_stream": selected["stream"], "chosen_view": selected["view"],
            "cer": selected["cer"], "nll": selected.get("nll"),
            "cos_se_pre": selected.get("cos_se_pre"), "src_wav": selected["wav"],
        })
    write_jsonl(root / "index.jsonl", index)


def model_dir(args: argparse.Namespace) -> Path:
    values = [args.model_dir]
    values.extend(Path(value) for key in ("ASR_MODEL_DIR", "QWEN3_ASR_DIR") if (value := os.environ.get(key)))
    values.extend([Path("/root/autodl-tmp/Qwen3-ASR-1.7B"), Path("/root/Qwen3-ASR-1.7B")])
    for path in values:
        if path is not None and Path(path).is_dir():
            return Path(path).resolve()
    raise SystemExit("[ERR] Qwen3-ASR weights missing; pass --model-dir or set ASR_MODEL_DIR")


def main() -> int:
    args = parse_args()
    if args.limit and args.expected_uids:
        raise SystemExit("[ERR] --limit is debug-only; set --expected-uids 0 explicitly")
    if args.resume and args.overwrite_scores:
        raise SystemExit("[ERR] choose only --resume or --overwrite-scores")
    args.model_dir = model_dir(args)
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    raw_rows, selected_arms = inventory(args, splits)
    uids = sorted({row["uid"] for row in raw_rows})
    if args.limit:
        keep = set(uids[: args.limit])
        raw_rows = [row for row in raw_rows if row["uid"] in keep]
        uids = sorted(keep)
    if args.expected_uids and len(uids) != args.expected_uids:
        raise SystemExit(f"[ERR] UID coverage={len(uids)} expected={args.expected_uids}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    all_rows = add_se_views(args, raw_rows)
    signature = {
        "schema": SCHEMA,
        "inventory_sha256": sha256_json(sorted(
            (row["uid"], row["role"], row["arm"], row["stream"], row["view"], row["audio_sha256"])
            for row in all_rows
        )),
        "model_dir": str(args.model_dir),
        "model_config_sha256": sha256_file(args.model_dir / "config.json") if (args.model_dir / "config.json").is_file() else None,
        "context_mode": args.context_mode,
        "with_nll": bool(args.with_nll),
    }
    scores = recompute_scores(args, all_rows, signature)
    for row in all_rows:
        score = scores[row["score_key"]]
        for key in ("hyp", "cer", "cer_char", "cer_py", "metric", "nll", "token_count"):
            if key in score:
                row[key] = score[key]
    add_speaker_safety(args, all_rows)
    write_jsonl(args.work_dir / "candidates_scored.jsonl", all_rows)
    report, decisions = build_report(args, all_rows, selected_arms)
    write_json(args.work_dir / "report.json", report)
    (args.work_dir / "report.md").write_text(render_md(report), encoding="utf-8")
    for name, values in decisions.items():
        write_jsonl(args.work_dir / f"decisions_{name}.jsonl", values)
    if args.export_best:
        export_route(args, decisions["s1_to_s7_raw"], "best_sep_s1_to_s7_raw")
        export_route(args, decisions["s1_to_s7_safe_se"], "best_sep_s1_s7_safe_se")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] report: {args.work_dir / 'report.md'}")
    return 0 if report["go_no_go"]["local_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
