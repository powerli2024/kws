"""Materialize named best_sep directories from T0–T4 picks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from .audio import load_wav_mono, save_wav_mono
from .se_backend import apply_se
from .wav_paths import resolve_kws_wav, resolve_stream_wav

# Local ranking groups. Presence / mix gate is a later extract@main step.
GROUP_SPECS = (
    ("raw_kws", "always datasetA kws; no BSS"),
    ("t0", "CER oracle (original wins ties)"),
    ("t2", "L2 cos-to-raw under CER slack"),
    ("skip_then_t0", "orig-unique-zero → original; else T0"),
    ("skip_then_t2", "orig-unique-zero → original; else T2"),
    ("t1_spectral", "T0 + conditional spectral SE (landable backend)"),
    ("t4_spectral", "T0 + always spectral SE (negative control)"),
)


def chosen_stream(rec: Mapping[str, Any], group: str) -> tuple[str, str]:
    """Return (stream_name, reason). stream_name 'raw_kws' means datasetA kws wav."""
    arms = rec.get("arms") or {}
    unique = bool(rec.get("orig_unique_zero") or rec.get("skip_sep_after_scores"))
    t0 = str((arms.get("T0") or {}).get("chosen") or rec.get("oracle_stream") or "original")
    t2 = str((arms.get("T2") or {}).get("chosen") or t0)
    if group == "raw_kws":
        return "raw_kws", "always_raw_kws"
    if group == "t0":
        return t0, "t0_cer_oracle"
    if group == "t2":
        return t2, "t2_l2"
    if group == "skip_then_t0":
        return ("original", "skip_unique_zero") if unique else (t0, "t0_cer_oracle")
    if group == "skip_then_t2":
        return ("original", "skip_unique_zero") if unique else (t2, "t2_l2")
    if group == "t1_spectral":
        return t0, "t0_then_conditional_se"
    if group == "t4_spectral":
        return t0, "t0_then_always_se"
    raise ValueError(group)


def resolve_enroll_wav(
    rec: Mapping[str, Any],
    stream: str,
    *,
    pos_neg: Path,
    data_dir: Path,
) -> Path | None:
    if stream == "raw_kws":
        return resolve_kws_wav(data_dir, rec)
    if stream == "original" and rec.get("orig_unique_zero"):
        # unique-zero: original BSS peak is fine; also accept raw kws
        p = resolve_stream_wav(pos_neg, rec, "original")
        return p or resolve_kws_wav(data_dir, rec)
    return resolve_stream_wav(pos_neg, rec, stream) or (
        resolve_kws_wav(data_dir, rec) if stream == "original" else None
    )


def se_wanted(group: str, rec: Mapping[str, Any], stream: str) -> bool:
    arms = rec.get("arms") or {}
    if group == "t4_spectral":
        return True
    if group == "t1_spectral":
        se = (arms.get("T1") or {}).get("se") or {}
        return bool(se.get("would_apply"))
    return False


def export_one(
    rec: Mapping[str, Any],
    group: str,
    dest_root: Path,
    *,
    pos_neg: Path,
    data_dir: Path,
    se_backend: str = "none",
) -> dict[str, Any]:
    uid = str(rec["uid"])
    split = str(rec.get("split") or "")
    stream, reason = chosen_stream(rec, group)
    src = resolve_enroll_wav(rec, stream, pos_neg=pos_neg, data_dir=data_dir)
    dest_rel = f"{split}/{uid}.wav"
    dest = dest_root / dest_rel
    out: dict[str, Any] = {
        "uid": uid,
        "split": split,
        "id": rec.get("id"),
        "kws_rel": rec.get("kws_rel"),
        "wake_text": rec.get("wake_text"),
        "lang": rec.get("lang"),
        "best_stage": rec.get("best_stage"),
        "oracle_stream": rec.get("oracle_stream"),
        "oracle_cer": rec.get("oracle_cer"),
        "group": group,
        "chosen": stream,
        "reason": reason,
        "dest_rel": dest_rel,
        "ok": False,
    }
    if src is None:
        out["error"] = "missing_src_wav"
        return out
    dest.parent.mkdir(parents=True, exist_ok=True)
    want_se = se_wanted(group, rec, stream) and se_backend not in ("", "none", "off")
    if want_se:
        wav, sr = load_wav_mono(src)
        applied = apply_se(wav, sr, backend=se_backend)
        save_wav_mono(dest, applied["wav"], sr)
        out["se_applied"] = bool(applied.get("se_applied"))
        out["se_reason"] = applied.get("reason")
    else:
        shutil.copy2(src, dest)
        out["se_applied"] = False
    out["ok"] = dest.is_file()
    out["src_wav"] = str(src)
    out["bytes"] = dest.stat().st_size if dest.is_file() else 0
    return out
