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
    ("e0_raw", "E0: always datasetA kws"),
    ("e1_t0", "E1: CER oracle (original wins ties)"),
    ("e2_qkw", "E2: q_kw rank + cos-to-raw catastrophe gate"),
    ("oracle_cmd", "offline max cos(enroll,cmd); not deployable"),
    ("raw_kws", "alias of e0_raw"),
    ("t0", "alias of e1_t0"),
    ("t2", "alias of e2_qkw"),
    ("skip_then_t0", "orig-unique-zero → original; else T0"),
    ("skip_then_t2", "orig-unique-zero → original; else T2"),
    ("t1_spectral", "T0 + conditional spectral SE"),
    ("t4_spectral", "E6: always spectral SE (negative control)"),
)


def chosen_stream(rec: Mapping[str, Any], group: str) -> tuple[str, str]:
    """Return (stream_name, reason). stream_name 'raw_kws' means datasetA kws wav."""
    arms = rec.get("arms") or {}
    unique = bool(rec.get("orig_unique_zero") or rec.get("skip_sep_after_scores"))
    t0 = str((arms.get("T0") or {}).get("chosen") or rec.get("oracle_stream") or "original")
    t2 = str((arms.get("T2") or {}).get("chosen") or t0)
    if group in ("raw_kws", "e0_raw"):
        return "raw_kws", "always_raw_kws"
    if group in ("t0", "e1_t0"):
        return t0, "t0_cer_oracle"
    if group in ("t2", "e2_qkw"):
        return t2, "e2_qkw"
    if group == "oracle_cmd":
        name = str(rec.get("oracle_cmd_stream") or "")
        return (name, "cmd_label_oracle") if name else ("reject", "oracle_cmd_missing")
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
    if stream == "reject":
        out["error"] = "rejected_enroll"
        return out
    src = resolve_enroll_wav(rec, stream, pos_neg=pos_neg, data_dir=data_dir)
    if src is None:
        out["error"] = "missing_src_wav"
        return out
    dest.parent.mkdir(parents=True, exist_ok=True)
    want_se = se_wanted(group, rec, stream) and se_backend not in ("", "none", "off")
    if want_se:
        wav, sr = load_wav_mono(src)
        encoder = rec.get("_encoder")
        if encoder is None:
            shutil.copy2(src, dest)
            out["se_applied"] = False
            out["se_reason"] = "se_safety_encoder_missing_refused"
        else:
            from .audio import cosine_sim
            from .need_se import se_safety_ok

            y = apply_se(wav, sr, backend=se_backend)["wav"]
            cos = float(cosine_sim(encoder.embed(y, sr), encoder.embed(wav, sr)))
            ok, why = se_safety_ok(cos_se_pre=cos, cer_se=0.0, cer_pre=0.0)
            if not ok:
                shutil.copy2(src, dest)
                out["se_applied"] = False
                out["se_reason"] = f"se_safety_{why}"
                out["cos_se_pre"] = cos
            else:
                save_wav_mono(dest, y, sr)
                out["se_applied"] = True
                out["se_reason"] = "spectral_subtract"
                out["cos_se_pre"] = cos
    else:
        shutil.copy2(src, dest)
        out["se_applied"] = False
    out["ok"] = dest.is_file()
    out["src_wav"] = str(src)
    out["bytes"] = dest.stat().st_size if dest.is_file() else 0
    return out
