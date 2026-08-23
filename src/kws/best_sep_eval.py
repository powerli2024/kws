"""Compare best_sep dumps. CER is a constraint; downstream Presence is the veto.

Do not rank directories by mean oracle CER — most winners are already 0.
"""

from __future__ import annotations

import hashlib
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .presence_protocol import CER0_DROP_MAX, CER_MEAN_MAX, enroll_go
from .cmd_eval import rank_groups as rank_cmd_groups
from .residual import p_music_heuristic, snr_med_db


def resolve_wav(best_sep: Path, rec: dict[str, Any]) -> Path | None:
    rel = str(rec.get("dest_rel") or "").replace("\\", "/").lstrip("/")
    if rel:
        p = (best_sep / rel)
        if p.is_file():
            return p.resolve()
    dest = str(rec.get("dest_wav") or "")
    if dest and Path(dest).is_file():
        return Path(dest).resolve()
    split = str(rec.get("split") or "")
    uid = str(rec.get("uid") or "")
    if split and uid:
        p = best_sep / split / f"{uid}.wav"
        if p.is_file():
            return p.resolve()
    return None


def load_wav_mono(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        import soundfile as sf

        x, sr = sf.read(str(path), always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        return x.reshape(-1), int(sr)
    except Exception:
        pass
    try:
        with wave.open(str(path), "rb") as w:
            sr = int(w.getframerate())
            nch = int(w.getnchannels())
            sw = int(w.getsampwidth())
            raw = w.readframes(w.getnframes())
        if sw == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return None
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        return x.reshape(-1), sr
    except Exception:
        return None


def _wav_fingerprint(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(65536))
        h.update(str(path.stat().st_size).encode())
    return h.hexdigest()


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    a = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(a, p))


def summarize_best_sep(best_sep: Path, *, max_audio: int = 0) -> dict[str, Any]:
    """Layer 0–2: inventory, CER constraint stats, acoustic proxies."""
    best_sep = Path(best_sep)
    idx = best_sep / "index.jsonl"
    rows: list[dict[str, Any]] = []
    if idx.is_file():
        from .iojson import load_jsonl

        rows = load_jsonl(idx)
    else:
        for split in ("pos", "neg"):
            d = best_sep / split
            if not d.is_dir():
                continue
            for wav in sorted(d.glob("*.wav")):
                rows.append(
                    {
                        "uid": wav.stem,
                        "split": split,
                        "dest_rel": f"{split}/{wav.name}",
                        "ok": True,
                    }
                )

    cers: list[float] = []
    durs: list[float] = []
    snrs: list[float] = []
    pmuse: list[float] = []
    stages: Counter[str] = Counter()
    streams: Counter[str] = Counter()
    n_ok = n_missing = n_orig = 0
    n_audio = 0
    uids: dict[str, dict[str, Any]] = {}

    for rec in rows:
        if rec.get("ok") is False:
            continue
        uid = str(rec.get("uid") or "")
        if not uid:
            continue
        wav = resolve_wav(best_sep, rec)
        meta = {
            "uid": uid,
            "split": rec.get("split"),
            "oracle_cer": rec.get("oracle_cer"),
            "oracle_stream": rec.get("oracle_stream"),
            "best_stage": rec.get("best_stage"),
            "wav": str(wav) if wav else None,
            "fp": _wav_fingerprint(wav) if wav else None,
        }
        uids[uid] = meta
        if wav is None:
            n_missing += 1
            continue
        n_ok += 1
        cer = rec.get("oracle_cer")
        if cer is not None:
            try:
                cers.append(float(cer))
            except (TypeError, ValueError):
                pass
        st = str(rec.get("best_stage") or "")
        if st:
            stages[st] += 1
        sm = str(rec.get("oracle_stream") or "")
        if sm:
            streams[sm] += 1
            if sm == "original":
                n_orig += 1
        if max_audio and n_audio >= max_audio:
            continue
        loaded = load_wav_mono(wav)
        if loaded is None:
            continue
        n_audio += 1
        x, sr = loaded
        durs.append(len(x) / float(sr) if sr else 0.0)
        snrs.append(snr_med_db(x, sr))
        pmuse.append(p_music_heuristic(x, sr))

    n_cer = len(cers)
    cer0 = sum(1 for c in cers if c <= 1e-12) / n_cer if n_cer else None
    return {
        "path": str(best_sep.resolve()),
        "n_index": len(rows),
        "n_ok": n_ok,
        "n_missing_wav": n_missing,
        "n_audio_scored": n_audio,
        "oracle_cer_mean": (sum(cers) / n_cer) if n_cer else None,
        "oracle_cer_p90": _pct(cers, 90),
        "cer0_rate": cer0,
        "n_with_cer": n_cer,
        "original_winner_rate": (n_orig / n_ok) if n_ok else None,
        "dur_sec_mean": (sum(durs) / len(durs)) if durs else None,
        "dur_sec_p10": _pct(durs, 10),
        "snr_med_db_mean": (sum(snrs) / len(snrs)) if snrs else None,
        "p_music_mean": (sum(pmuse) / len(pmuse)) if pmuse else None,
        "best_stage": dict(stages.most_common(12)),
        "oracle_stream": dict(streams.most_common(12)),
        "uids": uids,
    }


def cer_constraint(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    mean = candidate.get("oracle_cer_mean")
    cer0 = candidate.get("cer0_rate")
    ok = True
    reasons: list[str] = []
    if mean is None:
        reasons.append("cer_missing")
        ok = False
    elif float(mean) > CER_MEAN_MAX + 1e-12:
        reasons.append("cer_mean_over_0.03")
        ok = False
    if baseline and cer0 is not None and baseline.get("cer0_rate") is not None:
        drop = float(baseline["cer0_rate"]) - float(cer0)
        if drop > CER0_DROP_MAX + 1e-12:
            reasons.append("cer0_rate_drop_over_2pp")
            ok = False
    if ok:
        reasons.append("cer_holds")
    return {"ok": ok, "reasons": reasons, "cer_mean": mean, "cer0_rate": cer0}


def pairwise_disagreement(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ua, ub = a.get("uids") or {}, b.get("uids") or {}
    common = sorted(set(ua) & set(ub))
    n_fp = n_stream = n_stage = 0
    for uid in common:
        ra, rb = ua[uid], ub[uid]
        if ra.get("fp") and rb.get("fp") and ra["fp"] != rb["fp"]:
            n_fp += 1
        if (ra.get("oracle_stream") or "") != (rb.get("oracle_stream") or ""):
            n_stream += 1
        if (ra.get("best_stage") or "") != (rb.get("best_stage") or ""):
            n_stage += 1
    return {
        "n_common": len(common),
        "n_only_a": len(set(ua) - set(ub)),
        "n_only_b": len(set(ub) - set(ua)),
        "n_wav_fingerprint_diff": n_fp,
        "n_oracle_stream_diff": n_stream,
        "n_best_stage_diff": n_stage,
    }


def verdict(
    *,
    names: list[str],
    summaries: dict[str, dict[str, Any]],
    presence: dict[str, dict[str, float]] | None,
    baseline: str,
    cmd: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """CER is a constraint. KWS-local rank is CMD cosine. Presence is later."""
    base = summaries[baseline]
    out: dict[str, Any] = {"baseline": baseline, "adopt": None, "rows": []}
    for name in names:
        row: dict[str, Any] = {"name": name, "cer": cer_constraint(summaries[name], base)}
        if presence and name in presence and baseline in presence and name != baseline:
            go = enroll_go(
                baseline=presence[baseline],
                candidate=presence[name],
                cer_mean=float(summaries[name].get("oracle_cer_mean") or 0.0),
                cer0_rate=float(summaries[name].get("cer0_rate") or 0.0),
                cer0_rate_baseline=float(base.get("cer0_rate") or 0.0),
            )
            row["presence"] = {"accept": go.accept, "reason": go.reason, **go.metrics}
            if go.accept and row["cer"]["ok"]:
                out["adopt"] = name
        out["rows"].append(row)
    if cmd:
        out["cmd_rank"] = rank_cmd_groups(cmd, baseline=baseline)
        out["note"] = out["cmd_rank"]["note"]
    elif presence is None:
        out["note"] = (
            "KWS 本步用 enroll↔CMD cosine 排序（scripts/eval_cmd_cosine.py），"
            "不要用 CER/SNR 宣布某组更好。冻结 Presence 是后续 extract@main 的否决，不是这一步。"
        )
    elif out["adopt"] is None:
        out["note"] = "无候选通过 Presence 否决规则；保持 baseline。"
    else:
        out["note"] = f"建议采用 {out['adopt']}（冻结 Presence 改进且 CER 约束成立）。"
    return out
