"""ERes2NetV2 enroll encoder (same family as contest Presence).

Sidecar was empty because this repo never embedded wavs. Audio exists:
  datasetA/{pos,neg}/kws_*.wav  and  pos_neg/{split}/{stage}/wav/{uid}_{tag}.wav
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray: ...


class FftProxyEncoder:
    """Plumbing only. Never rank best_sep with this."""

    name = "fft_proxy_not_eres"

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        x = np.asarray(wav, dtype=np.float32).reshape(-1)
        if x.size < 16:
            x = np.pad(x, (0, 16 - x.size))
        n = 1 << int(np.floor(np.log2(min(max(x.size, 256), 1024))))
        mag = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
        logm = np.log(np.maximum(mag, 1e-8))
        target = 64
        if logm.size == target:
            return logm.astype(np.float32)
        idx = np.linspace(0, logm.size - 1, target)
        return np.interp(idx, np.arange(logm.size), logm).astype(np.float32)


def _extract_ve_scripts() -> Path | None:
    env = os.environ.get("EXTRACT_VE") or os.environ.get("EXTRACT_ROOT")
    if env:
        p = Path(env)
        if (p / "presence_encoder.py").is_file():
            return p
        cand = p / "ve" / "scripts"
        if (cand / "presence_encoder.py").is_file():
            return cand
    here = Path(__file__).resolve()
    for root in (here.parents[3], here.parents[2], Path(r"d:\media")):
        cand = root / "extract" / "ve" / "scripts"
        if (cand / "presence_encoder.py").is_file():
            return cand
    return None


def _import_modelscope_sv():
    try:
        import modelscope as ms
        from modelscope.pipelines import pipeline
    except ImportError as e:
        hint = (
            f"{sys.executable} -m pip install -U modelscope addict simplejson sortedcontainers "
            "datasets oss2"
        )
        raise ImportError(
            f"speaker-verification pipeline import failed ({type(e).__name__}: {e}). "
            f"Install into THIS interpreter: {hint}"
        ) from e
    tasks_sv = "speaker-verification"
    try:
        from modelscope.utils.constant import Tasks

        tasks_sv = getattr(Tasks, "speaker_verification", None) or tasks_sv
    except Exception:
        pass
    return pipeline, tasks_sv, ms


class ERes2NetV2Encoder:
    """ModelScope iic/speech_eres2netv2_sv_zh-cn_16k-common."""

    name = "eres2netv2_zh"

    def __init__(self, model_dir: str | Path | None = None, device: str = "cuda:0"):
        self.device = device
        self.model_dir = Path(model_dir) if model_dir else None
        self._sv = None
        self._load()

    def _load(self) -> None:
        pipeline, task, _ms = _import_modelscope_sv()
        model_id = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
        model_ref = model_id
        if self.model_dir and self.model_dir.is_dir():
            tip = self.model_dir / "MODELSCOPE_PATH.txt"
            if tip.is_file():
                model_ref = tip.read_text(encoding="utf-8").strip() or model_id
            elif (self.model_dir / "config.yaml").is_file() or (
                self.model_dir / "configuration.json"
            ).is_file():
                model_ref = str(self.model_dir)
        last_err: Exception | None = None
        attempts = [
            dict(task=task, model=model_ref, model_revision="master", device=self.device),
            dict(task=task, model=model_ref, device=self.device),
            dict(task=task, model=model_ref),
            dict(task="speaker-verification", model=model_id),
        ]
        for kwargs in attempts:
            try:
                self._sv = pipeline(**kwargs)
                return
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"ERes2NetV2 pipeline failed: {last_err}") from last_err

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        last_out: Any = None
        mem_calls = (
            lambda: self._sv([wav], output_emb=True),
            lambda: self._sv([wav], extract_emb=True),
            lambda: self._sv([{"array": wav, "sampling_rate": int(sr)}], output_emb=True),
            lambda: self._sv([wav]),
        )
        for call in mem_calls:
            try:
                out = call()
            except TypeError:
                continue
            except Exception:
                continue
            last_out = out
            emb = self._parse_emb(out)
            if emb is not None:
                return emb
        import os
        import tempfile

        from .audio import save_wav_mono

        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            save_wav_mono(tmp, wav, sr)
            for call in (
                lambda: self._sv([str(tmp)], output_emb=True),
                lambda: self._sv([str(tmp)]),
            ):
                try:
                    out = call()
                except TypeError:
                    continue
                except Exception:
                    continue
                last_out = out
                emb = self._parse_emb(out)
                if emb is not None:
                    return emb
        finally:
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ERes2NetV2 returned no embedding: type={type(last_out)}")

    @staticmethod
    def _parse_emb(out: Any) -> np.ndarray | None:
        if isinstance(out, dict):
            for k in ("embs", "embedding", "spk_embedding", "emb"):
                if k in out and out[k] is not None:
                    arr = np.asarray(out[k], dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[0]
                    return arr.reshape(-1)
        if isinstance(out, (list, tuple)) and out:
            return ERes2NetV2Encoder._parse_emb(out[0])
        if isinstance(out, np.ndarray):
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[0]
            return arr.reshape(-1)
        return None


class CAMPlusEncoder(ERes2NetV2Encoder):
    name = "campplus_zh"

    def _load(self) -> None:
        pipeline, task, _ms = _import_modelscope_sv()
        model_id = "iic/speech_campplus_sv_zh-cn_16k-common"
        model_ref = model_id
        if self.model_dir and self.model_dir.is_dir():
            tip = self.model_dir / "MODELSCOPE_PATH.txt"
            if tip.is_file():
                model_ref = tip.read_text(encoding="utf-8").strip() or model_id
            elif (self.model_dir / "config.yaml").is_file():
                model_ref = str(self.model_dir)
        last_err: Exception | None = None
        attempts = [
            dict(task=task, model=model_ref, device=self.device),
            dict(task=task, model=model_ref),
            dict(task="speaker-verification", model=model_id),
        ]
        for kwargs in attempts:
            try:
                self._sv = pipeline(**kwargs)
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"CAM++ pipeline failed: {last_err}") from last_err


def load_embedder(
    backend: str = "eres2netv2",
    *,
    model_dir: str | Path | None = None,
    device: str | None = None,
    extract_ve: str | Path | None = None,
) -> Embedder:
    backend = (backend or "eres2netv2").lower().strip()
    if device is None:
        device = os.environ.get("KWS_DEVICE") or ("cuda:0" if _cuda_ok() else "cpu")
    if backend in ("fft", "fft_proxy", "proxy"):
        return FftProxyEncoder()
    ve = Path(extract_ve) if extract_ve else _extract_ve_scripts()
    if ve and str(ve) not in sys.path:
        sys.path.insert(0, str(ve))
    if backend in ("eres2netv2", "eres", "eres2net"):
        if ve is not None:
            try:
                from presence_encoder import ERes2NetV2Encoder as Ext  # type: ignore

                return Ext(model_dir=model_dir, device=device)
            except Exception:
                pass
        return ERes2NetV2Encoder(model_dir=model_dir, device=device)
    if backend in ("campplus", "cam++", "campplus_zh"):
        return CAMPlusEncoder(model_dir=model_dir, device=device)
    raise ValueError(f"unknown encoder backend {backend!r}; use eres2netv2 | campplus | fft")


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
