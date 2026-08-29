"""Frozen Qwen3-ASR transcription adapter for raw/SE paired evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _configure_generation_logging(wrapper: object) -> None:
    """Set the Qwen generation pad token once; avoid one warning per batch."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*[Ss]etting.*pad[_ ]token[_ ]id.*eos[_ ]token[_ ]id.*",
    )
    try:
        import transformers

        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    objects = [wrapper, getattr(wrapper, "model", None)]
    objects.extend(getattr(obj, key, None) for obj in list(objects) for key in ("model", "thinker"))
    for obj in objects:
        if obj is None:
            continue
        config = getattr(obj, "generation_config", None)
        if config is None:
            config = getattr(obj, "config", None)
        eos = getattr(config, "eos_token_id", None) if config is not None else None
        if eos is not None and config is not None and getattr(config, "pad_token_id", None) is None:
            try:
                config.pad_token_id = eos
            except Exception:
                pass


def qwen_language(lang: str) -> str:
    value = str(lang).strip().lower()
    if value in {"zh", "cn", "chinese", "中文", "mandarin"}:
        return "Chinese"
    if value in {"en", "english", "英文"}:
        return "English"
    raise ValueError(f"unsupported language {lang!r}")


class Qwen3ASRTranscriber:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_new_tokens: int = 64,
    ) -> None:
        import torch
        try:
            from qwen_asr import Qwen3ASRModel
        except Exception as exc:
            detail = str(exc)
            if "numpy.dtype size changed" in detail or "sklearn" in detail.lower():
                raise RuntimeError(
                    "Qwen3-ASR dependency ABI mismatch: scikit-learn/NumPy cannot "
                    "be imported. Rebuild the ve environment with compatible "
                    "wheels, e.g. numpy==1.26.4 and scikit-learn==1.4.2, then "
                    "rerun the same route command."
                ) from exc
            raise

        dtype_obj = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(str(dtype).lower())
        if dtype_obj is None:
            raise ValueError(f"unsupported dtype {dtype!r}")
        _configure_generation_logging(Qwen3ASRModel)
        self.model = Qwen3ASRModel.from_pretrained(
            str(model_dir),
            dtype=dtype_obj,
            device_map=device,
            max_inference_batch_size=max(1, int(max_batch_size)),
            max_new_tokens=max(1, int(max_new_tokens)),
        )
        _configure_generation_logging(self.model)
        self.max_batch_size = max(1, int(max_batch_size))
        self.model_dir = str(model_dir)
        self.device = str(device)
        self.dtype_name = str(dtype)

    @property
    def runtime_info(self) -> dict[str, Any]:
        """Small provenance record written into ASR sidecars and caches."""
        info: dict[str, Any] = {
            "adapter": "kws.Qwen3ASRTranscriber",
            "model_dir": self.model_dir,
            "device": self.device,
            "dtype": self.dtype_name,
            "max_batch_size": self.max_batch_size,
            "sample_rate": 16000,
        }
        for package in ("qwen_asr", "transformers"):
            try:
                module = __import__(package)
                info[f"{package}_version"] = str(getattr(module, "__version__", "unknown"))
            except Exception:
                info[f"{package}_version"] = "unavailable"
        return info

    def transcribe_many(
        self,
        wavs: Sequence[np.ndarray],
        *,
        language: str,
        wake_text: str,
        context_mode: str = "none",
    ) -> list[str]:
        if not wavs:
            return []
        audio = [(np.asarray(w, dtype=np.float32).reshape(-1), 16000) for w in wavs]
        mode = str(context_mode).strip().lower()
        if mode not in {"wake", "none"}:
            raise ValueError(f"unknown context_mode={context_mode!r}; expected wake or none")
        # Q0 must be genuinely free transcription.  Passing an explicit empty
        # context keeps the request unambiguous across Qwen3-ASR releases.
        kwargs = {"audio": audio, "language": qwen_language(language), "context": ""}
        if mode == "wake":
            text = str(wake_text or "").strip()
            if not text:
                raise ValueError("context_mode='wake' requires non-empty wake_text")
            kwargs["context"] = text
        try:
            results = self.model.transcribe(**kwargs)
        except TypeError as exc:
            # Do not silently fall back to prompt/no-context: that changes Q0
            # into a different experiment and makes CER comparisons invalid.
            raise RuntimeError(
                "installed qwen_asr does not accept the explicit context field; "
                "upgrade/pin a compatible Qwen3-ASR package instead of falling back"
            ) from exc
        if len(results) != len(audio):
            raise RuntimeError(f"ASR returned {len(results)} rows for {len(audio)} inputs")
        return [str(getattr(item, "text", "") or "") for item in results]

    def transcribe_many_detailed(
        self,
        wavs: Sequence[np.ndarray],
        *,
        language: str,
        wake_text: str,
        context_mode: str = "none",
    ) -> list[dict[str, Any]]:
        """Return text plus deterministic preprocessing/runtime provenance."""
        texts = self.transcribe_many(
            wavs,
            language=language,
            wake_text=wake_text,
            context_mode=context_mode,
        )
        return [
            {
                "hyp": text,
                "language": qwen_language(language),
                "context_mode": str(context_mode),
                "sample_rate": 16000,
                "num_samples": int(np.asarray(wav).size),
                "duration_sec": float(np.asarray(wav).size / 16000.0),
            }
            for wav, text in zip(wavs, texts)
        ]
