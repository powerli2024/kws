"""Frozen Qwen3-ASR transcription adapter for raw/SE paired evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
        max_batch_size: int = 8,
        max_new_tokens: int = 64,
    ) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

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

    def transcribe_many(
        self,
        wavs: Sequence[np.ndarray],
        *,
        language: str,
        wake_text: str,
        context_mode: str = "wake",
    ) -> list[str]:
        if not wavs:
            return []
        audio = [(np.asarray(w, dtype=np.float32).reshape(-1), 16000) for w in wavs]
        kwargs = {"audio": audio, "language": qwen_language(language)}
        if context_mode == "wake":
            kwargs["context"] = str(wake_text)
        elif context_mode != "none":
            raise ValueError(f"unknown context_mode={context_mode!r}")
        try:
            results = self.model.transcribe(**kwargs)
        except TypeError:
            # Older qwen-asr packages call the same field ``prompt``.
            if "context" not in kwargs:
                raise
            kwargs["prompt"] = kwargs.pop("context")
            try:
                results = self.model.transcribe(**kwargs)
            except TypeError:
                kwargs.pop("prompt", None)
                results = self.model.transcribe(**kwargs)
        if len(results) != len(audio):
            raise RuntimeError(f"ASR returned {len(results)} rows for {len(audio)} inputs")
        return [str(getattr(item, "text", "") or "") for item in results]
