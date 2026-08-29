"""Frozen Qwen3-ASR teacher-forced token NLL for known wake text.

The known wake text is used only as the target labels. It is never inserted
into the system context. Raw NLL is a within-UID rank score; it is not a
calibrated probability and must not activate absolute q_kw reject thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def _configure_generation_logging(wrapper: object) -> None:
    """Keep Qwen/Transformers pad-token configuration from printing per call."""
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
    for obj in (wrapper, getattr(wrapper, "model", None)):
        config = getattr(obj, "generation_config", None) if obj is not None else None
        eos = getattr(config, "eos_token_id", None) if config is not None else None
        if config is not None and eos is not None and getattr(config, "pad_token_id", None) is None:
            try:
                config.pad_token_id = eos
            except Exception:
                pass


@dataclass(frozen=True)
class NLLResult:
    nll: float
    token_count: int


def qwen_language(lang: str | None) -> str:
    value = str(lang or "").strip().lower()
    if value in {"zh", "cn", "chinese", "中文", "mandarin"}:
        return "Chinese"
    if value in {"en", "english", "英文"}:
        return "English"
    raise ValueError(f"unsupported wake language {lang!r}; expected zh or en")


def build_target_labels(
    input_ids: Any,
    *,
    prefix_lengths: Sequence[int],
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> Any:
    """Mask prompt/audio/padding/EOS so only known-wake tokens contribute."""
    labels = input_ids.clone()
    if labels.ndim != 2 or labels.shape[0] != len(prefix_lengths):
        raise ValueError(
            f"input_ids shape={tuple(labels.shape)} does not match "
            f"prefix_lengths n={len(prefix_lengths)}"
        )
    width = int(labels.shape[1])
    for i, raw_len in enumerate(prefix_lengths):
        prefix_len = int(raw_len)
        if prefix_len < 0 or prefix_len > width:
            raise ValueError(f"bad prefix length {prefix_len} for width={width}")
        labels[i, :prefix_len] = -100
    if pad_token_id is not None:
        labels[labels == int(pad_token_id)] = -100
    if eos_token_id is not None:
        labels[labels == int(eos_token_id)] = -100
    return labels


def shifted_token_nll(logits: Any, labels: Any) -> list[NLLResult]:
    """Return per-sample causal-LM NLL averaged over non-masked target tokens."""
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"expected logits [B,T,V] and labels [B,T], got "
            f"{tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    if logits.shape[1] < 2:
        raise ValueError("sequence is too short for shifted causal loss")

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    flat_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shifted_labels)

    results: list[NLLResult] = []
    for i in range(labels.shape[0]):
        count = int(mask[i].sum().item())
        if count <= 0:
            raise ValueError(f"sample {i} has no target tokens after masking")
        value = torch.sum(flat_loss[i] * mask[i]).item() / count
        results.append(NLLResult(nll=float(value), token_count=count))
    return results


class Qwen3ASRNLLScorer:
    """Transformers-backend scorer using the official Qwen3-ASR processor."""

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_batch_size: int = 4,
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
        self.torch = torch
        self.dtype = dtype_obj
        self.wrapper = Qwen3ASRModel.from_pretrained(
            model_dir,
            dtype=dtype_obj,
            device_map=device,
            max_inference_batch_size=int(max_batch_size),
        )
        _configure_generation_logging(self.wrapper)
        if getattr(self.wrapper, "backend", "transformers") != "transformers":
            raise RuntimeError("q_kw NLL requires the Qwen3-ASR Transformers backend, not vLLM")
        self.model = getattr(self.wrapper, "model", None)
        self.processor = getattr(self.wrapper, "processor", None)
        if self.model is None or self.processor is None or not hasattr(self.model, "thinker"):
            raise RuntimeError("incompatible qwen_asr package: need wrapper.model.thinker and wrapper.processor")
        self.model.eval()
        self.device = getattr(self.model, "device", None)
        if self.device is None:
            self.device = next(self.model.parameters()).device

    def _prefix(self, lang: str) -> str:
        language = qwen_language(lang)
        builder = getattr(self.wrapper, "_build_text_prompt", None)
        if callable(builder):
            return str(builder(context="", force_language=language))
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        base = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return str(base) + f"language {language}<asr_text>"

    def score_batch(
        self,
        audios: Sequence[Any],
        targets: Sequence[str],
        languages: Sequence[str],
    ) -> list[NLLResult]:
        if not audios or not (len(audios) == len(targets) == len(languages)):
            raise ValueError("audios, targets and languages must have the same non-zero length")
        if any(not str(text).strip() for text in targets):
            raise ValueError("known wake target must be non-empty")

        prefixes = [self._prefix(lang) for lang in languages]
        tokenizer = self.processor.tokenizer
        eos_text = tokenizer.eos_token or ""
        full_texts = [prefix + str(target) + eos_text for prefix, target in zip(prefixes, targets)]
        full_inputs = self.processor(
            text=full_texts,
            audio=list(audios),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefixes,
            audio=list(audios),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = build_target_labels(
            full_inputs["input_ids"],
            prefix_lengths=prefix_lengths,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        model_inputs: dict[str, Any] = {}
        for key in ("input_ids", "attention_mask", "input_features", "feature_attention_mask"):
            value = full_inputs.get(key)
            if value is None:
                continue
            value = value.to(self.device)
            if value.is_floating_point():
                value = value.to(dtype=self.dtype)
            model_inputs[key] = value
        labels = labels.to(self.device)
        with self.torch.inference_mode():
            outputs = self.model.thinker(**model_inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError(f"Qwen3-ASR thinker returned no logits: {type(outputs)}")
        return shifted_token_nll(logits, labels)
