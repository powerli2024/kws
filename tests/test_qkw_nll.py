import math
from types import SimpleNamespace

import pytest
import torch

from kws.qkw_nll import Qwen3ASRNLLScorer, build_target_labels, qwen_language, shifted_token_nll


def test_build_target_labels_masks_prefix_pad_and_eos():
    ids = torch.tensor([[10, 11, 3, 4, 2, 0], [10, 3, 5, 2, 0, 0]])
    labels = build_target_labels(
        ids,
        prefix_lengths=[2, 1],
        pad_token_id=0,
        eos_token_id=2,
    )
    assert labels.tolist() == [
        [-100, -100, 3, 4, -100, -100],
        [-100, 3, 5, -100, -100, -100],
    ]


def test_shifted_token_nll_is_length_normalized():
    labels = torch.tensor([[-100, 1, 2, -100], [-100, 1, -100, -100]])
    logits = torch.zeros((2, 4, 3), dtype=torch.float32)
    logits[0, 0, 1] = math.log(3.0)
    logits[0, 1, 2] = math.log(3.0)
    logits[1, 0, 1] = math.log(3.0)
    results = shifted_token_nll(logits, labels)
    assert results[0].token_count == 2
    assert results[1].token_count == 1
    assert results[0].nll == pytest.approx(results[1].nll, abs=1e-7)


def test_shifted_token_nll_rejects_empty_target():
    with pytest.raises(ValueError, match="no target tokens"):
        shifted_token_nll(torch.zeros((1, 2, 3)), torch.full((1, 2), -100))


def test_qwen_language_is_strict():
    assert qwen_language("zh") == "Chinese"
    assert qwen_language("en") == "English"
    with pytest.raises(ValueError, match="unsupported"):
        qwen_language("")


def test_scorer_uses_empty_context_and_masks_eos():
    calls = []

    class Wrapper:
        def _build_text_prompt(self, *, context, force_language):
            calls.append((context, force_language))
            return "PREFIX"

    class Tokenizer:
        eos_token = "<eos>"
        eos_token_id = 2
        pad_token_id = 0

    class Processor:
        tokenizer = Tokenizer()

        def __call__(self, *, text, audio, return_tensors, padding, truncation):
            assert return_tensors == "pt" and padding and not truncation
            if text[0] == "PREFIX":
                return {
                    "input_ids": torch.tensor([[10, 11]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }
            assert text == ["PREFIX你好<eos>"]
            return {
                "input_ids": torch.tensor([[10, 11, 3, 2]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                "input_features": torch.zeros((1, 2, 2)),
                "feature_attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    class Thinker:
        def __call__(self, **kwargs):
            logits = torch.zeros((1, 4, 12))
            logits[0, 1, 3] = 5.0
            return SimpleNamespace(logits=logits)

    scorer = Qwen3ASRNLLScorer.__new__(Qwen3ASRNLLScorer)
    scorer.wrapper = Wrapper()
    scorer.processor = Processor()
    scorer.model = SimpleNamespace(thinker=Thinker())
    scorer.torch = torch
    scorer.dtype = torch.float32
    scorer.device = torch.device("cpu")
    result = scorer.score_batch([torch.zeros(160)], ["你好"], ["zh"])[0]
    assert calls == [("", "Chinese")]
    assert result.token_count == 1
    assert result.nll < 0.1
