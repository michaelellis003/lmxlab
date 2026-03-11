"""Tests for evaluation metrics."""

import math

import mlx.core as mx
import pytest

from lmt_metal.eval.metrics import (
    bits_per_byte,
    evaluate_pass_at_k,
    pass_at_k,
    perplexity,
)
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.gpt import gpt_tiny


@pytest.fixture
def tiny_model() -> LanguageModel:
    config = gpt_tiny()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def eval_data() -> list[mx.array]:
    """Random token batches for evaluation."""
    return [mx.random.randint(0, 256, shape=(2, 32)) for _ in range(3)]


class TestPerplexity:
    def test_positive(self, tiny_model, eval_data):
        """Perplexity is always positive."""
        ppl = perplexity(tiny_model, eval_data)
        assert ppl > 0

    def test_at_least_one(self, tiny_model, eval_data):
        """Perplexity is at least 1.0 (= perfect prediction)."""
        ppl = perplexity(tiny_model, eval_data)
        assert ppl >= 1.0

    def test_random_model_high_perplexity(self, tiny_model, eval_data):
        """Random model should have perplexity near vocab_size."""
        ppl = perplexity(tiny_model, eval_data)
        # Random model on vocab=256 should have PPL roughly ~256
        # Allow wide range since weights are random, not uniform
        assert ppl > 10  # definitely not perfect
        assert ppl < 10000  # not astronomically bad


class TestBitsPerByte:
    def test_positive(self, tiny_model, eval_data):
        """BPB is always positive."""
        bpb = bits_per_byte(tiny_model, eval_data)
        assert bpb > 0

    def test_relationship_to_perplexity(self, tiny_model, eval_data):
        """BPB and perplexity should be consistent.

        PPL = exp(loss), BPB = loss / ln(2)
        So BPB = log2(PPL) for bytes_per_token=1.
        """
        ppl = perplexity(tiny_model, eval_data)
        bpb = bits_per_byte(tiny_model, eval_data, bytes_per_token=1.0)
        expected_bpb = math.log2(ppl)
        assert abs(bpb - expected_bpb) < 0.01

    def test_bytes_per_token_scaling(self, tiny_model, eval_data):
        """Doubling bytes_per_token halves BPB."""
        bpb1 = bits_per_byte(tiny_model, eval_data, bytes_per_token=1.0)
        bpb2 = bits_per_byte(tiny_model, eval_data, bytes_per_token=2.0)
        assert abs(bpb2 - bpb1 / 2) < 0.01


class TestPassAtK:
    def test_all_pass(self):
        """All samples pass -> pass@k = 1.0."""
        assert pass_at_k(n=10, c=10, k=1) == 1.0
        assert pass_at_k(n=10, c=10, k=5) == 1.0

    def test_none_pass(self):
        """No samples pass -> pass@k = 0.0."""
        assert pass_at_k(n=10, c=0, k=1) == 0.0
        assert pass_at_k(n=10, c=0, k=5) == 0.0

    def test_pass_at_1(self):
        """pass@1 = c/n (proportion that pass)."""
        result = pass_at_k(n=10, c=3, k=1)
        assert abs(result - 0.3) < 1e-10

    def test_pass_at_k_increases_with_k(self):
        """More samples -> higher chance of passing."""
        p1 = pass_at_k(n=10, c=3, k=1)
        p5 = pass_at_k(n=10, c=3, k=5)
        p10 = pass_at_k(n=10, c=3, k=10)
        assert p1 < p5 < p10

    def test_k_equals_n(self):
        """k=n with any passing -> pass@k = 1.0."""
        assert pass_at_k(n=5, c=1, k=5) == 1.0

    def test_known_value(self):
        """Verify against hand-computed value.

        n=4, c=2, k=2: 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6
        """
        result = pass_at_k(n=4, c=2, k=2)
        assert abs(result - 5 / 6) < 1e-10


class TestEvaluatePassAtK:
    def test_perfect_test_fn(self):
        """All completions pass -> all pass@k = 1.0."""
        completions = [["correct"] * 5] * 3
        result = evaluate_pass_at_k(
            completions, lambda x: True, k_values=[1, 5]
        )
        assert result["pass@1"] == 1.0
        assert result["pass@5"] == 1.0

    def test_failing_test_fn(self):
        """No completions pass -> all pass@k = 0.0."""
        completions = [["wrong"] * 5] * 3
        result = evaluate_pass_at_k(completions, lambda x: False, k_values=[1])
        assert result["pass@1"] == 0.0

    def test_partial_pass(self):
        """Some completions pass."""
        completions = [
            ["yes", "no", "yes", "no", "no"],  # 2/5 pass
        ]
        result = evaluate_pass_at_k(
            completions,
            lambda x: x == "yes",
            k_values=[1],
        )
        assert abs(result["pass@1"] - 0.4) < 1e-10

    def test_default_k_values(self):
        """Default k_values are [1, 5, 10]."""
        completions = [["a"] * 15]
        result = evaluate_pass_at_k(completions, lambda x: True)
        assert "pass@1" in result
        assert "pass@5" in result
        assert "pass@10" in result
