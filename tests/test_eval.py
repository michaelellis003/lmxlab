"""Tests for evaluation metrics."""

import math

import mlx.core as mx
import pytest

from lmt_metal.eval.metrics import bits_per_byte, perplexity
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
