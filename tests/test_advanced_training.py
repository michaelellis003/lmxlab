"""Tests for advanced training: DPO, GRPO, curriculum, MTP."""

import mlx.core as mx
import pytest

from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_tiny
from lmxlab.training.curriculum import (
    difficulty_curriculum,
    length_curriculum,
)
from lmxlab.training.dpo import _sequence_log_probs, dpo_loss
from lmxlab.training.grpo import grpo_loss
from lmxlab.training.mtp import MultiTokenPrediction


@pytest.fixture
def tiny_model() -> LanguageModel:
    config = gpt_tiny()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def ref_model() -> LanguageModel:
    """Separate reference model (frozen)."""
    config = gpt_tiny()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


class TestSequenceLogProbs:
    def test_shape(self, tiny_model):
        """Log probs should return one value per sequence."""
        tokens = mx.random.randint(0, 256, shape=(4, 17))
        logits, _ = tiny_model(tokens[:, :-1])
        mx.eval(logits)
        log_probs = _sequence_log_probs(logits, tokens[:, 1:])
        mx.eval(log_probs)
        assert log_probs.shape == (4,)

    def test_negative(self, tiny_model):
        """Log probs should be negative."""
        tokens = mx.random.randint(0, 256, shape=(2, 10))
        logits, _ = tiny_model(tokens[:, :-1])
        mx.eval(logits)
        log_probs = _sequence_log_probs(logits, tokens[:, 1:])
        mx.eval(log_probs)
        assert mx.all(log_probs < 0).item()


class TestDPO:
    def test_loss_positive(self, tiny_model, ref_model):
        """DPO loss should be positive."""
        chosen = mx.random.randint(0, 256, shape=(2, 16))
        rejected = mx.random.randint(0, 256, shape=(2, 16))
        loss = dpo_loss(tiny_model, ref_model, chosen, rejected)
        mx.eval(loss)
        assert loss.item() > 0

    def test_loss_scalar(self, tiny_model, ref_model):
        """DPO loss should be a scalar."""
        chosen = mx.random.randint(0, 256, shape=(2, 16))
        rejected = mx.random.randint(0, 256, shape=(2, 16))
        loss = dpo_loss(tiny_model, ref_model, chosen, rejected)
        mx.eval(loss)
        assert loss.shape == ()


class TestGRPO:
    def test_loss_scalar(self, tiny_model, ref_model):
        """GRPO loss should be a scalar."""
        prompts = mx.random.randint(0, 256, shape=(4, 8))
        completions = mx.random.randint(0, 256, shape=(4, 16))
        rewards = mx.array([1.0, 0.5, -0.5, -1.0])
        loss = grpo_loss(tiny_model, ref_model, prompts, completions, rewards)
        mx.eval(loss)
        assert loss.shape == ()


class TestCurriculum:
    def test_length_curriculum_increasing(self):
        """Batch seq_len should increase across stages."""
        tokens = mx.arange(10000, dtype=mx.int32)
        batches = list(
            length_curriculum(
                tokens,
                batch_size=4,
                min_seq_len=16,
                max_seq_len=64,
                n_stages=4,
                batches_per_stage=2,
            )
        )
        assert len(batches) > 0
        # First batch should be shorter than last
        first_len = batches[0][0].shape[1]
        last_len = batches[-1][0].shape[1]
        assert last_len >= first_len

    def test_difficulty_curriculum_shapes(self):
        """Difficulty curriculum produces correct shapes."""
        easy = mx.arange(5000, dtype=mx.int32)
        hard = mx.arange(5000, dtype=mx.int32)
        batches = list(
            difficulty_curriculum(
                easy,
                hard,
                batch_size=4,
                seq_len=16,
                n_batches=5,
            )
        )
        assert len(batches) == 5
        for x, y in batches:
            assert x.shape == (4, 16)
            assert y.shape == (4, 16)


class TestMTP:
    """Tests for Multi-Token Prediction."""

    def test_returns_losses(self, tiny_model):
        """MTP should return main_loss, mtp_loss, total_loss."""
        mtp = MultiTokenPrediction(tiny_model, n_predict=2)
        mx.eval(mtp.parameters())

        x = mx.random.randint(0, 256, shape=(2, 16))
        targets = mx.random.randint(0, 256, shape=(2, 16))
        logits, losses = mtp(x, targets)
        mx.eval(logits, losses["main_loss"], losses["mtp_loss"])

        assert "main_loss" in losses
        assert "mtp_loss" in losses
        assert "total_loss" in losses

    def test_total_loss_includes_mtp(self, tiny_model):
        """Total loss should be main + weight * mtp."""
        mtp = MultiTokenPrediction(tiny_model, n_predict=2, mtp_weight=0.5)
        mx.eval(mtp.parameters())

        x = mx.random.randint(0, 256, shape=(2, 16))
        targets = mx.random.randint(0, 256, shape=(2, 16))
        _, losses = mtp(x, targets)
        mx.eval(
            losses["main_loss"],
            losses["mtp_loss"],
            losses["total_loss"],
        )

        expected = losses["main_loss"].item() + 0.5 * losses["mtp_loss"].item()
        assert abs(losses["total_loss"].item() - expected) < 1e-4

    def test_logits_shape(self, tiny_model):
        """MTP logits should match standard forward pass shape."""
        mtp = MultiTokenPrediction(tiny_model, n_predict=2)
        mx.eval(mtp.parameters())

        x = mx.random.randint(0, 256, shape=(2, 16))
        targets = mx.random.randint(0, 256, shape=(2, 16))
        logits, _ = mtp(x, targets)
        mx.eval(logits)

        assert logits.shape == (2, 16, tiny_model.config.vocab_size)

    def test_n_predict_1(self, tiny_model):
        """MTP with n_predict=1 should still work."""
        mtp = MultiTokenPrediction(tiny_model, n_predict=1)
        mx.eval(mtp.parameters())

        x = mx.random.randint(0, 256, shape=(2, 16))
        targets = mx.random.randint(0, 256, shape=(2, 16))
        _, losses = mtp(x, targets)
        mx.eval(losses["total_loss"])
        assert losses["total_loss"].item() > 0
