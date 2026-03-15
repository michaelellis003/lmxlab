"""Tests for experiment-specific metric callbacks."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_tiny
from lmxlab.training.config import TrainConfig
from lmxlab.training.metric_callbacks import (
    ActivationStatsCallback,
    AttentionEntropyCallback,
    EffectiveRankCallback,
    GradientStatsCallback,
    LossCurvatureCallback,
    WeightStatsCallback,
)


@pytest.fixture
def tiny_model() -> LanguageModel:
    """Create a tiny GPT model for testing."""
    config = gpt_tiny()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


def _loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """Simple CE loss for gradient testing."""
    logits, _ = model(x)
    logits = logits.reshape(-1, logits.shape[-1])
    return nn.losses.cross_entropy(logits, y.reshape(-1), reduction="mean")


class TestGradientStats:
    def test_injects_metrics(self, tiny_model):
        """exp_grad_norm_mean present after measurement."""
        cb = GradientStatsCallback(tiny_model, _loss_fn, log_interval=1)
        cb.on_train_begin(TrainConfig())
        x = mx.random.randint(0, 256, shape=(2, 16))
        y = mx.random.randint(0, 256, shape=(2, 16))
        cb.set_probe_batch((x, y))
        metrics: dict = {"loss": 1.0}
        cb.on_step_end(1, metrics)
        assert "exp_grad_norm_mean" in metrics
        assert "exp_grad_norm_std" in metrics
        assert "exp_grad_norm_max_layer" in metrics
        assert metrics["exp_grad_norm_mean"] > 0

    def test_skips_without_probe(self, tiny_model):
        """No metrics injected before probe batch is set."""
        cb = GradientStatsCallback(tiny_model, _loss_fn, log_interval=1)
        cb.on_train_begin(TrainConfig())
        metrics: dict = {"loss": 1.0}
        cb.on_step_end(1, metrics)
        assert "exp_grad_norm_mean" not in metrics

    def test_restores_training_mode(self, tiny_model):
        """Model stays in training mode after measurement."""
        cb = GradientStatsCallback(tiny_model, _loss_fn, log_interval=1)
        cb.on_train_begin(TrainConfig())
        x = mx.random.randint(0, 256, shape=(2, 16))
        y = mx.random.randint(0, 256, shape=(2, 16))
        cb.set_probe_batch((x, y))
        tiny_model.train()
        cb.on_step_end(1, {"loss": 1.0})
        assert tiny_model.training


class TestWeightStats:
    def test_injects_metrics(self, tiny_model):
        """exp_weight_norm present after measurement."""
        cb = WeightStatsCallback(tiny_model, log_interval=1)
        cb.on_train_begin(TrainConfig())
        metrics: dict = {"loss": 1.0}
        cb.on_step_end(1, metrics)
        assert "exp_weight_norm" in metrics
        assert metrics["exp_weight_norm"] > 0

    def test_delta_increases_after_step(self, tiny_model):
        """exp_weight_delta increases after a training step."""
        cb = WeightStatsCallback(tiny_model, log_interval=1)
        cb.on_train_begin(TrainConfig())

        # Initial: delta should be ~0
        m1: dict = {"loss": 1.0}
        cb.on_step_end(1, m1)
        initial_delta = m1["exp_weight_delta"]
        assert initial_delta < 1e-5

        # Modify weights to simulate training
        params = tiny_model.trainable_parameters()
        flat = tree_flatten(params)
        for name, p in flat:
            # Small perturbation
            tiny_model.update(tree_unflatten([(name, p + 0.01)]))
        mx.eval(tiny_model.parameters())

        m2: dict = {"loss": 1.0}
        cb.on_step_end(2, m2)
        assert m2["exp_weight_delta"] > initial_delta


class TestActivationStats:
    def test_injects_metrics(self, tiny_model):
        """exp_act_norm_ratio present after measurement."""
        probe = mx.random.randint(0, 256, shape=(2, 16))
        cb = ActivationStatsCallback(tiny_model, probe, eval_interval=1)
        cb.on_train_begin(TrainConfig())
        metrics: dict = {"loss": 1.0}
        cb.on_step_end(1, metrics)
        assert "exp_act_norm_ratio" in metrics
        assert "exp_act_sparsity_mean" in metrics
        assert metrics["exp_act_norm_ratio"] > 0

    def test_restores_training_mode(self, tiny_model):
        """Model stays in training mode after measurement."""
        probe = mx.random.randint(0, 256, shape=(2, 16))
        cb = ActivationStatsCallback(tiny_model, probe, eval_interval=1)
        tiny_model.train()
        cb.on_step_end(1, {"loss": 1.0})
        assert tiny_model.training


class TestAttentionEntropy:
    def test_injects_metrics(self, tiny_model):
        """exp_attn_entropy_mean present after measurement."""
        probe = mx.random.randint(0, 256, shape=(2, 8))
        cb = AttentionEntropyCallback(tiny_model, probe, eval_interval=1)
        cb.on_train_begin(TrainConfig())
        metrics: dict = {"loss": 1.0}
        cb.on_step_end(1, metrics)
        assert "exp_attn_entropy_mean" in metrics
        assert "exp_attn_entropy_std" in metrics
        # Entropy should be positive
        assert metrics["exp_attn_entropy_mean"] > 0

    def test_restores_training_mode(self, tiny_model):
        """Model stays in training mode after measurement."""
        probe = mx.random.randint(0, 256, shape=(2, 8))
        cb = AttentionEntropyCallback(tiny_model, probe, eval_interval=1)
        tiny_model.train()
        cb.on_step_end(1, {"loss": 1.0})
        assert tiny_model.training


class TestLossCurvature:
    def test_grad_noise_scale_after_window(self):
        """exp_grad_noise_scale present after enough steps."""
        cb = LossCurvatureCallback(window_size=3)
        cb.on_train_begin(TrainConfig())

        # Not enough data yet
        m1: dict = {"grad_norm": 1.0}
        cb.on_step_end(1, m1)
        assert "exp_grad_noise_scale" not in m1

        # After 2 steps, window has enough
        m2: dict = {"grad_norm": 1.5}
        cb.on_step_end(2, m2)
        assert "exp_grad_noise_scale" in m2
        assert m2["exp_grad_noise_scale"] > 0

    def test_no_inject_without_grad_norm(self):
        """No metrics if grad_norm missing from dict."""
        cb = LossCurvatureCallback(window_size=3)
        cb.on_train_begin(TrainConfig())
        metrics: dict = {"loss": 1.0}
        cb.on_step_end(1, metrics)
        assert "exp_grad_noise_scale" not in metrics


class TestEffectiveRank:
    def test_injects_metrics(self, tiny_model):
        """exp_effective_rank_mean present after measurement."""
        cb = EffectiveRankCallback(tiny_model, eval_interval=1)
        cb.on_train_begin(TrainConfig())
        metrics: dict = {"loss": 1.0}
        cb.on_step_end(1, metrics)
        assert "exp_effective_rank_mean" in metrics
        assert metrics["exp_effective_rank_mean"] > 1.0


class TestExpPrefix:
    def test_all_use_exp_prefix(self, tiny_model):
        """All injected keys start with exp_."""
        callbacks = [
            WeightStatsCallback(tiny_model, log_interval=1),
            LossCurvatureCallback(window_size=2),
            EffectiveRankCallback(tiny_model, eval_interval=1),
        ]
        for cb in callbacks:
            cb.on_train_begin(TrainConfig())

        metrics: dict = {
            "loss": 1.0,
            "grad_norm": 1.0,
        }
        # Run two steps for LossCurvature window
        for cb in callbacks:
            cb.on_step_end(1, metrics)
        metrics2: dict = {
            "loss": 1.0,
            "grad_norm": 1.5,
        }
        for cb in callbacks:
            cb.on_step_end(2, metrics2)

        exp_keys = [k for k in metrics2 if k.startswith("exp_")]
        assert len(exp_keys) >= 3
        # Verify no non-standard keys were injected
        for k in metrics2:
            if k not in ("loss", "grad_norm"):
                assert k.startswith("exp_"), f"Key {k} missing exp_ prefix"


class TestMlflowPrefixRouting:
    def test_exp_keys_routed_to_experiment_group(self):
        """exp_* keys routed to 4_experiment/ in MLflow."""
        from lmxlab.experiments.mlflow import _prefix_metrics

        metrics = {
            "loss": 1.0,
            "exp_grad_norm_mean": 0.5,
            "exp_weight_norm": 10.0,
            "tokens_per_sec": 1000.0,
        }
        prefixed = _prefix_metrics(metrics)
        assert "4_experiment/exp_grad_norm_mean" in prefixed
        assert "4_experiment/exp_weight_norm" in prefixed
        assert "1_core/loss" in prefixed
        assert "2_efficiency/tokens_per_sec" in prefixed
