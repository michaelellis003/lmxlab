"""Tests for GRPOTrainer."""

import mlx.core as mx
import mlx.optimizers as optim

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.base import LanguageModel
from lmxlab.training.grpo_trainer import GRPOConfig, GRPOTrainer


def _make_tiny_model() -> LanguageModel:
    """Create a tiny model for testing."""
    cfg = ModelConfig(
        block=BlockConfig(
            d_model=32,
            n_heads=2,
            d_ff=64,
            attention="gqa",
            ffn="gated",
            norm="rms_norm",
            position="none",
            bias=False,
            pre_norm=True,
        ),
        vocab_size=64,
        n_layers=1,
    )
    model = LanguageModel(cfg)
    mx.eval(model.parameters())
    return model


def _dummy_reward(prompt: mx.array, completion: mx.array) -> float:
    """Reward = sequence length (trivial reward)."""
    return float(completion.shape[0])


class TestGRPOTrainer:
    def test_basic_training(self):
        """GRPOTrainer runs without errors for a few steps."""
        model = _make_tiny_model()
        ref_model = _make_tiny_model()

        config = GRPOConfig(
            group_size=2,
            max_gen_tokens=8,
            temperature=1.0,
            beta=0.1,
            epsilon=0.2,
            learning_rate=1e-4,
        )

        optimizer = optim.Adam(learning_rate=config.learning_rate)

        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config,
            reward_fn=_dummy_reward,
            optimizer=optimizer,
        )

        def prompt_iter():
            while True:
                yield mx.array([[1, 2, 3]])

        history = trainer.train(prompt_iter(), n_steps=2)
        assert len(history) == 2
        assert "loss" in history[0]
        assert "mean_reward" in history[0]

    def test_ref_model_unchanged(self):
        """Reference model weights are unchanged after training."""
        model = _make_tiny_model()
        ref_model = _make_tiny_model()

        # Save ref weights
        ref_params_before = {}
        for k, v in ref_model.parameters().items():
            if isinstance(v, mx.array):
                ref_params_before[k] = mx.array(v)

        config = GRPOConfig(
            group_size=2,
            max_gen_tokens=4,
            learning_rate=1e-3,
        )
        optimizer = optim.Adam(learning_rate=config.learning_rate)

        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config,
            reward_fn=_dummy_reward,
            optimizer=optimizer,
        )

        def prompt_iter():
            while True:
                yield mx.array([[1, 2]])

        trainer.train(prompt_iter(), n_steps=2)

        # Check ref weights unchanged
        for k, v_before in ref_params_before.items():
            v_after = ref_model.parameters()[k]
            if isinstance(v_after, mx.array):
                assert mx.array_equal(v_before, v_after), (
                    f"ref_model param {k} changed"
                )

    def test_group_size_completions(self):
        """Generates group_size completions per prompt."""
        model = _make_tiny_model()

        config = GRPOConfig(group_size=3, max_gen_tokens=4)

        # Test generation directly
        trainer = GRPOTrainer(
            model=model,
            ref_model=model,
            config=config,
            reward_fn=_dummy_reward,
            optimizer=optim.Adam(learning_rate=1e-4),
        )

        prompt = mx.array([[1, 2, 3]])
        completions = trainer._generate_completions(prompt)
        mx.eval(completions)
        assert completions.shape[0] == 3

    def test_callbacks_called(self):
        """Callbacks are invoked during training."""
        model = _make_tiny_model()

        call_log = []

        class LogCallback:
            def on_train_begin(self, config):
                call_log.append("begin")

            def on_step_end(self, step, metrics):
                call_log.append(f"step_{step}")

            def on_eval_end(self, step, metrics):
                pass

            def on_train_end(self, history):
                call_log.append("end")

        config = GRPOConfig(group_size=2, max_gen_tokens=4)
        optimizer = optim.Adam(learning_rate=1e-4)

        trainer = GRPOTrainer(
            model=model,
            ref_model=model,
            config=config,
            reward_fn=_dummy_reward,
            optimizer=optimizer,
            callbacks=[LogCallback()],
        )

        def prompt_iter():
            while True:
                yield mx.array([[1, 2]])

        trainer.train(prompt_iter(), n_steps=2)

        assert call_log[0] == "begin"
        assert "step_1" in call_log
        assert "step_2" in call_log
        assert call_log[-1] == "end"
