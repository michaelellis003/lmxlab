"""Tests for training infrastructure."""

import mlx.core as mx
import pytest

from lmt_metal.models.base import LanguageModel
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.training.callbacks import EarlyStopping, MetricsLogger
from lmt_metal.training.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.optimizers import (
    create_optimizer,
    create_schedule,
)
from lmt_metal.training.trainer import Trainer


@pytest.fixture
def tiny_model() -> LanguageModel:
    """Create a tiny GPT model for testing."""
    config = gpt_tiny()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def tiny_batches() -> list[tuple[mx.array, mx.array]]:
    """Create small training batches."""
    batches = []
    for _ in range(10):
        tokens = mx.random.randint(0, 256, shape=(4, 17))
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        batches.append((x, y))
    return batches


class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig()
        assert config.learning_rate == 3e-4
        assert config.compile_step is True

    def test_frozen(self):
        config = TrainConfig()
        with pytest.raises(AttributeError):
            config.learning_rate = 0.1  # type: ignore[misc]


class TestOptimizers:
    def test_adamw(self):
        config = TrainConfig(optimizer="adamw")
        opt = create_optimizer(config)
        assert opt is not None

    def test_lion(self):
        config = TrainConfig(optimizer="lion")
        opt = create_optimizer(config)
        assert opt is not None

    def test_unknown_raises(self):
        config = TrainConfig(optimizer="unknown")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(config)

    def test_schedule_cosine(self):
        config = TrainConfig(lr_schedule="cosine")
        schedule = create_schedule(config)
        assert schedule is not None

    def test_schedule_linear(self):
        config = TrainConfig(lr_schedule="linear")
        schedule = create_schedule(config)
        assert schedule is not None

    def test_unknown_schedule_raises(self):
        config = TrainConfig(lr_schedule="unknown")
        with pytest.raises(ValueError, match="Unknown schedule"):
            create_schedule(config)


class TestTrainer:
    def test_single_step(self, tiny_model, tiny_batches):
        """Training step runs and reduces loss."""
        config = TrainConfig(
            max_steps=1,
            compile_step=False,
            learning_rate=1e-3,
        )
        trainer = Trainer(tiny_model, config)
        metrics = trainer.train_step(tiny_batches[0])
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_overfit_tiny(self, tiny_model, tiny_batches):
        """Loss should decrease when overfitting a single batch."""
        config = TrainConfig(
            max_steps=20,
            compile_step=False,
            learning_rate=1e-3,
            log_interval=100,
        )
        trainer = Trainer(tiny_model, config)

        # Repeat single batch
        single_batch = tiny_batches[0]
        first_loss = None
        last_loss = None

        for _ in range(20):
            metrics = trainer.train_step(single_batch)
            if first_loss is None:
                first_loss = metrics["loss"]
            last_loss = metrics["loss"]

        assert last_loss < first_loss

    def test_train_loop(self, tiny_model, tiny_batches):
        """Full training loop runs to completion."""
        config = TrainConfig(
            max_steps=5,
            compile_step=False,
            learning_rate=1e-3,
            log_interval=100,
        )
        trainer = Trainer(tiny_model, config)
        history = trainer.train(iter(tiny_batches))
        assert len(history) == 5


class TestCheckpoints:
    def test_save_load_roundtrip(self, tiny_model, tmp_path):
        """Weights survive save/load roundtrip."""
        mx.eval(tiny_model.parameters())

        # Get original output
        x = mx.array([[1, 2, 3]])
        orig_logits, _ = tiny_model(x)
        mx.eval(orig_logits)

        # Save
        save_checkpoint(tmp_path / "ckpt", tiny_model, step=42)

        # Load into fresh model
        config = gpt_tiny()
        new_model = LanguageModel(config)
        meta = load_checkpoint(tmp_path / "ckpt", new_model)

        assert meta["step"] == 42

        new_logits, _ = new_model(x)
        mx.eval(new_logits)
        assert mx.allclose(orig_logits, new_logits).item()


class TestCallbacks:
    def test_metrics_logger(self, capsys):
        logger = MetricsLogger(log_interval=1)
        logger.on_train_begin(TrainConfig())
        logger.on_step_end(1, {"loss": 5.0, "learning_rate": 1e-4})
        captured = capsys.readouterr()
        assert "loss=5.0000" in captured.out

    def test_early_stopping(self):
        es = EarlyStopping(patience=2, min_delta=0.01)
        es.on_train_begin(TrainConfig())

        es.on_eval_end(1, {"eval_loss": 5.0})
        assert not es.should_stop

        es.on_eval_end(2, {"eval_loss": 4.0})
        assert not es.should_stop

        # No improvement
        es.on_eval_end(3, {"eval_loss": 4.0})
        assert not es.should_stop

        es.on_eval_end(4, {"eval_loss": 4.0})
        assert es.should_stop
