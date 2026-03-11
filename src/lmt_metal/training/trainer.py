"""Compiled training loop using MLX idioms."""

from collections.abc import Iterator
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from lmt_metal.models.base import LanguageModel
from lmt_metal.training.callbacks import Callback
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.optimizers import create_optimizer


def _loss_fn(
    model: LanguageModel,
    x: mx.array,
    y: mx.array,
) -> mx.array:
    """Cross-entropy loss for language modeling.

    Args:
        model: Language model.
        x: Input token IDs (batch, seq_len).
        y: Target token IDs (batch, seq_len).

    Returns:
        Scalar loss value.
    """
    logits, _ = model(x)
    # Reshape for cross_entropy: (batch * seq_len, vocab)
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


class Trainer:
    """Training loop with compiled steps and gradient accumulation.

    Uses nn.value_and_grad for functional gradient computation
    and mx.compile for the full training step. When
    ``grad_accumulation_steps > 1``, gradients are averaged over
    multiple micro-batches before a single optimizer update.

    Args:
        model: Language model to train.
        config: Training configuration.
        optimizer: Optional pre-built optimizer.
        callbacks: Optional list of callbacks.
    """

    def __init__(
        self,
        model: LanguageModel,
        config: TrainConfig,
        optimizer: optim.Optimizer | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer or create_optimizer(config)
        self.callbacks = callbacks or []
        self.step = 0
        self._accum_steps = config.grad_accumulation_steps

        # Build the training step function
        self._loss_and_grad = nn.value_and_grad(model, _loss_fn)

        if self._accum_steps <= 1:
            # No accumulation: compile full step (fwd + bwd + update)
            if config.compile_step:
                self._step_fn = mx.compile(
                    self._single_step,
                    inputs=model.trainable_parameters(),
                    outputs=model.trainable_parameters(),
                )
            else:
                self._step_fn = self._single_step

    def _single_step(
        self,
        x: mx.array,
        y: mx.array,
    ) -> mx.array:
        """Single training step: forward + backward + update.

        Args:
            x: Input tokens (batch, seq_len).
            y: Target tokens (batch, seq_len).

        Returns:
            Scalar loss value.
        """
        loss, grads = self._loss_and_grad(self.model, x, y)

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            grads, _ = optim.clip_grad_norm(
                grads, max_norm=self.config.max_grad_norm
            )

        self.optimizer.update(self.model, grads)
        return loss

    def _accumulation_step(
        self,
        micro_batches: list[tuple[mx.array, mx.array]],
    ) -> mx.array:
        """Gradient accumulation step over multiple micro-batches.

        Computes gradients for each micro-batch, averages them,
        then applies a single optimizer update.

        Args:
            micro_batches: List of (input, target) micro-batches.

        Returns:
            Averaged loss across micro-batches.
        """
        n = len(micro_batches)
        total_loss = mx.array(0.0)
        acc_grads = None

        for x, y in micro_batches:
            loss, grads = self._loss_and_grad(self.model, x, y)
            total_loss = total_loss + loss
            if acc_grads is None:
                acc_grads = grads
            else:
                acc_grads = tree_map(lambda a, b: a + b, acc_grads, grads)

        # Average gradients
        avg_grads = tree_map(lambda g: g / n, acc_grads)
        avg_loss = total_loss / n

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            avg_grads, _ = optim.clip_grad_norm(
                avg_grads, max_norm=self.config.max_grad_norm
            )

        self.optimizer.update(self.model, avg_grads)
        return avg_loss

    def train_step(
        self,
        batch: tuple[mx.array, mx.array],
    ) -> dict[str, float]:
        """Execute one training step with eval boundary.

        Args:
            batch: Tuple of (input_tokens, target_tokens).

        Returns:
            Dict with 'loss' and 'learning_rate'.
        """
        x, y = batch
        loss = self._step_fn(x, y)

        # Explicit eval boundary
        mx.eval(loss, self.model.parameters(), self.optimizer.state)

        self.step += 1
        lr = self.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.step)

        metrics = {
            "loss": loss.item(),
            "learning_rate": float(lr),
        }

        for cb in self.callbacks:
            cb.on_step_end(self.step, metrics)

        return metrics

    def train_step_accumulated(
        self,
        micro_batches: list[tuple[mx.array, mx.array]],
    ) -> dict[str, float]:
        """Execute one accumulated training step.

        Averages gradients over micro-batches, then updates once.

        Args:
            micro_batches: List of (input, target) micro-batches.

        Returns:
            Dict with 'loss' and 'learning_rate'.
        """
        loss = self._accumulation_step(micro_batches)

        # Explicit eval boundary
        mx.eval(loss, self.model.parameters(), self.optimizer.state)

        self.step += 1
        lr = self.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.step)

        metrics = {
            "loss": loss.item(),
            "learning_rate": float(lr),
        }

        for cb in self.callbacks:
            cb.on_step_end(self.step, metrics)

        return metrics

    def train(
        self,
        train_data: Iterator[tuple[mx.array, mx.array]],
        eval_data: Iterator[tuple[mx.array, mx.array]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the full training loop.

        When ``grad_accumulation_steps > 1``, collects that many
        micro-batches before each optimizer update.

        Args:
            train_data: Iterator yielding (input, target) batches.
            eval_data: Optional eval data iterator.

        Returns:
            List of per-step metrics dicts.
        """
        for cb in self.callbacks:
            cb.on_train_begin(self.config)

        history: list[dict[str, Any]] = []

        if self._accum_steps > 1:
            history = self._train_accumulated(train_data, eval_data)
        else:
            history = self._train_simple(train_data, eval_data)

        for cb in self.callbacks:
            cb.on_train_end(history)

        return history

    def _train_simple(
        self,
        train_data: Iterator[tuple[mx.array, mx.array]],
        eval_data: Iterator[tuple[mx.array, mx.array]] | None,
    ) -> list[dict[str, Any]]:
        """Training loop without gradient accumulation."""
        history: list[dict[str, Any]] = []

        for batch in train_data:
            if self.step >= self.config.max_steps:
                break

            metrics = self.train_step(batch)
            history.append(metrics)
            self._maybe_eval(eval_data, metrics)

        return history

    def _train_accumulated(
        self,
        train_data: Iterator[tuple[mx.array, mx.array]],
        eval_data: Iterator[tuple[mx.array, mx.array]] | None,
    ) -> list[dict[str, Any]]:
        """Training loop with gradient accumulation."""
        history: list[dict[str, Any]] = []
        micro_batches: list[tuple[mx.array, mx.array]] = []

        for batch in train_data:
            if self.step >= self.config.max_steps:
                break

            micro_batches.append(batch)

            if len(micro_batches) == self._accum_steps:
                metrics = self.train_step_accumulated(micro_batches)
                history.append(metrics)
                self._maybe_eval(eval_data, metrics)
                micro_batches = []

        # Handle remaining micro-batches
        if micro_batches and self.step < self.config.max_steps:
            metrics = self.train_step_accumulated(micro_batches)
            history.append(metrics)

        return history

    def _maybe_eval(
        self,
        eval_data: Iterator[tuple[mx.array, mx.array]] | None,
        metrics: dict[str, Any],
    ) -> None:
        """Run evaluation if due."""
        if (
            eval_data is not None
            and self.step % self.config.eval_interval == 0
        ):
            eval_metrics = self.evaluate(eval_data)
            metrics.update(eval_metrics)
            for cb in self.callbacks:
                cb.on_eval_end(self.step, eval_metrics)

    def evaluate(
        self,
        eval_data: Iterator[tuple[mx.array, mx.array]],
    ) -> dict[str, float]:
        """Run evaluation over the eval dataset.

        Args:
            eval_data: Iterator yielding (input, target) batches.

        Returns:
            Dict with 'eval_loss'.
        """
        total_loss = 0.0
        n_batches = 0

        for x, y in eval_data:
            logits, _ = self.model(x)
            logits = logits.reshape(-1, logits.shape[-1])
            targets = y.reshape(-1)
            loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
            mx.eval(loss)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {"eval_loss": avg_loss}
