"""GRPO Trainer: reinforcement learning from rewards.

Implements the Group Relative Policy Optimization (GRPO) training
loop from DeepSeek-R1 (arXiv:2501.12948). For each prompt, the
policy model generates ``group_size`` completions, scores them
with a reward function, and optimizes using the clipped surrogate
GRPO objective.

References:
- DeepSeek-R1 (arXiv:2501.12948)
- HuggingFace TRL GRPOTrainer
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.training.callbacks import Callback
from lmxlab.training.grpo import grpo_loss


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO training.

    Args:
        group_size: Number of completions per prompt.
        max_gen_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature for generation.
        beta: KL penalty coefficient.
        epsilon: Clipping range for surrogate objective.
        learning_rate: Optimizer learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    group_size: int = 4
    max_gen_tokens: int = 256
    temperature: float = 0.8
    beta: float = 0.1
    epsilon: float = 0.2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0


class GRPOTrainer:
    """GRPO training loop.

    Generates completions, scores them with a reward function,
    and optimizes the policy model using clipped surrogate GRPO.

    Args:
        model: Policy model to train.
        ref_model: Frozen reference model for KL penalty.
        config: GRPO training configuration.
        reward_fn: Callable that scores (prompt, completion)
            pairs. Takes two ``mx.array`` arguments and returns
            a scalar ``float`` reward.
        optimizer: MLX optimizer instance.
        callbacks: Optional list of training callbacks.

    Example:
        >>> trainer = GRPOTrainer(
        ...     model, ref_model, GRPOConfig(),
        ...     reward_fn=lambda p, c: 1.0,
        ...     optimizer=optim.Adam(learning_rate=1e-5),
        ... )
        >>> trainer.train(prompt_iterator, n_steps=100)
    """

    def __init__(
        self,
        model: LanguageModel,
        ref_model: LanguageModel,
        config: GRPOConfig,
        reward_fn: Callable[[mx.array, mx.array], float],
        optimizer: optim.Optimizer,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.callbacks = callbacks or []

        # Build value_and_grad for the GRPO loss
        self._loss_and_grad = nn.value_and_grad(model, self._compute_loss)

    def _compute_loss(
        self,
        model: LanguageModel,
        prompts: mx.array,
        completions: mx.array,
        rewards: mx.array,
    ) -> mx.array:
        """Compute GRPO loss for gradient computation.

        Args:
            model: Policy model (passed by value_and_grad).
            prompts: Prompt tokens (group_size, prompt_len).
            completions: Full sequences
                (group_size, total_len).
            rewards: Scalar rewards (group_size,).

        Returns:
            Scalar GRPO loss.
        """
        return grpo_loss(
            model,
            self.ref_model,
            prompts,
            completions,
            rewards,
            beta=self.config.beta,
            epsilon=self.config.epsilon,
        )

    def _generate_completions(
        self,
        prompt: mx.array,
    ) -> mx.array:
        """Generate group_size completions for a prompt.

        Args:
            prompt: Single prompt (1, prompt_len).

        Returns:
            Full sequences (group_size, prompt_len + gen_len).
        """
        # Expand prompt to group_size copies
        prompts = mx.broadcast_to(
            prompt,
            (self.config.group_size, prompt.shape[1]),
        )
        # Force copy so broadcast doesn't interfere
        prompts = mx.array(prompts)

        completions = generate(
            self.model,
            prompts,
            max_tokens=self.config.max_gen_tokens,
            temperature=self.config.temperature,
        )
        return completions

    def _score_completions(
        self,
        prompt: mx.array,
        completions: mx.array,
    ) -> mx.array:
        """Score completions with the reward function.

        Args:
            prompt: Single prompt (1, prompt_len).
            completions: Full sequences
                (group_size, total_len).

        Returns:
            Rewards (group_size,).
        """
        rewards = []
        for i in range(completions.shape[0]):
            r = self.reward_fn(prompt[0], completions[i])
            rewards.append(r)
        return mx.array(rewards)

    def train(
        self,
        prompt_iterator: Iterator[mx.array],
        n_steps: int,
    ) -> list[dict[str, Any]]:
        """Run the GRPO training loop.

        Args:
            prompt_iterator: Yields prompt tensors of shape
                (1, prompt_len).
            n_steps: Number of optimization steps.

        Returns:
            List of per-step metrics dicts.
        """
        history: list[dict[str, Any]] = []

        for cb in self.callbacks:
            cb.on_train_begin(None)

        for step in range(1, n_steps + 1):
            prompt = next(prompt_iterator)
            if prompt.ndim == 1:
                prompt = prompt[None, :]

            # Generate completions
            completions = self._generate_completions(prompt)
            mx.eval(completions)

            # Score with reward function
            rewards = self._score_completions(prompt, completions)

            # Expand prompt to match group_size
            prompts = mx.broadcast_to(
                prompt,
                (self.config.group_size, prompt.shape[1]),
            )
            prompts = mx.array(prompts)

            # Compute loss and gradients
            loss, grads = self._loss_and_grad(
                self.model,
                prompts,
                completions,
                rewards,
            )

            # Clip gradients
            grads, grad_norm = optim.clip_grad_norm(
                grads, max_norm=self.config.max_grad_norm
            )

            # Update model
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)

            metrics = {
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "mean_reward": mx.mean(rewards).item(),
            }
            history.append(metrics)

            for cb in self.callbacks:
                cb.on_step_end(step, metrics)

        for cb in self.callbacks:
            cb.on_train_end(history)

        return history
