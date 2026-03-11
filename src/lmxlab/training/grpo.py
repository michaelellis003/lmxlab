"""Group Relative Policy Optimization (GRPO)."""

import mlx.core as mx

from lmxlab.models.base import LanguageModel
from lmxlab.training.dpo import _sequence_log_probs


def grpo_loss(
    model: LanguageModel,
    ref_model: LanguageModel,
    prompts: mx.array,
    completions: mx.array,
    rewards: mx.array,
    beta: float = 0.1,
    epsilon: float = 0.2,
) -> mx.array:
    """Compute GRPO loss.

    GRPO uses group-relative rewards: for each prompt, generate
    multiple completions, compute rewards, normalize within the
    group, and optimize using a clipped surrogate objective.

    Args:
        model: Policy model being trained.
        ref_model: Reference (frozen) model.
        prompts: Prompt token IDs (batch, prompt_len).
        completions: Full sequences (batch, total_len).
        rewards: Scalar rewards per completion (batch,).
        beta: KL penalty coefficient.
        epsilon: Clipping range for surrogate objective.

    Returns:
        Scalar GRPO loss.
    """
    # Normalize rewards within the group (zero mean, unit variance)
    reward_mean = mx.mean(rewards)
    reward_std = mx.maximum(mx.std(rewards), mx.array(1e-8))
    advantages = (rewards - reward_mean) / reward_std

    # Compute log probs for completions
    inputs = completions[:, :-1]
    targets = completions[:, 1:]

    logits, _ = model(inputs)
    ref_logits, _ = ref_model(inputs)

    log_probs = _sequence_log_probs(logits, targets)
    ref_log_probs = _sequence_log_probs(ref_logits, targets)

    # Ratio and clipped surrogate
    ratio = mx.exp(log_probs - ref_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)

    surrogate = mx.minimum(
        ratio * advantages,
        clipped_ratio * advantages,
    )

    # KL penalty
    kl = ref_log_probs - log_probs

    loss = -(surrogate - beta * kl)
    return mx.mean(loss)
