"""Direct Preference Optimization (DPO) training."""

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.models.base import LanguageModel


def dpo_loss(
    model: LanguageModel,
    ref_model: LanguageModel,
    chosen: mx.array,
    rejected: mx.array,
    beta: float = 0.1,
) -> mx.array:
    """Compute DPO loss.

    DPO directly optimizes preferences without reward modeling.
    L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    Args:
        model: Policy model being trained.
        ref_model: Reference (frozen) model.
        chosen: Preferred sequence token IDs (batch, seq_len).
        rejected: Dispreferred sequence token IDs (batch, seq_len).
        beta: Temperature parameter controlling deviation from ref.

    Returns:
        Scalar DPO loss.
    """
    # Compute log probs for chosen sequences
    chosen_logits, _ = model(chosen[:, :-1])
    chosen_ref_logits, _ = ref_model(chosen[:, :-1])
    chosen_targets = chosen[:, 1:]

    chosen_logps = _sequence_log_probs(chosen_logits, chosen_targets)
    chosen_ref_logps = _sequence_log_probs(chosen_ref_logits, chosen_targets)

    # Compute log probs for rejected sequences
    rejected_logits, _ = model(rejected[:, :-1])
    rejected_ref_logits, _ = ref_model(rejected[:, :-1])
    rejected_targets = rejected[:, 1:]

    rejected_logps = _sequence_log_probs(rejected_logits, rejected_targets)
    rejected_ref_logps = _sequence_log_probs(
        rejected_ref_logits, rejected_targets
    )

    # DPO objective
    chosen_rewards = beta * (chosen_logps - chosen_ref_logps)
    rejected_rewards = beta * (rejected_logps - rejected_ref_logps)

    # log_sigmoid(x) = -softplus(-x) for numerical stability
    loss = mx.logaddexp(0, -(chosen_rewards - rejected_rewards))
    return mx.mean(loss)


def _sequence_log_probs(
    logits: mx.array,
    targets: mx.array,
) -> mx.array:
    """Compute per-sequence log probabilities.

    Args:
        logits: Model logits (batch, seq_len, vocab).
        targets: Target token IDs (batch, seq_len).

    Returns:
        Sum of log probs per sequence (batch,).
    """
    log_probs = nn.log_softmax(logits, axis=-1)
    # Gather log probs for target tokens
    target_log_probs = mx.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)
    return mx.sum(target_log_probs, axis=-1)
