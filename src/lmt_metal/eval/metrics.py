"""Evaluation metrics for language models."""

import math

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.models.base import LanguageModel


def _compute_loss(
    model: LanguageModel,
    tokens: mx.array,
) -> mx.array:
    """Compute cross-entropy loss over a sequence.

    Args:
        model: Language model.
        tokens: Token IDs of shape (batch, seq_len).

    Returns:
        Mean cross-entropy loss (scalar).
    """
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits, _ = model(x)
    logits = logits.reshape(-1, logits.shape[-1])
    targets = y.reshape(-1)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def perplexity(
    model: LanguageModel,
    data: list[mx.array],
) -> float:
    """Compute perplexity over a dataset.

    PPL = exp(average cross-entropy loss)

    Args:
        model: Language model.
        data: List of token ID arrays, each (batch, seq_len).

    Returns:
        Perplexity score (lower is better).
    """
    total_loss = 0.0
    n_batches = 0

    for tokens in data:
        loss = _compute_loss(model, tokens)
        mx.eval(loss)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return math.exp(avg_loss)


def bits_per_byte(
    model: LanguageModel,
    data: list[mx.array],
    bytes_per_token: float = 1.0,
) -> float:
    """Compute bits-per-byte (BPB).

    BPB = (cross-entropy in nats) / (ln(2) * bytes_per_token)

    For character-level tokenizers, bytes_per_token ≈ 1.0.
    For BPE tokenizers, estimate from data.

    Args:
        model: Language model.
        data: List of token ID arrays.
        bytes_per_token: Average bytes per token.

    Returns:
        BPB score (lower is better).
    """
    total_loss = 0.0
    n_batches = 0

    for tokens in data:
        loss = _compute_loss(model, tokens)
        mx.eval(loss)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss / (math.log(2) * bytes_per_token)
