"""Autoregressive text generation with sampling strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from lmt_metal.models.base import LanguageModel


def _sample_top_p(logits: mx.array, top_p: float) -> mx.array:
    """Nucleus (top-p) sampling.

    Args:
        logits: Logits for next token (batch, vocab).
        top_p: Cumulative probability threshold.

    Returns:
        Sampled token IDs (batch, 1).
    """
    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)

    # Zero out tokens beyond top-p threshold
    mask = cumulative - sorted_probs > top_p
    sorted_probs = mx.where(mask, 0.0, sorted_probs)

    # Re-normalize and sample
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
    token = mx.random.categorical(mx.log(sorted_probs + 1e-10))
    # Map back to original indices
    return mx.take_along_axis(sorted_indices, token[:, None], axis=-1)


def _sample_top_k(logits: mx.array, top_k: int) -> mx.array:
    """Top-k sampling.

    Args:
        logits: Logits for next token (batch, vocab).
        top_k: Number of top tokens to keep.

    Returns:
        Sampled token IDs (batch, 1).
    """
    # Get top-k values and indices
    top_values = mx.topk(logits, k=top_k, axis=-1)
    top_indices = mx.argpartition(-logits, kth=top_k, axis=-1)[:, :top_k]

    # Sort by value for proper sampling
    sorted_order = mx.argsort(-top_values, axis=-1)
    top_values = mx.take_along_axis(top_values, sorted_order, axis=-1)
    top_indices = mx.take_along_axis(top_indices, sorted_order, axis=-1)

    # Sample from top-k
    token_idx = mx.random.categorical(top_values)
    return mx.take_along_axis(top_indices, token_idx[:, None], axis=-1)


def generate(
    model: LanguageModel,
    prompt: mx.array,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> mx.array:
    """Generate tokens autoregressively with KV caching.

    Args:
        model: Language model to generate from.
        prompt: Input token IDs of shape (batch, prompt_len).
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: If > 0, only sample from top-k tokens.
        top_p: If < 1.0, use nucleus sampling.

    Returns:
        Generated token IDs of shape
        (batch, prompt_len + max_tokens).
    """
    tokens = prompt
    cache = None

    # Process prompt (prefill)
    logits, cache = model(tokens, cache=cache)
    mx.eval(logits, *[c for pair in cache for c in pair])

    generated = []
    for _ in range(max_tokens):
        # Get logits for last position
        next_logits = logits[:, -1, :]

        # Apply temperature
        if temperature > 0:
            next_logits = next_logits / temperature

            if top_k > 0:
                next_token = _sample_top_k(next_logits, top_k)
            elif top_p < 1.0:
                next_token = _sample_top_p(next_logits, top_p)
            else:
                next_token = mx.random.categorical(next_logits)[:, None]
        else:
            # Greedy
            next_token = mx.argmax(next_logits, axis=-1, keepdims=True)

        generated.append(next_token)

        # Forward pass with single token + cache
        logits, cache = model(next_token, cache=cache)
        mx.eval(logits, *[c for pair in cache for c in pair])

    if generated:
        all_generated = mx.concatenate(generated, axis=1)
        return mx.concatenate([prompt, all_generated], axis=1)
    return prompt
