"""Autoregressive text generation with sampling strategies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from lmxlab.models.base import LanguageModel


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


def _apply_repetition_penalty(
    logits: mx.array,
    generated_ids: list[mx.array],
    penalty: float,
) -> mx.array:
    """Apply repetition penalty to discourage repeated tokens.

    Tokens that have already been generated get their logits
    divided by the penalty (if positive) or multiplied (if
    negative). A penalty of 1.0 has no effect.

    This follows the approach from Keskar et al. (2019,
    arXiv:1909.05858): for each
    previously generated token, divide its logit by the penalty
    if the logit is positive, or multiply by the penalty if
    negative. This consistently reduces the probability of the
    token regardless of logit sign.

    Args:
        logits: Raw logits (batch, vocab).
        generated_ids: List of previously generated token arrays.
        penalty: Repetition penalty factor (> 1.0 discourages
            repetition).

    Returns:
        Modified logits.
    """
    if not generated_ids or penalty == 1.0:
        return logits

    # Collect all generated token IDs into a set per batch
    all_ids = mx.concatenate(generated_ids, axis=1)  # (batch, n)

    # Build a mask of which vocab entries have been generated.
    # Compare each vocab index against all generated token IDs to
    # find which positions need the penalty applied.
    batch_size, vocab_size = logits.shape
    vocab_range = mx.arange(vocab_size)[None, :]  # (1, V)
    # all_ids is (batch, n) -- check if any generated id matches
    mask = mx.zeros((batch_size, vocab_size))
    for i in range(all_ids.shape[1]):
        token_id = all_ids[:, i : i + 1]  # (batch, 1)
        mask = mx.maximum(mask, (vocab_range == token_id).astype(logits.dtype))  # type: ignore[union-attr]

    # Apply penalty: divide positive, multiply negative
    penalized = mx.where(logits > 0, logits / penalty, logits * penalty)
    # Only apply to tokens that appeared before
    return mx.where(mask > 0, penalized, logits)


def _sample_next(
    logits: mx.array,
    temperature: float,
    top_k: int,
    top_p: float,
) -> mx.array:
    """Sample next token from logits.

    Args:
        logits: Logits for next position (batch, vocab).
        temperature: Sampling temperature (0 = greedy).
        top_k: If > 0, only sample from top-k.
        top_p: If < 1.0, use nucleus sampling.

    Returns:
        Token IDs (batch, 1).
    """
    if temperature > 0:
        scaled = logits / temperature
        if top_k > 0:
            return _sample_top_k(scaled, top_k)
        elif top_p < 1.0:
            return _sample_top_p(scaled, top_p)
        else:
            return mx.random.categorical(scaled)[:, None]
    else:
        return mx.argmax(logits, axis=-1, keepdims=True)


def generate(
    model: LanguageModel,
    prompt: mx.array,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_tokens: list[int] | None = None,
) -> mx.array:
    """Generate tokens autoregressively with KV caching.

    Args:
        model: Language model to generate from.
        prompt: Input token IDs of shape (batch, prompt_len).
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: If > 0, only sample from top-k tokens.
        top_p: If < 1.0, use nucleus sampling.
        repetition_penalty: Penalty for repeating tokens (> 1.0
            discourages repetition, 1.0 = no effect).
        stop_tokens: List of token IDs that stop generation.
            When any batch element generates a stop token,
            generation stops for all.

    Returns:
        Generated token IDs of shape
        (batch, prompt_len + generated_len).
    """
    tokens = prompt
    cache = None
    stop_set = set(stop_tokens) if stop_tokens else set()

    # Process prompt (prefill)
    logits, cache = model(tokens, cache=cache)
    mx.eval(logits, *[c for pair in cache for c in pair])

    generated: list[mx.array] = []
    for _ in range(max_tokens):
        next_logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            next_logits = _apply_repetition_penalty(
                next_logits, generated, repetition_penalty
            )

        next_token = _sample_next(next_logits, temperature, top_k, top_p)
        mx.eval(next_token)

        # Check stop tokens
        if stop_set:
            token_val = next_token[0, 0].item()
            if token_val in stop_set:
                break

        generated.append(next_token)

        logits, cache = model(next_token, cache=cache)
        mx.eval(logits, *[c for pair in cache for c in pair])

    if generated:
        all_generated = mx.concatenate(generated, axis=1)
        return mx.concatenate([prompt, all_generated], axis=1)
    return prompt


def stream_generate(
    model: LanguageModel,
    prompt: mx.array,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_tokens: list[int] | None = None,
) -> Iterator[int]:
    """Generate tokens one at a time, yielding each as produced.

    This is the standard interface for interactive/streaming
    applications. Each token is yielded immediately after
    generation, enabling real-time display.

    Args:
        model: Language model to generate from.
        prompt: Input token IDs of shape (1, prompt_len).
        max_tokens: Maximum number of new tokens.
        temperature: Sampling temperature (0 = greedy).
        top_k: If > 0, only sample from top-k.
        top_p: If < 1.0, use nucleus sampling.
        repetition_penalty: Penalty for repeating tokens.
        stop_tokens: Token IDs that stop generation.

    Yields:
        Generated token IDs one at a time.
    """
    cache = None
    stop_set = set(stop_tokens) if stop_tokens else set()

    # Prefill
    logits, cache = model(prompt, cache=cache)
    mx.eval(logits, *[c for pair in cache for c in pair])

    generated: list[mx.array] = []
    for _ in range(max_tokens):
        next_logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            next_logits = _apply_repetition_penalty(
                next_logits, generated, repetition_penalty
            )

        next_token = _sample_next(next_logits, temperature, top_k, top_p)
        mx.eval(next_token)

        token_id = next_token[0, 0].item()

        if stop_set and token_id in stop_set:
            return

        generated.append(next_token)
        yield token_id

        logits, cache = model(next_token, cache=cache)
        mx.eval(logits, *[c for pair in cache for c in pair])
