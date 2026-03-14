"""Beam search decoding.

Standard beam search with log-probability scoring. Optionally
supports a custom scoring function (e.g., RewardModel) for
reranking candidates at each step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from lmxlab.models.base import LanguageModel


def beam_search(
    model: LanguageModel,
    prompt: mx.array,
    beam_width: int = 4,
    max_tokens: int = 100,
    score_fn: Callable[[mx.array], mx.array] | None = None,
) -> list[tuple[mx.array, float]]:
    """Generate completions using beam search.

    Maintains ``beam_width`` candidate sequences and expands
    each by one token at each step, keeping the top-scoring
    beams. By default, scores by cumulative log probability.

    Args:
        model: Language model.
        prompt: Input token IDs (1, prompt_len).
        beam_width: Number of beams to maintain.
        max_tokens: Maximum tokens to generate.
        score_fn: Optional scoring function that takes
            sequences (beam_width, seq_len) and returns
            scores (beam_width,). If None, uses log-prob.

    Returns:
        List of (sequence, score) tuples sorted by score
        descending. Each sequence is (1, total_len).
    """
    if prompt.ndim == 1:
        prompt = prompt[None, :]

    B = prompt.shape[0]
    if B != 1:
        raise ValueError("beam_search expects a single prompt (batch=1)")

    # Initialize beams: (sequence, cumulative_log_prob)
    beams: list[tuple[mx.array, float]] = [
        (prompt, 0.0),
    ]

    for _ in range(max_tokens):
        all_candidates: list[tuple[mx.array, float]] = []

        # Batch all current beams for efficiency
        beam_seqs = mx.concatenate(
            [b[0] for b in beams], axis=0
        )  # (n_beams, seq_len)
        beam_scores = [b[1] for b in beams]

        logits, _ = model(beam_seqs)
        mx.eval(logits)
        # Get log probs for last position
        last_logits = logits[:, -1, :]  # (n_beams, vocab)
        log_probs = nn.log_softmax(last_logits, axis=-1)

        # Get top-k candidates per beam
        n_beams = len(beams)
        for i in range(n_beams):
            # Get top beam_width tokens for this beam
            top_k_vals = mx.topk(log_probs[i], k=beam_width)
            top_k_idx = mx.argpartition(-log_probs[i], kth=beam_width - 1)[
                :beam_width
            ]

            # Sort by value
            sort_order = mx.argsort(-top_k_vals)
            top_k_vals = mx.take(top_k_vals, sort_order)
            top_k_idx = mx.take(top_k_idx, sort_order)

            mx.eval(top_k_vals, top_k_idx)

            for j in range(beam_width):
                token = top_k_idx[j : j + 1][None, :]
                new_seq = mx.concatenate([beams[i][0], token], axis=1)
                new_score = beam_scores[i] + top_k_vals[j].item()
                all_candidates.append((new_seq, new_score))

        # Keep top beam_width candidates
        all_candidates.sort(key=lambda x: -x[1])
        beams = all_candidates[:beam_width]

    # Optional reranking with custom score_fn
    if score_fn is not None:
        beam_seqs = mx.concatenate([b[0] for b in beams], axis=0)
        scores = score_fn(beam_seqs)
        mx.eval(scores)
        if scores.ndim > 1:
            scores = scores.squeeze(-1)
        scored = [(beams[i][0], scores[i].item()) for i in range(len(beams))]
        scored.sort(key=lambda x: -x[1])
        return scored

    return beams
