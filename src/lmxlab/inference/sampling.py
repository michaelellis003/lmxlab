"""Advanced sampling strategies: best-of-N, majority vote."""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate


def best_of_n(
    model: LanguageModel,
    prompt: mx.array,
    n: int = 4,
    max_tokens: int = 100,
    temperature: float = 0.8,
    score_fn: str = "log_prob",
) -> mx.array:
    """Generate N completions and return the best one.

    Args:
        model: Language model.
        prompt: Input token IDs (1, prompt_len).
        n: Number of candidate completions.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        score_fn: Scoring function
            ('log_prob' or 'length_normalized').

    Returns:
        Best completion token IDs (1, total_len).
    """
    # Generate N completions by repeating prompt
    prompts = mx.repeat(prompt, repeats=n, axis=0)
    completions = generate(
        model,
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    mx.eval(completions)

    # Score each completion
    scores = _score_sequences(model, completions, prompt.shape[1])
    mx.eval(scores)

    if score_fn == "length_normalized":
        gen_len = completions.shape[1] - prompt.shape[1]
        scores = scores / gen_len

    # Return best
    best_idx = mx.argmax(scores).item()
    return completions[best_idx : best_idx + 1]


def _score_sequences(
    model: LanguageModel,
    sequences: mx.array,
    prompt_len: int,
) -> mx.array:
    """Score sequences by total log probability of generated tokens.

    Args:
        model: Language model.
        sequences: Full sequences (batch, seq_len).
        prompt_len: Length of the prompt prefix.

    Returns:
        Log probability scores (batch,).
    """
    logits, _ = model(sequences[:, :-1])
    mx.eval(logits)
    log_probs = nn.log_softmax(logits, axis=-1)
    targets = sequences[:, 1:]

    # Gather log probs for actual tokens
    token_log_probs = mx.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)

    # Only score generated tokens (after prompt)
    if prompt_len > 0:
        token_log_probs = token_log_probs[:, prompt_len - 1 :]

    return mx.sum(token_log_probs, axis=-1)


def majority_vote(
    model: LanguageModel,
    prompt: mx.array,
    n: int = 5,
    max_tokens: int = 50,
    temperature: float = 0.8,
) -> list[tuple[list[int], int]]:
    """Generate N completions and return majority vote results.

    Useful for tasks with discrete answers (e.g., math, code).
    Groups completions by content and returns counts.

    Args:
        model: Language model.
        prompt: Input token IDs (1, prompt_len).
        n: Number of completions to generate.
        max_tokens: Maximum tokens per completion.
        temperature: Sampling temperature.

    Returns:
        List of (token_list, count) sorted by count descending.
    """
    prompts = mx.repeat(prompt, repeats=n, axis=0)
    completions = generate(
        model,
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    mx.eval(completions)

    prompt_len = prompt.shape[1]

    # Group by generated content
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        gen = tuple(completions[i, prompt_len:].tolist())
        counts[gen] = counts.get(gen, 0) + 1

    # Sort by frequency
    results = [
        (list(tokens), count)
        for tokens, count in sorted(counts.items(), key=lambda x: -x[1])
    ]
    return results
