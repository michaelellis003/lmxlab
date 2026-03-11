"""Evaluation metrics for language models."""

import math
from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from lmxlab.models.base import LanguageModel


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


def pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """Compute pass@k metric (Chen et al., 2021).

    Estimates the probability that at least one of k samples
    passes a given test, given that c of n total samples pass.
    Uses the unbiased estimator from the Codex paper.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of generated samples.
        c: Number of samples that pass the test.
        k: Number of samples to consider.

    Returns:
        pass@k probability in [0, 1].

    Example::

        # 10 samples generated, 3 pass the test
        p1 = pass_at_k(n=10, c=3, k=1)   # ~0.30
        p5 = pass_at_k(n=10, c=3, k=5)   # ~0.83
    """
    if n - c < k:
        return 1.0
    # Use log-space for numerical stability
    # pass@k = 1 - prod((n-c-i)/(n-i) for i in range(k))
    log_prod = 0.0
    for i in range(k):
        log_prod += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_prod)


def evaluate_pass_at_k(
    completions: list[list[str]],
    test_fn: Callable[[str], bool],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate pass@k over multiple problems.

    For each problem, generates N completions and checks how
    many pass using the provided test function.

    Args:
        completions: List of problems, each a list of N
            completion strings.
        test_fn: Function that returns True if a completion
            is correct.
        k_values: Values of k to evaluate. Default: [1, 5, 10].

    Returns:
        Dict mapping 'pass@k' to the average score across
        problems.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    results: dict[str, list[float]] = {f"pass@{k}": [] for k in k_values}

    for problem_completions in completions:
        n = len(problem_completions)
        c = sum(1 for comp in problem_completions if test_fn(comp))
        for k in k_values:
            if k <= n:
                score = pass_at_k(n, c, k)
                results[f"pass@{k}"].append(score)

    return {
        key: sum(vals) / len(vals) if vals else 0.0
        for key, vals in results.items()
    }
