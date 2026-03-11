"""Evaluation framework: metrics and benchmarks."""

from lmxlab.eval.metrics import (
    bits_per_byte,
    evaluate_pass_at_k,
    pass_at_k,
    perplexity,
)

__all__ = [
    "bits_per_byte",
    "evaluate_pass_at_k",
    "pass_at_k",
    "perplexity",
]
