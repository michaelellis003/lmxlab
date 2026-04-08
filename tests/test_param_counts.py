"""Tests that scaled research configs hit their param targets."""

import mlx.core as mx
import mlx.utils
import pytest

from lmxlab.models.base import LanguageModel
from lmxlab.models.falcon import falcon_h1_10m
from lmxlab.models.gpt import gpt_10m, gpt_30m
from lmxlab.models.jamba import jamba_10m
from lmxlab.models.llama import llama_10m, llama_30m

# (factory_fn, target_params, tolerance)
SCALED_CONFIGS = [
    (gpt_10m, 10_000_000, 0.10),
    (gpt_30m, 30_000_000, 0.10),
    (llama_10m, 10_000_000, 0.10),
    (llama_30m, 30_000_000, 0.10),
    (falcon_h1_10m, 10_000_000, 0.10),
    (jamba_10m, 10_000_000, 0.10),
]


@pytest.mark.parametrize(
    "factory,target,tol",
    SCALED_CONFIGS,
    ids=[f.__name__ for f, _, _ in SCALED_CONFIGS],
)
def test_param_count_within_target(factory, target, tol):
    """Each scaled config should be within tolerance of target."""
    config = factory()
    model = LanguageModel(config)
    mx.eval(model.parameters())

    leaves = mlx.utils.tree_flatten(model.parameters())
    actual = sum(p.size for _, p in leaves)

    lo = target * (1 - tol)
    hi = target * (1 + tol)
    assert lo <= actual <= hi, (
        f"{factory.__name__}: {actual:,} params "
        f"not within {tol:.0%} of {target:,} "
        f"(range {lo:,.0f}–{hi:,.0f})"
    )


@pytest.mark.parametrize(
    "factory,target,tol",
    SCALED_CONFIGS,
    ids=[f.__name__ for f, _, _ in SCALED_CONFIGS],
)
def test_forward_pass(factory, target, tol):
    """Each scaled config should produce valid logits."""
    config = factory()
    model = LanguageModel(config)
    mx.eval(model.parameters())

    tokens = mx.zeros((1, 16), dtype=mx.int32)
    logits, _ = model(tokens)
    mx.eval(logits)

    assert logits.shape == (1, 16, config.vocab_size)
