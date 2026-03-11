"""Hyperparameter sweep utilities."""

import itertools
from collections.abc import Iterator
from typing import Any


def grid_sweep(
    param_grid: dict[str, list[Any]],
) -> Iterator[dict[str, Any]]:
    """Generate all combinations from a parameter grid.

    Args:
        param_grid: Dict mapping parameter names to lists
            of values to try.

    Yields:
        Dicts with one value per parameter.

    Example:
        >>> list(grid_sweep({'lr': [1e-3, 1e-4], 'layers': [2, 4]}))
        [{'lr': 0.001, 'layers': 2}, {'lr': 0.001, 'layers': 4},
         {'lr': 0.0001, 'layers': 2}, {'lr': 0.0001, 'layers': 4}]
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=True))


def random_sweep(
    param_ranges: dict[str, tuple[float, float]],
    n_trials: int = 10,
    seed: int = 42,
    log_scale: set[str] | None = None,
) -> Iterator[dict[str, float]]:
    """Generate random parameter combinations.

    Samples uniformly from continuous ranges by default.
    Parameters listed in ``log_scale`` are sampled in
    log-space, which is standard for learning rates and
    other parameters spanning multiple orders of magnitude.

    Args:
        param_ranges: Dict mapping parameter names to
            (min, max) tuples.
        n_trials: Number of random combinations.
        seed: Random seed for reproducibility.
        log_scale: Set of parameter names to sample in
            log-space. For these, (min, max) must both
            be positive.

    Yields:
        Dicts with one sampled value per parameter.

    Example:
        >>> configs = list(random_sweep(
        ...     param_ranges={"lr": (1e-5, 1e-1), "d_model": (64, 512)},
        ...     n_trials=5,
        ...     log_scale={"lr"},
        ... ))
    """
    import math

    import mlx.core as mx

    log_params = log_scale or set()
    mx.random.seed(seed)
    keys = list(param_ranges.keys())
    ranges = list(param_ranges.values())

    for _ in range(n_trials):
        config = {}
        for key, (lo, hi) in zip(keys, ranges, strict=True):
            if key in log_params:
                log_lo = math.log(lo)
                log_hi = math.log(hi)
                val = mx.random.uniform(low=log_lo, high=log_hi)
                mx.eval(val)
                config[key] = math.exp(float(val.item()))
            else:
                val = mx.random.uniform(low=lo, high=hi)
                mx.eval(val)
                config[key] = float(val.item())
        yield config
