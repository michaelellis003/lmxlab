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
) -> Iterator[dict[str, float]]:
    """Generate random parameter combinations.

    Samples uniformly from continuous ranges.

    Args:
        param_ranges: Dict mapping parameter names to
            (min, max) tuples.
        n_trials: Number of random combinations.
        seed: Random seed for reproducibility.

    Yields:
        Dicts with one sampled value per parameter.
    """
    import mlx.core as mx

    mx.random.seed(seed)
    keys = list(param_ranges.keys())
    ranges = list(param_ranges.values())

    for _ in range(n_trials):
        config = {}
        for key, (lo, hi) in zip(keys, ranges, strict=True):
            val = mx.random.uniform(low=lo, high=hi)
            mx.eval(val)
            config[key] = float(val.item())
        yield config
