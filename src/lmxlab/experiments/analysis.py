"""Analysis utilities for experiment results."""

import math
from typing import Any

from lmxlab.experiments.tracking import ExperimentLog, LogEntry


def compare_experiments(
    log: ExperimentLog,
    metric: str = "val_bpb",
) -> list[dict[str, Any]]:
    """Compare all kept experiments by a metric.

    Returns experiments sorted by the metric (ascending).

    Args:
        log: Experiment log to analyze.
        metric: Metric name to compare.

    Returns:
        List of dicts with experiment name, metric value,
        param_count, and wall_time.
    """
    entries = [e for e in log.load() if e.status == "keep"]
    entries.sort(key=lambda e: getattr(e, metric, float("inf")))
    return [
        {
            "experiment": e.experiment,
            metric: getattr(e, metric),
            "param_count": e.param_count,
            "wall_time_s": e.wall_time_s,
            "description": e.description,
        }
        for e in entries
    ]


def compute_statistics(
    values: list[float],
) -> dict[str, float]:
    """Compute basic statistics for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Dict with mean, std, min, max, n.
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
        }

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def cohens_d(
    group_a: list[float],
    group_b: list[float],
) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation (equal-variance assumption).
    Useful for reporting effect sizes alongside p-values, as
    recommended in the pre-registered experiment plans.

    Args:
        group_a: Values from the first group.
        group_b: Values from the second group.

    Returns:
        Cohen's d. Positive means group_a > group_b.
        Conventions: |d| < 0.2 small, 0.5 medium, 0.8 large.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = sum(group_a) / n_a
    mean_b = sum(group_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = math.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a confidence interval for the mean.

    Uses the t-distribution for small samples. Falls back to
    z-approximation for n >= 30.

    Args:
        values: Sample values.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0.0
        return (mean, mean)

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_err = math.sqrt(variance / n)

    # t critical values for common confidence levels and small n
    # For n >= 30, use z-approximation
    if n >= 30:
        z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
        z = z_map.get(confidence, 1.960)
    else:
        # Approximate t critical value using Abramowitz & Stegun
        # For educational use; production code should use scipy
        df = n - 1
        z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
        z_approx = z_map.get(confidence, 1.960)
        # Crude t correction: t ≈ z + (z + z^3) / (4 * df)
        z = z_approx + (z_approx + z_approx**3) / (4 * df)

    margin = z * std_err
    return (mean - margin, mean + margin)


def simplicity_score(
    entry: LogEntry,
    baseline_params: int,
    baseline_metric: float,
    metric: str = "val_bpb",
) -> float:
    """Score an experiment by the simplicity bias principle.

    Rewards improvements that use fewer parameters.
    Score = metric_improvement * (baseline_params / param_count)

    Higher is better. Positive means improvement over baseline.

    Args:
        entry: Experiment entry to score.
        baseline_params: Baseline parameter count.
        baseline_metric: Baseline metric value.
        metric: Metric name (lower is better).

    Returns:
        Simplicity-weighted improvement score.
    """
    metric_val: float = getattr(entry, metric)
    improvement = baseline_metric - metric_val  # positive = better
    param_ratio = baseline_params / max(entry.param_count, 1)
    return improvement * param_ratio
