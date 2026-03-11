"""Analysis utilities for experiment results."""

import math
from typing import Any

from lmt_metal.experiments.tracking import ExperimentLog, LogEntry


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
    metric_val = getattr(entry, metric)
    improvement = baseline_metric - metric_val  # positive = better
    param_ratio = baseline_params / max(entry.param_count, 1)
    return improvement * param_ratio
