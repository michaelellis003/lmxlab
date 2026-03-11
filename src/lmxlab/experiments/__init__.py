"""Experiment framework: runner, tracking, sweep, analysis, profiling."""

from lmxlab.experiments.analysis import (
    cohens_d,
    compare_experiments,
    compute_statistics,
    confidence_interval,
    simplicity_score,
)
from lmxlab.experiments.profiling import (
    benchmark_fn,
    count_parameters_by_module,
    memory_estimate,
    profile_forward,
    profile_generation,
)
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner
from lmxlab.experiments.sweep import grid_sweep, random_sweep
from lmxlab.experiments.tracking import ExperimentLog, LogEntry

__all__ = [
    "ExperimentConfig",
    "ExperimentLog",
    "ExperimentRunner",
    "LogEntry",
    "grid_sweep",
    "random_sweep",
    "benchmark_fn",
    "cohens_d",
    "compare_experiments",
    "compute_statistics",
    "confidence_interval",
    "count_parameters_by_module",
    "memory_estimate",
    "profile_forward",
    "profile_generation",
    "simplicity_score",
]
