"""Experiment framework: runner, tracking, sweep, analysis, profiling."""

from lmxlab.experiments.analysis import (
    cohens_d,
    compare_experiments,
    compute_statistics,
    confidence_interval,
    simplicity_score,
)
from lmxlab.experiments.flops import (
    estimate_flops_per_step,
    estimate_flops_per_token,
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
    "estimate_flops_per_step",
    "estimate_flops_per_token",
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

try:
    from lmxlab.experiments.mlflow import (
        MLflowCallback,
        MLflowExperimentRunner,
    )

    __all__ += ["MLflowCallback", "MLflowExperimentRunner"]
except ImportError:
    pass
