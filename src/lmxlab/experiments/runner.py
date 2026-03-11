"""Experiment runner with autoresearch patterns."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from lmxlab.experiments.tracking import ExperimentLog, LogEntry


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run.

    Args:
        name: Experiment name/tag.
        description: Human-readable description.
        time_budget_s: Maximum wall-clock time in seconds.
        seed: Random seed.
        output_dir: Directory for outputs.
    """

    name: str = "experiment"
    description: str = ""
    time_budget_s: float = 300.0  # 5 minutes default
    seed: int = 42
    output_dir: str = "experiments"


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class ExperimentRunner:
    """Run experiments with autoresearch patterns.

    Enforces fixed time budgets, logs results to results.jsonl,
    and tracks git commits for reproducibility.

    Args:
        config: Experiment configuration.
        log: Experiment log (defaults to results.jsonl in output_dir).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        log: ExperimentLog | None = None,
    ) -> None:
        self.config = config
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.log = log or ExperimentLog(output / "results.jsonl")
        self._start_time: float = 0.0

    def time_remaining(self) -> float:
        """Seconds remaining in the time budget."""
        if self._start_time == 0:
            return self.config.time_budget_s
        elapsed = time.monotonic() - self._start_time
        return max(0.0, self.config.time_budget_s - elapsed)

    def is_time_up(self) -> bool:
        """Check if the time budget has been exceeded."""
        return self.time_remaining() <= 0

    def start(self) -> None:
        """Start the experiment timer and set the random seed."""
        self._start_time = time.monotonic()
        mx.random.seed(self.config.seed)

    def finish(
        self,
        metrics: dict[str, Any],
        param_count: int = 0,
        config_dict: dict[str, Any] | None = None,
        status: str = "keep",
    ) -> LogEntry:
        """Finish the experiment and log results.

        Args:
            metrics: Dict of result metrics (must include
                'val_loss' or 'val_bpb').
            param_count: Number of model parameters.
            config_dict: Full experiment config for logging.
            status: 'keep', 'discard', or 'crash'.

        Returns:
            The logged entry.
        """
        wall_time = time.monotonic() - self._start_time

        entry = LogEntry(
            experiment=self.config.name,
            commit=_get_git_commit(),
            status=status,
            val_bpb=metrics.get("val_bpb", 0.0),
            val_loss=metrics.get("val_loss", 0.0),
            train_loss=metrics.get("train_loss", 0.0),
            param_count=param_count,
            wall_time_s=wall_time,
            description=self.config.description,
            config=config_dict or {},
            metrics=metrics,
            seed=self.config.seed,
        )
        self.log.log(entry)
        return entry
