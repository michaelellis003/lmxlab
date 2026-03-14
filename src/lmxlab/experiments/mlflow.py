"""MLflow integration for experiment tracking.

Provides MLflowCallback for per-step metric logging and
MLflowExperimentRunner for experiment-level parameter/summary
logging. Requires mlflow (optional dependency).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from lmxlab.experiments.runner import (
    ExperimentConfig,
    ExperimentRunner,
    _get_git_commit,
)
from lmxlab.experiments.tracking import ExperimentLog
from lmxlab.training.config import TrainConfig

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

_MISSING_MSG = (
    "mlflow is required for MLflow integration. "
    "Install with: uv sync --extra experiments"
)


def _require_mlflow() -> None:
    """Raise ImportError if mlflow is not installed."""
    if mlflow is None:
        raise ImportError(_MISSING_MSG)


class MLflowCallback:
    """Logs training metrics to MLflow.

    Implements the Callback protocol. Assumes an MLflow run is
    already active (started by MLflowExperimentRunner or manually).

    Args:
        log_interval: Steps between metric logs (default 10).
        log_model_params: Log TrainConfig fields as MLflow
            params on train begin.
    """

    def __init__(
        self,
        log_interval: int = 10,
        log_model_params: bool = True,
    ) -> None:
        _require_mlflow()
        self.log_interval = log_interval
        self.log_model_params = log_model_params

    def on_train_begin(self, config: TrainConfig) -> None:
        """Log TrainConfig fields as MLflow params."""
        if self.log_model_params:
            params = asdict(config)
            mlflow.log_params(params)

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Log all numeric metrics at configured interval."""
        if step % self.log_interval == 0:
            logged = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            if logged:
                mlflow.log_metrics(logged, step=step)

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Log all numeric eval metrics."""
        logged = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if logged:
            mlflow.log_metrics(logged, step=step)

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """Log final summary metrics."""
        if not history:
            return
        final = history[-1]
        summary: dict[str, float] = {}
        if "loss" in final:
            summary["final_loss"] = float(final["loss"])
        summary["total_steps"] = float(len(history))
        if summary:
            mlflow.log_metrics(summary)


class MLflowExperimentRunner:
    """Wraps ExperimentRunner with MLflow run lifecycle.

    Creates an MLflow experiment, starts/ends runs, and logs
    ExperimentConfig fields as MLflow params alongside the
    standard results.jsonl tracking.

    Args:
        config: Experiment configuration.
        experiment_name: MLflow experiment name
            (defaults to config.name).
        tags: Extra MLflow tags.
        log: Optional ExperimentLog instance.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
        log: ExperimentLog | None = None,
    ) -> None:
        _require_mlflow()
        self.config = config
        self.experiment_name = experiment_name or config.name
        self.tags = tags or {}
        self.runner = ExperimentRunner(config, log=log)
        self._git_commit = _get_git_commit()

    def start(self) -> None:
        """Start the experiment runner and an MLflow run."""
        self.runner.start()
        # Filesystem tracking is deprecated (Feb 2026). Use SQLite.
        uri = mlflow.get_tracking_uri()
        if not uri.startswith("sqlite"):
            db_path = Path(self.config.output_dir).resolve() / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(
            run_name=self.config.description or self.config.name,
        )
        # Log experiment-level params
        mlflow.log_params(
            {
                "experiment_name": self.config.name,
                "seed": self.config.seed,
                "time_budget_s": self.config.time_budget_s,
                "git_commit": self._git_commit,
            }
        )
        # Log extra tags
        if self.tags:
            mlflow.set_tags(self.tags)

    def finish(
        self,
        metrics: dict[str, Any],
        param_count: int = 0,
        config_dict: dict[str, Any] | None = None,
        status: str = "keep",
    ) -> Any:
        """Finish the run: log to both results.jsonl and MLflow.

        Args:
            metrics: Result metrics dict.
            param_count: Number of model parameters.
            config_dict: Full config for logging.
            status: Outcome status.

        Returns:
            The LogEntry from ExperimentRunner.finish().
        """
        # Log model config and param count to MLflow
        if config_dict:
            # Prefix to avoid collision with TrainConfig params
            prefixed = {f"model/{k}": v for k, v in config_dict.items()}
            mlflow.log_params(prefixed)
        if param_count:
            mlflow.log_param("model/param_count", param_count)

        # Log final metrics
        mlflow_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow_metrics[k] = float(v)
        if mlflow_metrics:
            mlflow.log_metrics(mlflow_metrics)

        entry = self.runner.finish(
            metrics=metrics,
            param_count=param_count,
            config_dict=config_dict,
            status=status,
        )
        mlflow.end_run()
        return entry

    def time_remaining(self) -> float:
        """Seconds remaining in the time budget."""
        return self.runner.time_remaining()

    def is_time_up(self) -> bool:
        """Check if the time budget has been exceeded."""
        return self.runner.is_time_up()
