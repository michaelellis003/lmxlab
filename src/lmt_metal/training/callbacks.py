"""Training callbacks."""

import time
from typing import Any, Protocol

from lmt_metal.training.config import TrainConfig


class Callback(Protocol):
    """Protocol for training callbacks."""

    def on_train_begin(self, config: TrainConfig) -> None: ...
    def on_train_end(self, history: list[dict[str, Any]]) -> None: ...
    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None: ...
    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None: ...


class MetricsLogger:
    """Logs training metrics at configured intervals.

    Args:
        log_interval: Steps between log outputs.
    """

    def __init__(self, log_interval: int = 10) -> None:
        self.log_interval = log_interval
        self._start_time: float = 0.0

    def on_train_begin(self, config: TrainConfig) -> None:
        self._start_time = time.monotonic()

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        elapsed = time.monotonic() - self._start_time
        print(f"Training complete: {len(history)} steps in {elapsed:.1f}s")

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        if step % self.log_interval == 0:
            loss = metrics.get("loss", 0.0)
            lr = metrics.get("learning_rate", 0.0)
            print(f"step {step}: loss={loss:.4f}, lr={lr:.2e}")

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        eval_loss = metrics.get("eval_loss", 0.0)
        print(f"step {step} eval: loss={eval_loss:.4f}")


class EarlyStopping:
    """Stop training when eval loss stops improving.

    Args:
        patience: Steps without improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._wait: int = 0
        self.should_stop: bool = False

    def on_train_begin(self, config: TrainConfig) -> None:
        self._best_loss = float("inf")
        self._wait = 0
        self.should_stop = False

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        pass

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        pass

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        eval_loss = metrics.get("eval_loss", float("inf"))
        if eval_loss < self._best_loss - self.min_delta:
            self._best_loss = eval_loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.should_stop = True
                print(
                    f"Early stopping at step {step} "
                    f"(best loss: {self._best_loss:.4f})"
                )
