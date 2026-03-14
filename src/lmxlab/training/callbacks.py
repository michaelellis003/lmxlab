"""Training callbacks."""

import time
from typing import Any, Protocol

from lmxlab.training.config import TrainConfig


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


class ThroughputMonitor:
    """Tracks training throughput (tokens/sec, steps/sec).

    Reports throughput at configured intervals, useful for
    understanding MLX performance characteristics and comparing
    compiled vs uncompiled training.

    Args:
        log_interval: Steps between throughput reports.
        tokens_per_step: Tokens processed per step
            (batch_size * seq_len). If None, only reports
            steps/sec.
    """

    def __init__(
        self,
        log_interval: int = 10,
        tokens_per_step: int | None = None,
    ) -> None:
        self.log_interval = log_interval
        self.tokens_per_step = tokens_per_step
        self._step_times: list[float] = []
        self._last_time: float = 0.0
        self._total_steps: int = 0
        self._train_start: float = 0.0

    def on_train_begin(self, config: TrainConfig) -> None:
        self._last_time = time.monotonic()
        self._train_start = self._last_time
        self._step_times = []
        self._total_steps = 0

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        elapsed = time.monotonic() - self._train_start
        if self._total_steps > 0 and elapsed > 0:
            avg_steps_sec = self._total_steps / elapsed
            msg = (
                f"Throughput summary: {self._total_steps} steps "
                f"in {elapsed:.1f}s ({avg_steps_sec:.1f} steps/s"
            )
            if self.tokens_per_step is not None:
                total_tokens = self._total_steps * self.tokens_per_step
                tok_sec = total_tokens / elapsed
                msg += f", {tok_sec:.0f} tok/s"
            msg += ")"
            print(msg)

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        self._step_times.append(dt)
        self._total_steps += 1

        if step % self.log_interval == 0 and dt > 0:
            # Use recent window for smoother reporting
            window = self._step_times[-self.log_interval :]
            avg_dt = sum(window) / len(window)
            steps_sec = 1.0 / avg_dt if avg_dt > 0 else 0

            msg = f"step {step}: {steps_sec:.1f} steps/s"
            if self.tokens_per_step is not None:
                tok_sec = self.tokens_per_step * steps_sec
                msg += f", {tok_sec:.0f} tok/s"
            print(msg)

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        pass


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


class FLOPCounter:
    """Tracks cumulative FLOPs during training.

    Accumulates FLOPs per step, injects total into metrics,
    and optionally stops training at a FLOP budget.

    Args:
        flops_per_step: FLOPs consumed per training step.
        log_interval: Steps between TFLOP/s reports.
        flop_budget: Stop training after this many FLOPs.
            None means no budget (run indefinitely).
    """

    def __init__(
        self,
        flops_per_step: float,
        log_interval: int = 10,
        flop_budget: float | None = None,
    ) -> None:
        self.flops_per_step = flops_per_step
        self.log_interval = log_interval
        self.flop_budget = flop_budget
        self.total_flops: float = 0.0
        self.should_stop: bool = False
        self._start_time: float = 0.0

    def on_train_begin(self, config: TrainConfig) -> None:
        """Reset counters and record start time."""
        self.total_flops = 0.0
        self.should_stop = False
        self._start_time = time.monotonic()

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Accumulate FLOPs and check budget."""
        self.total_flops += self.flops_per_step
        metrics["total_flops"] = self.total_flops

        if (
            self.flop_budget is not None
            and self.total_flops >= self.flop_budget
        ):
            self.should_stop = True
            pflops = self.total_flops / 1e15
            print(f"FLOP budget reached at step {step} ({pflops:.3f} PFLOPs)")

        if step % self.log_interval == 0:
            elapsed = time.monotonic() - self._start_time
            if elapsed > 0:
                tflops_sec = self.total_flops / elapsed / 1e12
                print(
                    f"step {step}: {tflops_sec:.2f} TFLOP/s "
                    f"({self.total_flops:.2e} total)"
                )

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """No action on eval."""

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """Print FLOP summary."""
        elapsed = time.monotonic() - self._start_time
        pflops = self.total_flops / 1e15
        if elapsed > 0:
            tflops_sec = self.total_flops / elapsed / 1e12
            print(
                f"FLOPs summary: {pflops:.4f} PFLOPs "
                f"in {elapsed:.1f}s ({tflops_sec:.2f} TFLOP/s)"
            )
        else:
            print(f"FLOPs summary: {pflops:.4f} PFLOPs")
