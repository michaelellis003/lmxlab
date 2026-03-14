"""Experiment tracking via results.jsonl (autoresearch pattern)."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LogEntry:
    """A single experiment result entry.

    Args:
        experiment: Experiment name/tag.
        commit: Git commit hash.
        status: Outcome ('keep', 'discard', 'crash').
        val_bpb: Validation bits-per-byte.
        val_loss: Validation loss.
        train_loss: Final training loss.
        param_count: Number of model parameters.
        total_flops: Total FLOPs consumed during training.
        peak_memory_mb: Peak memory usage in MB.
        wall_time_s: Wall clock time in seconds.
        description: Human-readable description.
        config: Full experiment config dict.
        metrics: Additional metrics dict.
        timestamp: Unix timestamp (auto-filled).
        seed: Random seed used.
    """

    experiment: str = ""
    commit: str = ""
    status: str = "keep"
    val_bpb: float = 0.0
    val_loss: float = 0.0
    train_loss: float = 0.0
    param_count: int = 0
    total_flops: float = 0.0
    peak_memory_mb: float = 0.0
    wall_time_s: float = 0.0
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    seed: int = 42


class ExperimentLog:
    """Append-only experiment log backed by results.jsonl.

    This is the ground truth for all experiments. Zero
    dependencies, git-trackable, easy for agents to parse.

    Args:
        path: Path to results.jsonl file.
    """

    def __init__(self, path: str | Path = "results.jsonl") -> None:
        self.path = Path(path)

    def log(self, entry: LogEntry) -> None:
        """Append an entry to the log.

        Args:
            entry: Experiment result to log.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def load(self) -> list[LogEntry]:
        """Load all entries from the log.

        Returns:
            List of LogEntry objects.
        """
        if not self.path.exists():
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entries.append(LogEntry(**data))
        return entries

    def best(
        self,
        metric: str = "val_bpb",
        lower_is_better: bool = True,
    ) -> LogEntry | None:
        """Find the best entry by a metric.

        Args:
            metric: Name of the metric field.
            lower_is_better: If True, minimize; else maximize.

        Returns:
            Best LogEntry, or None if log is empty.
        """
        entries = [e for e in self.load() if e.status == "keep"]
        if not entries:
            return None
        return min(
            entries,
            key=lambda e: getattr(e, metric) * (1 if lower_is_better else -1),
        )

    def summary(self) -> dict[str, Any]:
        """Get summary statistics of all experiments.

        Returns:
            Dict with counts, best metrics, etc.
        """
        entries = self.load()
        if not entries:
            return {"total": 0}
        kept = [e for e in entries if e.status == "keep"]
        return {
            "total": len(entries),
            "kept": len(kept),
            "discarded": sum(1 for e in entries if e.status == "discard"),
            "crashed": sum(1 for e in entries if e.status == "crash"),
            "best_val_bpb": min(
                (e.val_bpb for e in kept), default=float("inf")
            ),
        }
