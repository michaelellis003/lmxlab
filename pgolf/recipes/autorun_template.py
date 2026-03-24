"""Autorun template: autonomous experiment iteration.

Copy this file for each research task. Edit TASK_NAME and the
``propose()`` function. Everything else is immutable infrastructure.

The agent loop (driven by Claude Code via /autorun) calls:
  1. propose(past_results) -> config dict
  2. run(config) -> metrics dict
  3. Log result and print JSON summary

Usage:
    # Single iteration (for agent loop):
    uv run python recipes/autorun_template.py --single

    # Single iteration with tiny FLOP budget (for testing):
    uv run python recipes/autorun_template.py --single \
        --flop-budget 1e10

    # Full local loop (no agent):
    uv run python recipes/autorun_template.py

Requires: ``uv sync --extra hf --extra experiments``
"""

import argparse
import json
from dataclasses import asdict
from typing import Any

import mlx.core as mx

from lmxlab.core.config import ModelConfig
from lmxlab.data.dataset import HFDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.experiments.flops import estimate_flops_per_step
from lmxlab.experiments.mlflow import (
    MLflowCallback,
    MLflowExperimentRunner,
)
from lmxlab.experiments.runner import ExperimentConfig
from lmxlab.experiments.tracking import ExperimentLog
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_10m
from lmxlab.models.llama import llama_10m
from lmxlab.training.callbacks import (
    FLOPCounter,
    ValTracker,
    standard_callbacks,
)
from lmxlab.training.config import TrainConfig
from lmxlab.training.hardware import detect_peak_tflops
from lmxlab.training.trainer import Trainer

# ── Task identity (edit per task) ──────────────────────────────

TASK_NAME = "arch-search-10m"
MAX_ITERATIONS = 20

# ── Architecture registry ─────────────────────────────────────

ARCH_REGISTRY: dict[str, Any] = {
    "llama_10m": llama_10m,
    "gpt_10m": gpt_10m,
}

# ── Defaults ───────────────────────────────────────────────────

DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 256
DEFAULT_EVAL_BATCHES = 20
DEFAULT_FLOP_BUDGET = 5e13


# ── MUTABLE: Claude edits this function between iterations ────


def propose(
    past_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return experiment config dict.

    MUTABLE -- the agent edits this function between iterations.
    Everything else in this file is immutable infrastructure.

    Args:
        past_results: List of dicts from prior kept runs
            (filtered to TASK_NAME). Each has keys: val_loss,
            train_loss, config, description, etc.

    Returns:
        Config dict with keys:
        - arch_factory: str name from ARCH_REGISTRY
        - train_config: dict of TrainConfig overrides
        - description: str describing this iteration
        - batch_size: int (optional, default 8)
        - seq_len: int (optional, default 256)
        - flop_budget: float (optional, default 5e13)
    """
    return {
        "arch_factory": "llama_10m",
        "train_config": {
            "learning_rate": 3e-4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        },
        "description": "baseline llama 10m",
        "batch_size": DEFAULT_BATCH_SIZE,
        "seq_len": DEFAULT_SEQ_LEN,
        "flop_budget": DEFAULT_FLOP_BUDGET,
    }


# ── IMMUTABLE: infrastructure below ───────────────────────────


def _resolve_arch(name: str) -> ModelConfig:
    """Resolve architecture factory by string name.

    Args:
        name: Key in ARCH_REGISTRY.

    Returns:
        ModelConfig from the factory.

    Raises:
        KeyError: If name is not in the registry.
    """
    if name not in ARCH_REGISTRY:
        avail = ", ".join(sorted(ARCH_REGISTRY))
        raise KeyError(f"Unknown arch '{name}'. Available: {avail}")
    return ARCH_REGISTRY[name]()


def run(
    config_dict: dict[str, Any],
    flop_budget_override: float | None = None,
) -> dict[str, Any]:
    """Train and evaluate one iteration.

    IMMUTABLE infrastructure. Do not edit.

    Args:
        config_dict: Config from ``propose()``.
        flop_budget_override: Override FLOP budget (for testing).

    Returns:
        Metrics dict with val_loss, train_loss, steps, etc.
    """
    # Unpack config
    arch_name = config_dict["arch_factory"]
    tc_overrides = config_dict.get("train_config", {})
    description = config_dict.get("description", "")
    batch_size = config_dict.get("batch_size", DEFAULT_BATCH_SIZE)
    seq_len = config_dict.get("seq_len", DEFAULT_SEQ_LEN)
    flop_budget = flop_budget_override or config_dict.get(
        "flop_budget", DEFAULT_FLOP_BUDGET
    )

    # Build model
    mx.random.seed(42)
    model_config = _resolve_arch(arch_name)
    model = LanguageModel(model_config)
    mx.eval(model.parameters())
    n_params = model.count_parameters()

    # Data
    tokenizer = TiktokenTokenizer("gpt2")
    train_ds = HFDataset(
        "roneneldan/TinyStories",
        tokenizer,
        seq_len=seq_len,
        split="train",
    )
    val_ds = HFDataset(
        "roneneldan/TinyStories",
        tokenizer,
        seq_len=seq_len,
        split="validation",
    )
    val_batches = list(
        val_ds.batch_iterator(
            batch_size=batch_size,
            max_batches=DEFAULT_EVAL_BATCHES,
        )
    )

    # FLOP estimation
    flops_per_step = estimate_flops_per_step(model_config, batch_size, seq_len)

    # Callbacks
    cbs = standard_callbacks(
        log_interval=100,
        tokens_per_step=batch_size * seq_len,
        flops_per_step=flops_per_step,
        flop_budget=flop_budget,
        hardware_peak_tflops=detect_peak_tflops(),
        model=model,
        val_batches=val_batches,
        eval_interval=500,
    )
    flop_counter = next(c for c in cbs if isinstance(c, FLOPCounter))
    val_tracker = next(c for c in cbs if isinstance(c, ValTracker))
    mlflow_cb = MLflowCallback(log_interval=100, log_model_params=False)
    cbs.append(mlflow_cb)

    # Train config
    train_config = TrainConfig(
        learning_rate=tc_overrides.get("learning_rate", 3e-4),
        weight_decay=tc_overrides.get("weight_decay", 0.01),
        warmup_steps=tc_overrides.get("warmup_steps", 100),
        max_steps=100_000,
        batch_size=batch_size,
        max_grad_norm=tc_overrides.get("max_grad_norm", 1.0),
        eval_interval=500,
        compile_step=False,
    )
    trainer = Trainer(model, train_config, callbacks=cbs)

    # Experiment tracking
    exp_config = ExperimentConfig(
        name=TASK_NAME,
        description=description,
        time_budget_s=600.0,
        seed=42,
        output_dir="experiments",
    )
    runner = MLflowExperimentRunner(
        exp_config,
        tags={"arch": arch_name, "task": TASK_NAME},
    )
    runner.start()

    # Training loop with FLOP budget
    def data_iter():
        for batch in train_ds.batch_iterator(
            batch_size=batch_size,
        ):
            if flop_counter.should_stop:
                break
            yield batch

    history = trainer.train(data_iter())

    # Collect metrics
    train_loss = history[-1]["loss"] if history else float("inf")
    steps = len(history)
    metrics = {
        "val_loss": val_tracker.best_val_loss,
        "train_loss": train_loss,
        "train_val_gap": train_loss - val_tracker.best_val_loss,
        "steps": steps,
        "total_flops": flop_counter.total_flops,
    }

    # Log to MLflow + results.jsonl
    entry = runner.finish(
        metrics=metrics,
        param_count=n_params,
        config_dict=config_dict,
    )

    return {
        **metrics,
        "param_count": n_params,
        "arch": arch_name,
        "description": description,
        "wall_time_s": entry.wall_time_s,
    }


def load_past_results() -> list[dict[str, Any]]:
    """Load kept results for this task from results.jsonl.

    Returns:
        List of dicts for kept runs matching TASK_NAME.
    """
    log = ExperimentLog("experiments/results.jsonl")
    entries = log.load()
    return [
        asdict(e)
        for e in entries
        if e.experiment == TASK_NAME and e.status == "keep"
    ]


def main() -> None:
    """Run the experiment loop."""
    parser = argparse.ArgumentParser(description="Autorun experiment template")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single iteration (for agent loop)",
    )
    parser.add_argument(
        "--flop-budget",
        type=float,
        default=None,
        help="Override FLOP budget (e.g. 1e10 for testing)",
    )
    args = parser.parse_args()

    iterations = 1 if args.single else MAX_ITERATIONS

    for i in range(iterations):
        past = load_past_results()
        config = propose(past)
        print(f"\n{'=' * 60}")
        print(f"Iteration {i + 1}/{iterations}")
        print(f"Config: {config['description']}")
        print(f"{'=' * 60}")

        metrics = run(config, flop_budget_override=args.flop_budget)

        # Print JSON summary for agent parsing
        summary = {
            "iteration": i + 1,
            "task": TASK_NAME,
            **metrics,
        }
        print("\n=== RESULT ===")
        print(json.dumps(summary, indent=2))

        # Check convergence (simple: val_loss < 0.01 improvement
        # for 3 consecutive runs)
        if len(past) >= 3:
            recent = sorted(past[-3:], key=lambda r: r.get("timestamp", 0))
            losses = [r.get("val_loss", float("inf")) for r in recent]
            if all(
                abs(losses[j] - losses[j + 1]) < 0.01
                for j in range(len(losses) - 1)
            ):
                print("Convergence detected. Stopping.")
                break


if __name__ == "__main__":
    main()
