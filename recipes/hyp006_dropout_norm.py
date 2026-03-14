"""HYP-006: Dropout x normalization interaction at 30M params.

Pre-registered hypothesis:
    Does the dropout x normalization interaction (ANOM-009/010/011)
    replicate at 30M params with BPE tokenization?

Competing hypotheses:
    H1 (Replicates): LLaMA (RMSNorm) and GPT (LayerNorm) have
        different optimal dropout rates; non-monotonic pattern holds
    H2 (Partially): Interaction exists but optimal rates shift
    H3 (Null): No interaction at 30M — was a small-scale artifact

Design:
    2 architectures (GPT-30M LayerNorm, LLaMA-30M RMSNorm)
    x 4 dropout rates (0.0, 0.1, 0.2, 0.3)
    x 3 seeds (42, 43, 44)
    = 24 runs, FLOP-matched via FLOPCounter
    Dataset: TinyStories BPE (train/validation splits)
    Primary metric: val_loss

Requires: ``uv sync --extra hf --extra tokenizers``

Usage:
    uv run python recipes/hyp006_dropout_norm.py
    uv run python recipes/hyp006_dropout_norm.py --dry-run
    uv run python recipes/hyp006_dropout_norm.py --max-runs 4
    uv run python recipes/hyp006_dropout_norm.py --target-steps 500
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from lmxlab.data.dataset import HFDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.experiments.flops import estimate_flops_per_step
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_30m
from lmxlab.models.llama import llama_30m
from lmxlab.training.callbacks import FLOPCounter, MetricsLogger
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

# ── Grid definition ──────────────────────────────────────

ARCHS = {
    "gpt_30m": gpt_30m,
    "llama_30m": llama_30m,
}
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3]
SEEDS = [42, 43, 44]
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
SEQ_LEN = 256
EVAL_INTERVAL = 500
EVAL_BATCHES = 20


# ── Periodic eval callback ───────────────────────────────


class _PeriodicEval:
    """Evaluate with dropout disabled at fixed intervals."""

    def __init__(
        self,
        model: LanguageModel,
        val_batches: list[tuple[mx.array, mx.array]],
        interval: int,
    ) -> None:
        self.model = model
        self.val_batches = val_batches
        self.interval = interval
        self.best_val: float = float("inf")

    def on_train_begin(self, config: TrainConfig) -> None:
        """No-op."""

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Run eval every ``interval`` steps."""
        if step > 0 and step % self.interval == 0:
            val_loss = self._evaluate()
            metrics["val_loss"] = val_loss
            self.best_val = min(self.best_val, val_loss)
            print(
                f"  eval step {step}: "
                f"val={val_loss:.4f} "
                f"best={self.best_val:.4f}"
            )

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """No-op."""

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """No-op."""

    def _evaluate(self) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for x, y in self.val_batches:
            logits, _ = self.model(x)
            logits = logits.reshape(-1, logits.shape[-1])
            loss = nn.losses.cross_entropy(
                logits, y.reshape(-1), reduction="mean"
            )
            mx.eval(loss)
            total += loss.item()
            n += 1
        self.model.train()
        return total / max(n, 1)


# ── Core functions ───────────────────────────────────────


def build_grid() -> list[tuple[str, float, int]]:
    """Generate all (arch, dropout, seed) combinations."""
    return [
        (arch, dropout, seed)
        for arch in ARCHS
        for dropout in DROPOUT_RATES
        for seed in SEEDS
    ]


def make_config(arch_name: str, dropout: float):
    """Create model config with specified dropout rate."""
    factory = ARCHS[arch_name]
    config = factory()
    block = replace(config.block, dropout=dropout)
    return replace(config, block=block)


def evaluate(
    model: LanguageModel,
    val_batches: list[tuple[mx.array, mx.array]],
) -> float:
    """Compute val loss with dropout disabled."""
    model.eval()
    total = 0.0
    n = 0
    for x, y in val_batches:
        logits, _ = model(x)
        logits = logits.reshape(-1, logits.shape[-1])
        loss = nn.losses.cross_entropy(logits, y.reshape(-1), reduction="mean")
        mx.eval(loss)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def compute_flop_budget(target_steps: int = 2000) -> int:
    """Compute shared FLOP budget from GPT-30M reference.

    Uses GPT-30M as the reference since it has slightly more
    FLOPs/step than LLaMA-30M (standard FFN × 22 layers vs
    gated FFN × 18 layers). Both get ~target_steps at this
    budget.
    """
    config = gpt_30m()
    flops_per_step = estimate_flops_per_step(config, BATCH_SIZE, SEQ_LEN)
    return int(flops_per_step * target_steps)


def run_single(
    arch_name: str,
    dropout: float,
    seed: int,
    flop_budget: int,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run one training experiment."""
    run_name = f"{arch_name}_d{dropout:.1f}_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  arch={arch_name}, dropout={dropout}, seed={seed}")

    if dry_run:
        print("  [DRY RUN] Skipping.")
        return None

    mx.random.seed(seed)

    # Model
    model_config = make_config(arch_name, dropout)
    model = LanguageModel(model_config)
    mx.eval(model.parameters())
    n_params = model.count_parameters()

    # FLOPs per step for this architecture
    flops_per_step = estimate_flops_per_step(model_config, BATCH_SIZE, SEQ_LEN)
    est_steps = flop_budget / flops_per_step
    print(f"  params={n_params:,}")
    print(f"  flops/step={flops_per_step:.2e}")
    print(f"  flop_budget={flop_budget:.2e}")
    print(f"  est_steps={est_steps:.0f}")

    # Data
    tokenizer = TiktokenTokenizer("gpt2")
    train_ds = HFDataset(
        "roneneldan/TinyStories",
        tokenizer,
        seq_len=SEQ_LEN,
        split="train",
    )
    val_ds = HFDataset(
        "roneneldan/TinyStories",
        tokenizer,
        seq_len=SEQ_LEN,
        split="validation",
    )
    val_batches = list(
        val_ds.batch_iterator(
            batch_size=BATCH_SIZE,
            max_batches=EVAL_BATCHES,
        )
    )

    # Callbacks
    flop_counter = FLOPCounter(
        flops_per_step=flops_per_step,
        log_interval=EVAL_INTERVAL,
        flop_budget=flop_budget,
    )
    periodic_eval = _PeriodicEval(model, val_batches, EVAL_INTERVAL)
    logger = MetricsLogger(log_interval=100)

    # Trainer (max_steps high — FLOP budget is the limiter)
    train_config = TrainConfig(
        learning_rate=LEARNING_RATE,
        max_steps=100_000,
        batch_size=BATCH_SIZE,
        warmup_steps=100,
        eval_interval=EVAL_INTERVAL,
        compile_step=True,
    )
    trainer = Trainer(
        model,
        train_config,
        callbacks=[logger, flop_counter, periodic_eval],
    )

    # Experiment tracking
    exp_config = ExperimentConfig(
        name="HYP-006",
        description=run_name,
        time_budget_s=600.0,
        seed=seed,
        output_dir="experiments",
    )
    runner = ExperimentRunner(exp_config)
    runner.start()

    # Initial eval
    init_val = evaluate(model, val_batches)
    print(f"  init_val_loss={init_val:.4f}")

    # Train (generator stops when FLOP budget reached)
    start = time.monotonic()

    def data_iter():
        for batch in train_ds.batch_iterator(
            batch_size=BATCH_SIZE,
        ):
            if flop_counter.should_stop:
                break
            yield batch

    history = trainer.train(data_iter())
    elapsed = time.monotonic() - start

    # Final eval (clean, no dropout)
    final_val = evaluate(model, val_batches)
    train_loss = history[-1]["loss"] if history else float("inf")
    steps = len(history)

    # Log results
    entry = runner.finish(
        metrics={
            "val_loss": final_val,
            "best_val_loss": periodic_eval.best_val,
            "train_loss": train_loss,
            "train_val_gap": train_loss - final_val,
            "init_val_loss": init_val,
            "steps": steps,
            "total_flops": flop_counter.total_flops,
        },
        param_count=n_params,
        config_dict={
            "arch": arch_name,
            "dropout": dropout,
            "d_model": model_config.block.d_model,
            "n_layers": model_config.n_layers,
            "norm": model_config.block.norm,
            "lr": LEARNING_RATE,
            "flop_budget": flop_budget,
        },
    )

    print(f"\n  Steps:      {steps}")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val loss:   {final_val:.4f}")
    print(f"  Best val:   {periodic_eval.best_val:.4f}")
    print(f"  Gap:        {train_loss - final_val:+.4f}")
    print(f"  Wall time:  {elapsed:.1f}s")
    print(f"  FLOPs:      {flop_counter.total_flops:.2e}")
    print(f"  Status:     {entry.status}")

    return {
        "run": run_name,
        "arch": arch_name,
        "norm": model_config.block.norm,
        "dropout": dropout,
        "seed": seed,
        "val_loss": final_val,
        "best_val_loss": periodic_eval.best_val,
        "train_loss": train_loss,
        "gap": train_loss - final_val,
        "steps": steps,
        "wall_time": elapsed,
        "total_flops": flop_counter.total_flops,
        "params": n_params,
    }


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-006 grid sweep."""
    parser = argparse.ArgumentParser(
        description="HYP-006: Dropout x normalization at 30M"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid without running",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit number of runs (for testing)",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=2000,
        help="Target steps for FLOP budget (default: 2000)",
    )
    args = parser.parse_args()

    grid = build_grid()
    flop_budget = compute_flop_budget(args.target_steps)

    print("HYP-006: Dropout x Normalization at 30M")
    print(f"Grid: {len(grid)} runs")
    print(f"  Archs: {list(ARCHS.keys())}")
    print(f"  Dropout rates: {DROPOUT_RATES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  FLOP budget: {flop_budget:.2e} per run")
    print(f"  Target steps: ~{args.target_steps}")
    print(f"  LR: {LEARNING_RATE}")

    if args.max_runs:
        grid = grid[: args.max_runs]
        print(f"  Limited to {len(grid)} runs")

    results: list[dict[str, Any]] = []
    for i, (arch, dropout, seed) in enumerate(grid):
        print(f"\n[{i + 1}/{len(grid)}]", end="")
        result = run_single(
            arch,
            dropout,
            seed,
            flop_budget,
            dry_run=args.dry_run,
        )
        if result:
            results.append(result)

    if not results:
        return

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    header = f"{'Run':<30} {'Val':>8} {'BestV':>8} {'Train':>8} {'Gap':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['run']:<30} "
            f"{r['val_loss']:>8.4f} "
            f"{r['best_val_loss']:>8.4f} "
            f"{r['train_loss']:>8.4f} "
            f"{r['gap']:>+8.4f}"
        )

    # Save results
    out = Path("experiments") / "hyp006_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
