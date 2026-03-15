"""HYP-007: Test-time compute scaling on modular arithmetic.

Pre-registered hypothesis:
    Can best-of-N sampling with execution verification compensate
    for model size at 10M parameters on (a+b) mod 97?

Competing hypotheses:
    H7-a (Effective): pass@64 significantly exceeds pass@1,
        demonstrating test-time compute scaling
    H7-b (Ceiling): pass@k plateaus early (k<8), model lacks
        the knowledge to generate correct answers
    H7-c (Dropout helps): dropout>0 improves pass@k via
        implicit ensembling at inference time
    H7-d (Null): no meaningful pass@k improvement beyond
        random chance

Design:
    3 dropout rates (0.0, 0.1, 0.2)
    x 3 seeds (42, 43, 44)
    = 9 runs, FLOP-matched via FLOPCounter
    Dataset: modular arithmetic (a+b) mod 97
    Primary metric: pass@k curves (k=1,2,4,8,16,32,64)

Usage:
    uv run python recipes/hyp007_test_time_compute.py
    uv run python recipes/hyp007_test_time_compute.py --dry-run
    uv run python recipes/hyp007_test_time_compute.py --pilot
    uv run python recipes/hyp007_test_time_compute.py --max-runs 1
    uv run python recipes/hyp007_test_time_compute.py --target-steps 500
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.modular_arithmetic import ModularArithmeticDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.eval.metrics import pass_at_k
from lmxlab.experiments.flops import estimate_flops_per_step
from lmxlab.experiments.mlflow import (
    MLflowCallback,
    MLflowExperimentRunner,
)
from lmxlab.experiments.runner import ExperimentConfig
from lmxlab.models.base import LanguageModel
from lmxlab.models.llama import llama_10m
from lmxlab.training.callbacks import (
    FLOPCounter,
    ValTracker,
    standard_callbacks,
)
from lmxlab.training.config import TrainConfig
from lmxlab.training.hardware import detect_peak_tflops
from lmxlab.training.metric_callbacks import (
    ActivationStatsCallback,
    AttentionEntropyCallback,
    WeightStatsCallback,
)
from lmxlab.training.trainer import Trainer

# ── Grid definition ──────────────────────────────────────

DROPOUT_RATES = [0.0, 0.1, 0.2]
SEEDS = [42, 43, 44]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
N_SAMPLES = 64
MODULUS = 97
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
SEQ_LEN = 256
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
TEMPERATURE = 0.8


# ── Core functions ───────────────────────────────────────


def build_grid() -> list[tuple[float, int]]:
    """Generate all (dropout, seed) combinations."""
    return [(dropout, seed) for dropout in DROPOUT_RATES for seed in SEEDS]


def make_config(dropout: float):
    """Create LLaMA-10M config with specified dropout.

    Args:
        dropout: Dropout rate to apply.

    Returns:
        ModelConfig with dropout set.
    """
    config = llama_10m()
    block = replace(config.block, dropout=dropout)
    return replace(config, block=block)


def compute_flop_budget(target_steps: int = 2000) -> int:
    """Compute FLOP budget from LLaMA-10M reference.

    Args:
        target_steps: Target number of training steps.

    Returns:
        Total FLOP budget.
    """
    config = llama_10m()
    flops_per_step = estimate_flops_per_step(config, BATCH_SIZE, SEQ_LEN)
    return int(flops_per_step * target_steps)


def evaluate_pass_at_k_modular(
    model: LanguageModel,
    dataset: ModularArithmeticDataset,
    tokenizer: TiktokenTokenizer,
    k_values: list[int],
    n_samples: int,
    temperature: float,
) -> dict[str, float]:
    """Evaluate pass@k on modular arithmetic.

    Uses single forward pass (fast path): the answer is always
    a single token, so we forward the prompt once and sample
    from the next-token logits.

    Args:
        model: Trained language model.
        dataset: Test split of ModularArithmeticDataset.
        tokenizer: GPT-2 BPE tokenizer.
        k_values: Values of k for pass@k.
        n_samples: Number of samples per prompt (N).
        temperature: Sampling temperature.

    Returns:
        Dict mapping ``'pass@k'`` to average score.
    """
    model.eval()
    prompts = dataset.get_prompts()
    answer_ids = dataset.answer_token_ids

    results: dict[str, list[float]] = {f"pass_at_{k}": [] for k in k_values}

    for prompt_tokens, answer in prompts:
        # Batch the prompt: (n_samples, prompt_len)
        prompt_batch = mx.broadcast_to(
            prompt_tokens[None, :],
            (n_samples, prompt_tokens.shape[0]),
        )

        # Single forward pass — no KV cache needed
        logits, _ = model(prompt_batch)
        mx.eval(logits)

        # Sample from next-token logits
        next_logits = logits[:, -1, :] / temperature
        sampled = mx.random.categorical(next_logits)
        mx.eval(sampled)

        # Check correctness
        correct_id = answer_ids[answer]
        sampled_list = sampled.tolist()
        c = sum(1 for s in sampled_list if s == correct_id)

        for k in k_values:
            if k <= n_samples:
                score = pass_at_k(n_samples, c, k)
                results[f"pass_at_{k}"].append(score)

    model.train()

    return {
        key: sum(vals) / len(vals) if vals else 0.0
        for key, vals in results.items()
    }


def run_single(
    dropout: float,
    seed: int,
    flop_budget: int,
    dry_run: bool = False,
    target_steps: int = 2000,
) -> dict[str, Any] | None:
    """Train one model and evaluate pass@k.

    Args:
        dropout: Dropout rate.
        seed: Random seed.
        flop_budget: FLOP budget for training.
        dry_run: If True, skip actual training.
        target_steps: Target steps (for display).

    Returns:
        Result dict or None if dry_run.
    """
    run_name = f"llama10m_d{dropout:.1f}_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  dropout={dropout}, seed={seed}")

    if dry_run:
        print("  [DRY RUN] Skipping.")
        return None

    mx.random.seed(seed)

    # Model
    model_config = make_config(dropout)
    model = LanguageModel(model_config)
    mx.eval(model.parameters())
    n_params = model.count_parameters()

    # FLOPs per step
    flops_per_step = estimate_flops_per_step(model_config, BATCH_SIZE, SEQ_LEN)
    est_steps = flop_budget / flops_per_step
    print(f"  params={n_params:,}")
    print(f"  flops/step={flops_per_step:.2e}")
    print(f"  flop_budget={flop_budget:.2e}")
    print(f"  est_steps={est_steps:.0f}")

    # Data — same train split for all runs
    tokenizer = TiktokenTokenizer("gpt2")
    train_ds = ModularArithmeticDataset(p=MODULUS, split="train", seed=42)
    test_ds = ModularArithmeticDataset(p=MODULUS, split="test", seed=42)
    train_tokens = train_ds.get_tokens()

    # Hold out 10% of train tokens for val loss monitoring
    val_split = int(len(train_tokens) * 0.9)
    train_stream = train_tokens[:val_split]
    val_stream = train_tokens[val_split:]
    val_batches = list(
        batch_iterator(
            val_stream,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            shuffle=False,
        )
    )

    # Probe batch for metric callbacks
    probe_batch = mx.broadcast_to(
        train_tokens[:SEQ_LEN][None, :],
        (1, SEQ_LEN),
    )

    # Callbacks (standard stack + metric callbacks + MLflow)
    cbs = standard_callbacks(
        log_interval=100,
        tokens_per_step=BATCH_SIZE * SEQ_LEN,
        flops_per_step=flops_per_step,
        flop_budget=flop_budget,
        hardware_peak_tflops=detect_peak_tflops(),
        model=model,
        val_batches=val_batches,
        eval_interval=EVAL_INTERVAL,
    )
    flop_counter = next(c for c in cbs if isinstance(c, FLOPCounter))
    val_tracker = next(c for c in cbs if isinstance(c, ValTracker))

    # Metric callbacks for diagnostics
    metric_cbs = [
        AttentionEntropyCallback(model, probe_batch, eval_interval=500),
        ActivationStatsCallback(model, probe_batch, eval_interval=500),
        WeightStatsCallback(model, log_interval=100),
    ]
    cbs.extend(metric_cbs)

    mlflow_cb = MLflowCallback(
        log_interval=100,
        log_model_params=False,
    )
    cbs.append(mlflow_cb)

    # Trainer
    train_config = TrainConfig(
        learning_rate=LEARNING_RATE,
        max_steps=100_000,
        batch_size=BATCH_SIZE,
        warmup_steps=100,
        eval_interval=EVAL_INTERVAL,
        compile_step=False,
    )
    trainer = Trainer(model, train_config, callbacks=cbs)

    # Experiment tracking
    exp_config = ExperimentConfig(
        name="HYP-007",
        description=run_name,
        time_budget_s=600.0,
        seed=seed,
        output_dir="experiments",
    )
    runner = MLflowExperimentRunner(
        exp_config,
        tags={
            "dropout": str(dropout),
            "hypothesis": "HYP-007",
        },
    )
    runner.start()

    # ── Phase 1: Train ──
    start = time.monotonic()

    def data_iter():
        while not flop_counter.should_stop:
            for batch in batch_iterator(
                train_stream,
                batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
            ):
                if flop_counter.should_stop:
                    break
                yield batch

    history = trainer.train(data_iter())

    train_loss = history[-1]["loss"] if history else float("inf")
    steps = len(history)

    # ── Phase 2: Evaluate pass@k ──
    print("  Evaluating pass@k...")
    eval_start = time.monotonic()
    pass_at_k_results = evaluate_pass_at_k_modular(
        model=model,
        dataset=test_ds,
        tokenizer=tokenizer,
        k_values=K_VALUES,
        n_samples=N_SAMPLES,
        temperature=TEMPERATURE,
    )
    eval_elapsed = time.monotonic() - eval_start
    print(f"  pass@k eval: {eval_elapsed:.1f}s")
    for k_name, score in pass_at_k_results.items():
        print(f"    {k_name}: {score:.4f}")

    # ── Phase 3: Log results ──
    metrics = {
        "val_loss": val_tracker.best_val_loss,
        "best_val_loss": val_tracker.best_val_loss,
        "train_loss": train_loss,
        "train_val_gap": train_loss - val_tracker.best_val_loss,
        "init_val_loss": val_tracker.init_val_loss,
        "steps": steps,
        "total_flops": flop_counter.total_flops,
        **pass_at_k_results,
    }

    entry = runner.finish(
        metrics=metrics,
        param_count=n_params,
        config_dict={
            "dropout": dropout,
            "d_model": model_config.block.d_model,
            "n_layers": model_config.n_layers,
            "lr": LEARNING_RATE,
            "flop_budget": flop_budget,
            "modulus": MODULUS,
            "n_samples": N_SAMPLES,
            "temperature": TEMPERATURE,
        },
    )

    total_elapsed = time.monotonic() - start
    print(f"\n  Steps:      {steps}")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Best val:   {val_tracker.best_val_loss:.4f}")
    print(f"  Init val:   {val_tracker.init_val_loss:.4f}")
    gap = train_loss - val_tracker.best_val_loss
    print(f"  Gap:        {gap:+.4f}")
    print(f"  Wall time:  {total_elapsed:.1f}s")
    print(f"  FLOPs:      {flop_counter.total_flops:.2e}")
    print(f"  Status:     {entry.status}")

    return {
        "run": run_name,
        "dropout": dropout,
        "seed": seed,
        "val_loss": val_tracker.best_val_loss,
        "best_val_loss": val_tracker.best_val_loss,
        "train_loss": train_loss,
        "gap": gap,
        "steps": steps,
        "wall_time": total_elapsed,
        "total_flops": flop_counter.total_flops,
        "params": n_params,
        **pass_at_k_results,
    }


def run_pilot() -> None:
    """Quick single run to calibrate FLOP budget."""
    print("=" * 60)
    print("PILOT RUN: calibrating FLOP budget")
    print("=" * 60)
    result = run_single(
        dropout=0.0,
        seed=42,
        flop_budget=compute_flop_budget(200),
        target_steps=200,
    )
    if result:
        print(f"\nPilot result: {json.dumps(result, indent=2)}")
        # Estimate full budget
        per_step_time = result["wall_time"] / result["steps"]
        est_2k = per_step_time * 2000
        print(f"\nEstimated time for 2000 steps: {est_2k:.0f}s")
        est_full = est_2k * 9
        print(f"Estimated time for full grid (9 runs): {est_full:.0f}s")


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-007 grid sweep."""
    parser = argparse.ArgumentParser(
        description=(
            "HYP-007: Test-time compute scaling on modular arithmetic"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Single run to calibrate FLOP budget",
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

    if args.pilot:
        run_pilot()
        return

    grid = build_grid()
    flop_budget = compute_flop_budget(args.target_steps)

    print("HYP-007: Test-Time Compute Scaling")
    print(f"Grid: {len(grid)} runs")
    print(f"  Dropout rates: {DROPOUT_RATES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  K values: {K_VALUES}")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  Modulus: {MODULUS}")
    print(f"  FLOP budget: {flop_budget:.2e} per run")
    print(f"  Target steps: ~{args.target_steps}")
    print(f"  LR: {LEARNING_RATE}")

    if args.max_runs:
        grid = grid[: args.max_runs]
        print(f"  Limited to {len(grid)} runs")

    results: list[dict[str, Any]] = []
    for i, (dropout, seed) in enumerate(grid):
        print(f"\n[{i + 1}/{len(grid)}]", end="")
        result = run_single(
            dropout,
            seed,
            flop_budget,
            dry_run=args.dry_run,
            target_steps=args.target_steps,
        )
        if result:
            results.append(result)

    if not results:
        return

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    header = f"{'Run':<25} {'Val':>7} {'p@1':>6} {'p@8':>6} {'p@64':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['run']:<25} "
            f"{r['val_loss']:>7.4f} "
            f"{r.get('pass_at_1', 0):>6.3f} "
            f"{r.get('pass_at_8', 0):>6.3f} "
            f"{r.get('pass_at_64', 0):>6.3f}"
        )

    # Save results
    out = Path("experiments") / "hyp007_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
