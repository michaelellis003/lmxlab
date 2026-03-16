"""HYP-012: TTC amplification across tasks.

Pre-registered hypothesis:
    Is the ~12-15x TTC amplification factor (pass@64/pass@1)
    specific to modular addition, or does it generalize across
    modular arithmetic operations?

Competing hypotheses:
    H12-a (Task-independent): Multiplication shows a similar
        amplification factor (~10-20x) as addition.
    H12-b (Harder task, higher amp): Multiplication is harder,
        so lower pass@1 but proportionally higher amplification
        (>20x). More room for TTC to help.
    H12-c (Harder task, lower amp): Multiplication is harder,
        so models can't produce correct answers even with
        sampling. Amplification < 5x.
    H12-d (Null): Multiplication is too hard at 10M params.
        pass@1 ≈ 0, no meaningful amplification.

Design:
    2 operations (add, mul) — add is control from HYP-007
    x 3 seeds (42, 43, 44)
    = 6 runs (only mul needs training; add reuses HYP-007)
    But we train both for clean comparison.
    dropout=0.0 only (HYP-007 finding)
    Dataset: modular arithmetic mod 97
    Primary metric: pass@k curves, p@64/p@1 ratio

Usage:
    uv run python recipes/hyp012_ttc_cross_task.py
    uv run python recipes/hyp012_ttc_cross_task.py --dry-run
    uv run python recipes/hyp012_ttc_cross_task.py --pilot
    uv run python recipes/hyp012_ttc_cross_task.py --max-runs 1
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.modular_arithmetic import ModularArithmeticDataset
from lmxlab.eval.metrics import pass_at_k
from lmxlab.experiments.flops import estimate_flops_per_step
from lmxlab.experiments.mlflow import (
    MLflowCallback,
    MLflowExperimentRunner,
)
from lmxlab.experiments.runner import ExperimentConfig
from lmxlab.models.base import LanguageModel, ModelConfig
from lmxlab.models.llama import llama_10m
from lmxlab.training.callbacks import (
    FLOPCounter,
    ValTracker,
    standard_callbacks,
)
from lmxlab.training.config import TrainConfig
from lmxlab.training.hardware import detect_peak_tflops
from lmxlab.training.trainer import Trainer

# ── Grid definition ──────────────────────────────────────

OPERATIONS = ["add", "mul"]
SEEDS = [42, 43, 44]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
N_SAMPLES = 64
MODULUS = 97
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
SEQ_LEN = 256
EVAL_INTERVAL = 500
TEMPERATURE = 0.8


# ── Core functions ───────────────────────────────────────


def build_grid() -> list[tuple[str, int]]:
    """Generate all (operation, seed) combinations."""
    return [(op, seed) for op in OPERATIONS for seed in SEEDS]


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
    k_values: list[int],
    n_samples: int,
    temperature: float,
) -> dict[str, float]:
    """Evaluate pass@k on modular arithmetic.

    Args:
        model: Trained language model.
        dataset: Test split of ModularArithmeticDataset.
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
        prompt_batch = mx.broadcast_to(
            prompt_tokens[None, :],
            (n_samples, prompt_tokens.shape[0]),
        )
        logits, _ = model(prompt_batch)
        mx.eval(logits)

        next_logits = logits[:, -1, :] / temperature
        sampled = mx.random.categorical(next_logits)
        mx.eval(sampled)

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
    operation: str,
    seed: int,
    flop_budget: int,
    dry_run: bool = False,
    target_steps: int = 2000,
) -> dict[str, Any] | None:
    """Train one model and evaluate pass@k.

    Args:
        operation: ``'add'`` or ``'mul'``.
        seed: Random seed.
        flop_budget: FLOP budget for training.
        dry_run: If True, skip actual training.
        target_steps: Target steps (for display).

    Returns:
        Result dict or None if dry_run.
    """
    run_name = f"{operation}_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  operation={operation}, seed={seed}")

    if dry_run:
        config = llama_10m()
        model = LanguageModel(config)
        n_params = model.count_parameters()
        flops_per_step = estimate_flops_per_step(config, BATCH_SIZE, SEQ_LEN)
        est_steps = flop_budget / flops_per_step
        print(f"  params={n_params:,}")
        print(f"  flops/step={flops_per_step:.2e}")
        print(f"  est_steps={est_steps:.0f}")
        print("  [DRY RUN] Skipping.")
        return None

    mx.random.seed(seed)

    # Model — always LLaMA-10M
    model_config: ModelConfig = llama_10m()
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

    # Data — operation-specific
    train_ds = ModularArithmeticDataset(
        p=MODULUS, split="train", seed=42, operation=operation
    )
    test_ds = ModularArithmeticDataset(
        p=MODULUS, split="test", seed=42, operation=operation
    )
    train_tokens = train_ds.get_tokens()

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

    # Callbacks
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
        name="HYP-012",
        description=run_name,
        time_budget_s=1200.0,
        seed=seed,
        output_dir="experiments",
    )
    runner = MLflowExperimentRunner(
        exp_config,
        tags={
            "operation": operation,
            "hypothesis": "HYP-012",
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
            "operation": operation,
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
    gap = train_loss - val_tracker.best_val_loss
    print(f"  Gap:        {gap:+.4f}")
    print(f"  Wall time:  {total_elapsed:.1f}s")
    print(f"  FLOPs:      {flop_counter.total_flops:.2e}")
    print(f"  Status:     {entry.status}")

    return {
        "run": run_name,
        "operation": operation,
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
    """Quick single run per operation to calibrate."""
    print("=" * 60)
    print("PILOT RUN: calibrating FLOP budget")
    print("=" * 60)
    budget = compute_flop_budget(200)
    for op in OPERATIONS:
        result = run_single(
            operation=op,
            seed=42,
            flop_budget=budget,
            target_steps=200,
        )
        if result:
            per_step = result["wall_time"] / result["steps"]
            est_2k = per_step * 2000
            print(f"\n  {op}: ~{est_2k:.0f}s for 2K steps")


# ── Analysis ─────────────────────────────────────────────


def analyze_results(
    results: list[dict[str, Any]],
) -> None:
    """Print analysis tables grouped by operation."""
    import statistics

    print(f"\n{'=' * 70}")
    print("Analysis by Operation")
    print(f"{'=' * 70}")

    by_op: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        op = r["operation"]
        by_op.setdefault(op, []).append(r)

    # Summary table
    header = (
        f"{'Op':<8} {'Val':>7} {'p@1':>7} "
        f"{'p@16':>7} {'p@64':>7} {'p16/p1':>7} "
        f"{'p64/p1':>7}"
    )
    print(header)
    print("-" * len(header))

    for op, runs in sorted(by_op.items()):
        vals = [r["val_loss"] for r in runs]
        p1s = [r.get("pass_at_1", 0) for r in runs]
        p16s = [r.get("pass_at_16", 0) for r in runs]
        p64s = [r.get("pass_at_64", 0) for r in runs]

        mean_val = statistics.mean(vals)
        mean_p1 = statistics.mean(p1s)
        mean_p16 = statistics.mean(p16s)
        mean_p64 = statistics.mean(p64s)
        ratio_16 = mean_p16 / mean_p1 if mean_p1 > 0 else 0
        ratio_64 = mean_p64 / mean_p1 if mean_p1 > 0 else 0

        print(
            f"{op:<8} {mean_val:>7.4f} "
            f"{mean_p1:>7.4f} {mean_p16:>7.4f} "
            f"{mean_p64:>7.4f} {ratio_16:>7.1f}x "
            f"{ratio_64:>7.1f}x"
        )

    # Pass@k curves per operation
    print(f"\n{'=' * 70}")
    print("Pass@k Curves (mean across seeds)")
    print(f"{'=' * 70}")
    k_header = f"{'Op':<8}"
    for k in K_VALUES:
        k_header += f" {'k=' + str(k):>7}"
    print(k_header)
    print("-" * len(k_header))
    for op, runs in sorted(by_op.items()):
        line = f"{op:<8}"
        for k in K_VALUES:
            key = f"pass_at_{k}"
            vals = [r.get(key, 0) for r in runs]
            mean = statistics.mean(vals) if vals else 0
            line += f" {mean:>7.4f}"
        print(line)

    # Cross-task comparison
    if "add" in by_op and "mul" in by_op:
        print(f"\n{'=' * 70}")
        print("Cross-Task TTC Amplification Comparison")
        print(f"{'=' * 70}")
        add_runs = by_op["add"]
        mul_runs = by_op["mul"]

        add_p1 = statistics.mean([r.get("pass_at_1", 0) for r in add_runs])
        add_p64 = statistics.mean([r.get("pass_at_64", 0) for r in add_runs])
        mul_p1 = statistics.mean([r.get("pass_at_1", 0) for r in mul_runs])
        mul_p64 = statistics.mean([r.get("pass_at_64", 0) for r in mul_runs])

        add_amp = add_p64 / add_p1 if add_p1 > 0 else 0
        mul_amp = mul_p64 / mul_p1 if mul_p1 > 0 else 0
        ratio = mul_amp / add_amp if add_amp > 0 else 0

        print(
            f"Addition:       p@1={add_p1:.4f}, "
            f"p@64={add_p64:.4f}, amp={add_amp:.1f}x"
        )
        print(
            f"Multiplication: p@1={mul_p1:.4f}, "
            f"p@64={mul_p64:.4f}, amp={mul_amp:.1f}x"
        )
        print(f"Amp ratio (mul/add): {ratio:.2f}x")


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-012 grid sweep."""
    parser = argparse.ArgumentParser(
        description=("HYP-012: TTC amplification across tasks"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Single run per operation to calibrate",
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

    print("HYP-012: TTC Amplification Across Tasks")
    print(f"Grid: {len(grid)} runs")
    print(f"  Operations: {OPERATIONS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  K values: {K_VALUES}")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  Modulus: {MODULUS}")
    print(f"  FLOP budget: {flop_budget:.2e} per run")
    print(f"  Target steps: ~{args.target_steps}")
    print(f"  LR: {LEARNING_RATE}")
    print("  Dropout: 0.0 (fixed — HYP-007 finding)")

    if args.max_runs:
        grid = grid[: args.max_runs]
        print(f"  Limited to {len(grid)} runs")

    results: list[dict[str, Any]] = []
    for i, (operation, seed) in enumerate(grid):
        print(f"\n[{i + 1}/{len(grid)}]", end="")
        result = run_single(
            operation,
            seed,
            flop_budget,
            dry_run=args.dry_run,
            target_steps=args.target_steps,
        )
        if result:
            results.append(result)

    if not results:
        return

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    header = f"{'Run':<20} {'Val':>7} {'p@1':>6} {'p@8':>6} {'p@64':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['run']:<20} "
            f"{r['val_loss']:>7.4f} "
            f"{r.get('pass_at_1', 0):>6.3f} "
            f"{r.get('pass_at_8', 0):>6.3f} "
            f"{r.get('pass_at_64', 0):>6.3f}"
        )

    # Detailed analysis
    analyze_results(results)

    # Save results
    out = Path("experiments") / "hyp012_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
