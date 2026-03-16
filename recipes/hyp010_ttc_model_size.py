"""HYP-010: TTC scaling exponent vs model size.

Pre-registered hypothesis:
    How does the TTC amplification factor (pass@64/pass@1)
    change as model size increases from 10M to 30M params
    on modular arithmetic (a+b) mod 97?

Competing hypotheses:
    H10-a (Stable exponent): p@64/p@1 ratios within 2x
        across sizes — TTC is task-dependent, not size
    H10-b (Both up): 30M has higher pass@1 AND steeper
        TTC curve (higher p@64/p@1 ratio)
    H10-c (Exponent down): 30M has higher pass@1 but
        lower p@64/p@1 — larger model is more deterministic
    H10-d (Diminishing returns): 30M barely improves over
        10M — task bottleneck is algorithmic, not capacity

Design:
    2 model sizes (10M, 30M) x 3 seeds (42, 43, 44)
    = 6 runs, FLOP-matched within each size class
    Dataset: modular arithmetic (a+b) mod 97
    Primary metric: pass@k curves (k=1,2,4,8,16,32,64)

Usage:
    uv run python recipes/hyp010_ttc_model_size.py
    uv run python recipes/hyp010_ttc_model_size.py --dry-run
    uv run python recipes/hyp010_ttc_model_size.py --pilot
    uv run python recipes/hyp010_ttc_model_size.py --max-runs 1
    uv run python recipes/hyp010_ttc_model_size.py --target-steps 500
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
from lmxlab.models.base import LanguageModel, ModelConfig
from lmxlab.models.llama import llama_10m, llama_30m
from lmxlab.training.callbacks import (
    FLOPCounter,
    ValTracker,
    standard_callbacks,
)
from lmxlab.training.config import TrainConfig
from lmxlab.training.hardware import detect_peak_tflops
from lmxlab.training.trainer import Trainer

# ── Grid definition ──────────────────────────────────────

MODEL_CONFIGS: dict[str, callable] = {
    "llama_10m": llama_10m,
    "llama_30m": llama_30m,
}
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


def build_grid() -> list[tuple[str, int]]:
    """Generate all (model_name, seed) combinations."""
    return [(name, seed) for name in MODEL_CONFIGS for seed in SEEDS]


def make_config(model_name: str) -> ModelConfig:
    """Create model config with dropout=0.0.

    Args:
        model_name: Key into MODEL_CONFIGS.

    Returns:
        ModelConfig with dropout disabled.
    """
    config = MODEL_CONFIGS[model_name]()
    block = replace(config.block, dropout=0.0)
    return replace(config, block=block)


def compute_flop_budget(model_name: str, target_steps: int = 2000) -> int:
    """Compute FLOP budget for a given model size.

    Args:
        model_name: Key into MODEL_CONFIGS.
        target_steps: Target number of training steps.

    Returns:
        Total FLOP budget.
    """
    config = MODEL_CONFIGS[model_name]()
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

    Uses single forward pass (fast path): the answer is
    always a single token, so we forward the prompt once
    and sample from the next-token logits.

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

        # Single forward pass
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
    model_name: str,
    seed: int,
    flop_budget: int,
    dry_run: bool = False,
    target_steps: int = 2000,
) -> dict[str, Any] | None:
    """Train one model and evaluate pass@k.

    Args:
        model_name: Model size key.
        seed: Random seed.
        flop_budget: FLOP budget for training.
        dry_run: If True, skip actual training.
        target_steps: Target steps (for display).

    Returns:
        Result dict or None if dry_run.
    """
    run_name = f"{model_name}_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  model={model_name}, seed={seed}")

    if dry_run:
        config = make_config(model_name)
        model = LanguageModel(config)
        mx.eval(model.parameters())
        n_params = model.count_parameters()
        print(f"  params={n_params:,}")
        print("  [DRY RUN] Skipping.")
        return None

    mx.random.seed(seed)

    # Model
    model_config = make_config(model_name)
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

    # Hold out 10% of train tokens for val loss
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
        name="HYP-010",
        description=run_name,
        time_budget_s=1200.0,
        seed=seed,
        output_dir="experiments",
    )
    runner = MLflowExperimentRunner(
        exp_config,
        tags={
            "model_size": model_name,
            "hypothesis": "HYP-010",
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
            "model_name": model_name,
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
        "model_name": model_name,
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
    """Quick single run per model size to calibrate."""
    print("=" * 60)
    print("PILOT RUN: calibrating FLOP budget")
    print("=" * 60)
    for name in MODEL_CONFIGS:
        budget = compute_flop_budget(name, 200)
        result = run_single(
            name,
            seed=42,
            flop_budget=budget,
            target_steps=200,
        )
        if result:
            per_step = result["wall_time"] / result["steps"]
            est_2k = per_step * 2000
            print(f"\n{name}: {est_2k:.0f}s for 2000 steps")


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-010 grid sweep."""
    parser = argparse.ArgumentParser(
        description=("HYP-010: TTC scaling exponent vs model size"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Single run per size to calibrate",
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

    # Compute per-size FLOP budgets
    flop_budgets = {
        name: compute_flop_budget(name, args.target_steps)
        for name in MODEL_CONFIGS
    }

    print("HYP-010: TTC Scaling Exponent vs Model Size")
    print(f"Grid: {len(grid)} runs")
    print(f"  Model sizes: {list(MODEL_CONFIGS.keys())}")
    print(f"  Seeds: {SEEDS}")
    print(f"  K values: {K_VALUES}")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  Modulus: {MODULUS}")
    for name, budget in flop_budgets.items():
        print(f"  {name} FLOP budget: {budget:.2e}")
    print(f"  Target steps: ~{args.target_steps}")
    print(f"  LR: {LEARNING_RATE}")

    if args.max_runs:
        grid = grid[: args.max_runs]
        print(f"  Limited to {len(grid)} runs")

    results: list[dict[str, Any]] = []
    for i, (model_name, seed) in enumerate(grid):
        print(f"\n[{i + 1}/{len(grid)}]", end="")
        result = run_single(
            model_name,
            seed,
            flop_budgets[model_name],
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
    header = (
        f"{'Run':<25} {'Params':>8} {'Val':>7} "
        f"{'p@1':>6} {'p@16':>6} {'p@64':>6} "
        f"{'p64/p1':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        p1 = r.get("pass_at_1", 0)
        p64 = r.get("pass_at_64", 0)
        ratio = p64 / p1 if p1 > 0 else 0
        print(
            f"{r['run']:<25} "
            f"{r['params']:>8,} "
            f"{r['val_loss']:>7.4f} "
            f"{p1:>6.4f} "
            f"{r.get('pass_at_16', 0):>6.4f} "
            f"{p64:>6.4f} "
            f"{ratio:>7.1f}x"
        )

    # Per-size averages
    print(f"\n{'=' * 60}")
    print("Per-size averages (across seeds)")
    print(f"{'=' * 60}")
    for name in MODEL_CONFIGS:
        size_results = [r for r in results if r["model_name"] == name]
        if not size_results:
            continue
        n = len(size_results)
        avg_p1 = sum(r.get("pass_at_1", 0) for r in size_results) / n
        avg_p16 = sum(r.get("pass_at_16", 0) for r in size_results) / n
        avg_p64 = sum(r.get("pass_at_64", 0) for r in size_results) / n
        avg_val = sum(r["val_loss"] for r in size_results) / n
        ratio = avg_p64 / avg_p1 if avg_p1 > 0 else 0
        print(
            f"  {name:<12} "
            f"val={avg_val:.4f}  "
            f"p@1={avg_p1:.4f}  "
            f"p@16={avg_p16:.4f}  "
            f"p@64={avg_p64:.4f}  "
            f"p@64/p@1={ratio:.1f}x"
        )

    # Save results
    out = Path("experiments") / "hyp010_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
