"""HYP-008: SSM/hybrid test-time compute scaling.

Pre-registered hypothesis:
    Does test-time compute scaling (best-of-N with execution
    verification) work for SSM and hybrid architectures at 10M
    params on modular arithmetic, and how does it compare to
    pure attention (LLaMA)?

Competing hypotheses:
    H8-a (Architecture-independent): All architectures show
        similar pass@k scaling exponents — TTC effectiveness
        depends on model quality, not architecture type.
    H8-b (Attention advantage): Pure attention (LLaMA) has
        steeper pass@k curves than SSM/hybrid models because
        attention's explicit context access generates more
        diverse outputs.
    H8-c (Hybrid advantage): Hybrid models (Falcon-H1, Jamba,
        Bamba) show steeper pass@k curves because SSM state
        + attention provides complementary generation modes.
    H8-d (SSM disadvantage): SSM-heavy architectures have
        flatter pass@k curves — fixed-size state limits output
        diversity.

Design:
    4 architectures (LLaMA, Falcon-H1, Jamba, Bamba)
    x 3 seeds (42, 43, 44)
    = 12 runs, FLOP-matched via FLOPCounter
    dropout=0.0 only (HYP-007 showed dropout hurts diversity)
    Dataset: modular arithmetic (a+b) mod 97
    Primary metric: pass@k curves (k=1,2,4,8,16,32,64)

Usage:
    uv run python recipes/hyp008_ssm_ttc.py
    uv run python recipes/hyp008_ssm_ttc.py --dry-run
    uv run python recipes/hyp008_ssm_ttc.py --pilot
    uv run python recipes/hyp008_ssm_ttc.py --max-runs 1
    uv run python recipes/hyp008_ssm_ttc.py --target-steps 500
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
from lmxlab.models.bamba import bamba_10m
from lmxlab.models.base import LanguageModel, ModelConfig
from lmxlab.models.falcon import falcon_h1_10m
from lmxlab.models.jamba import jamba_10m
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

ARCHITECTURES: dict[str, Any] = {
    "llama": llama_10m,
    "falcon_h1": falcon_h1_10m,
    "jamba": jamba_10m,
    "bamba": bamba_10m,
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
    """Generate all (arch_name, seed) combinations."""
    return [(arch, seed) for arch in ARCHITECTURES for seed in SEEDS]


def compute_flop_budget(target_steps: int = 2000) -> int:
    """Compute FLOP budget from LLaMA-10M reference.

    Uses LLaMA-10M as the reference to match HYP-007.

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

    Uses single forward pass (fast path): the answer is always
    a single token, so we forward the prompt once and sample
    from the next-token logits.

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
    arch_name: str,
    seed: int,
    flop_budget: int,
    dry_run: bool = False,
    target_steps: int = 2000,
) -> dict[str, Any] | None:
    """Train one model and evaluate pass@k.

    Args:
        arch_name: Architecture name (key in ARCHITECTURES).
        seed: Random seed.
        flop_budget: FLOP budget for training.
        dry_run: If True, skip actual training.
        target_steps: Target steps (for display).

    Returns:
        Result dict or None if dry_run.
    """
    run_name = f"{arch_name}_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  arch={arch_name}, seed={seed}")

    if dry_run:
        config = ARCHITECTURES[arch_name]()
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

    # Model
    model_config: ModelConfig = ARCHITECTURES[arch_name]()
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
        name="HYP-008",
        description=run_name,
        time_budget_s=1200.0,
        seed=seed,
        output_dir="experiments",
    )
    runner = MLflowExperimentRunner(
        exp_config,
        tags={
            "architecture": arch_name,
            "hypothesis": "HYP-008",
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
            "architecture": arch_name,
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
        "architecture": arch_name,
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
    """Quick single run per arch to calibrate."""
    print("=" * 60)
    print("PILOT RUN: calibrating FLOP budget")
    print("=" * 60)
    budget = compute_flop_budget(200)
    for arch_name in ARCHITECTURES:
        result = run_single(
            arch_name=arch_name,
            seed=42,
            flop_budget=budget,
            target_steps=200,
        )
        if result:
            per_step = result["wall_time"] / result["steps"]
            est_2k = per_step * 2000
            print(f"\n  {arch_name}: ~{est_2k:.0f}s for 2K steps")


# ── Analysis ─────────────────────────────────────────────


def analyze_results(
    results: list[dict[str, Any]],
) -> None:
    """Print analysis table grouped by architecture."""
    import statistics

    print(f"\n{'=' * 70}")
    print("Analysis by Architecture")
    print(f"{'=' * 70}")

    # Group by architecture
    by_arch: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        arch = r["architecture"]
        by_arch.setdefault(arch, []).append(r)

    # Summary table
    header = (
        f"{'Arch':<12} {'Val':>7} {'p@1':>7} "
        f"{'p@16':>7} {'p@64':>7} {'p16/p1':>7} "
        f"{'p64/p1':>7}"
    )
    print(header)
    print("-" * len(header))

    arch_stats: dict[str, dict[str, float]] = {}
    for arch, runs in sorted(by_arch.items()):
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

        arch_stats[arch] = {
            "val_loss": mean_val,
            "pass_at_1": mean_p1,
            "pass_at_16": mean_p16,
            "pass_at_64": mean_p64,
            "ratio_16": ratio_16,
            "ratio_64": ratio_64,
        }

        print(
            f"{arch:<12} {mean_val:>7.4f} "
            f"{mean_p1:>7.4f} {mean_p16:>7.4f} "
            f"{mean_p64:>7.4f} {ratio_16:>7.1f}x "
            f"{ratio_64:>7.1f}x"
        )

    # Pass@k curves per architecture
    print(f"\n{'=' * 70}")
    print("Pass@k Curves (mean across seeds)")
    print(f"{'=' * 70}")
    k_header = f"{'Arch':<12}"
    for k in K_VALUES:
        k_header += f" {'k=' + str(k):>7}"
    print(k_header)
    print("-" * len(k_header))
    for arch, runs in sorted(by_arch.items()):
        line = f"{arch:<12}"
        for k in K_VALUES:
            key = f"pass_at_{k}"
            vals = [r.get(key, 0) for r in runs]
            mean = statistics.mean(vals) if vals else 0
            line += f" {mean:>7.4f}"
        print(line)


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-008 grid sweep."""
    parser = argparse.ArgumentParser(
        description=("HYP-008: SSM/hybrid test-time compute scaling"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Single run per arch to calibrate",
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

    print("HYP-008: SSM/Hybrid Test-Time Compute Scaling")
    print(f"Grid: {len(grid)} runs")
    print(f"  Architectures: {list(ARCHITECTURES.keys())}")
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
    for i, (arch_name, seed) in enumerate(grid):
        print(f"\n[{i + 1}/{len(grid)}]", end="")
        result = run_single(
            arch_name,
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

    # Detailed analysis
    analyze_results(results)

    # Save results
    out = Path("experiments") / "hyp008_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
