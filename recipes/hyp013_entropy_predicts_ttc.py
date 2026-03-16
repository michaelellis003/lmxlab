"""HYP-013: Does answer-token entropy predict TTC amplification?

Pre-registered hypothesis:
    The ~12-15x TTC amplification on addition vs ~3.8x on
    multiplication (HYP-012) is predicted by the answer-token
    entropy of the model's distribution. Higher entropy means
    more room for sampling to find correct answers.

Competing hypotheses:
    H13-a (Entropy predicts): Answer-token entropy strongly
        correlates (|r| > 0.8) with TTC amplification
        (p@64/p@1) across tasks and seeds.
    H13-b (Correct-prob predicts): P(correct) at the answer
        token is the primary predictor, not entropy. High
        P(correct) -> high p@1 -> low amplification.
    H13-c (Both contribute): Both entropy and P(correct) are
        needed — neither alone explains the variance.
    H13-d (Null): No clean relationship. Seed variance
        dominates.

Design:
    2 operations (add, mul) x 3 seeds = 6 runs
    LLaMA-10M only, dropout=0.0
    Per-token loss decomposition (from HYP-011)
    pass@k evaluation (from HYP-012)
    Primary analysis: correlation between answer-token
    entropy and p@64/p@1 ratio

Usage:
    uv run python recipes/hyp013_entropy_predicts_ttc.py
    uv run python recipes/hyp013_entropy_predicts_ttc.py --dry-run
    uv run python recipes/hyp013_entropy_predicts_ttc.py --pilot
    uv run python recipes/hyp013_entropy_predicts_ttc.py --max-runs 1
"""

import argparse
import json
import math
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


def evaluate_per_token_loss(
    model: LanguageModel,
    dataset: ModularArithmeticDataset,
) -> dict[str, float]:
    """Compute per-position cross-entropy on test prompts.

    Works with both add and mul operations by using the
    dataset's operation symbol.

    Args:
        model: Trained language model.
        dataset: Test split of ModularArithmeticDataset.

    Returns:
        Dict with prompt_loss, answer_loss, answer_entropy,
        answer_top5_mass, answer_correct_prob.
    """
    model.eval()
    tokenizer = dataset._tokenizer
    answer_ids = dataset.answer_token_ids
    op_symbol = dataset._op_symbol

    prompt_losses: list[float] = []
    answer_losses: list[float] = []
    answer_entropies: list[float] = []
    answer_top5_masses: list[float] = []
    answer_correct_probs: list[float] = []

    for a, b, c in dataset._pairs:
        text = f"{a} {op_symbol} {b} = {c}\n"
        tokens = tokenizer.encode(text)
        token_arr = mx.array([tokens])

        logits, _ = model(token_arr)
        mx.eval(logits)
        logits = logits[0]

        prompt_text = f"{a} {op_symbol} {b} ="
        prompt_toks = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_toks)

        for i in range(len(tokens) - 1):
            target = tokens[i + 1]
            logit_row = logits[i]
            log_probs = logit_row - mx.logsumexp(logit_row)
            ce = -log_probs[target].item()

            if i < prompt_len - 1:
                prompt_losses.append(ce)
            elif i == prompt_len - 1:
                answer_losses.append(ce)

                probs = mx.softmax(logits[i])
                mx.eval(probs)
                p = probs.tolist()
                entropy = -sum(pi * math.log(pi + 1e-30) for pi in p)
                answer_entropies.append(entropy)

                answer_probs = [probs[aid].item() for aid in answer_ids]
                top5 = sorted(answer_probs, reverse=True)[:5]
                answer_top5_masses.append(sum(top5))

                correct_prob = probs[answer_ids[c]].item()
                answer_correct_probs.append(correct_prob)

    model.train()

    n_prompt = len(prompt_losses)
    n_answer = len(answer_losses)

    return {
        "prompt_loss": (sum(prompt_losses) / n_prompt if n_prompt else 0.0),
        "answer_loss": (sum(answer_losses) / n_answer if n_answer else 0.0),
        "answer_entropy": (
            sum(answer_entropies) / n_answer if n_answer else 0.0
        ),
        "answer_top5_mass": (
            sum(answer_top5_masses) / n_answer if n_answer else 0.0
        ),
        "answer_correct_prob": (
            sum(answer_correct_probs) / n_answer if n_answer else 0.0
        ),
    }


def run_single(
    operation: str,
    seed: int,
    flop_budget: int,
    dry_run: bool = False,
    target_steps: int = 2000,
) -> dict[str, Any] | None:
    """Train one model, evaluate pass@k + per-token loss.

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

    # Model
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

    # Data
    train_ds = ModularArithmeticDataset(
        p=MODULUS,
        split="train",
        seed=42,
        operation=operation,
    )
    test_ds = ModularArithmeticDataset(
        p=MODULUS,
        split="test",
        seed=42,
        operation=operation,
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
        name="HYP-013",
        description=run_name,
        time_budget_s=1200.0,
        seed=seed,
        output_dir="experiments",
    )
    runner = MLflowExperimentRunner(
        exp_config,
        tags={
            "operation": operation,
            "hypothesis": "HYP-013",
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

    # ── Phase 3: Per-token loss decomposition ──
    print("  Evaluating per-token loss...")
    ptl_start = time.monotonic()
    ptl_results = evaluate_per_token_loss(model=model, dataset=test_ds)
    ptl_elapsed = time.monotonic() - ptl_start
    print(f"  per-token loss eval: {ptl_elapsed:.1f}s")
    for k_name, val in ptl_results.items():
        print(f"    {k_name}: {val:.4f}")

    # ── Phase 4: Log results ──
    metrics = {
        "val_loss": val_tracker.best_val_loss,
        "best_val_loss": val_tracker.best_val_loss,
        "train_loss": train_loss,
        "train_val_gap": train_loss - val_tracker.best_val_loss,
        "init_val_loss": val_tracker.init_val_loss,
        "steps": steps,
        "total_flops": flop_counter.total_flops,
        **pass_at_k_results,
        **ptl_results,
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

    # Compute amplification
    p1 = pass_at_k_results.get("pass_at_1", 0)
    p64 = pass_at_k_results.get("pass_at_64", 0)
    amp = p64 / p1 if p1 > 0 else 0

    return {
        "run": run_name,
        "operation": operation,
        "seed": seed,
        "val_loss": val_tracker.best_val_loss,
        "train_loss": train_loss,
        "steps": steps,
        "wall_time": total_elapsed,
        "total_flops": flop_counter.total_flops,
        "params": n_params,
        "amplification": amp,
        **pass_at_k_results,
        **ptl_results,
    }


def run_pilot() -> None:
    """Quick single run per operation to calibrate."""
    print("=" * 60)
    print("PILOT RUN: calibrating")
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
            print(f"  amp={result['amplification']:.1f}x")
            print(f"  entropy={result['answer_entropy']:.3f}")


# ── Analysis ─────────────────────────────────────────────


def analyze_results(
    results: list[dict[str, Any]],
) -> None:
    """Print analysis: entropy vs amplification."""
    import statistics

    print(f"\n{'=' * 70}")
    print("Per-Run Entropy vs Amplification")
    print(f"{'=' * 70}")

    header = (
        f"{'Run':<15} {'Entropy':>8} {'P(corr)':>8} "
        f"{'p@1':>7} {'p@64':>7} {'Amp':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['run']:<15} "
            f"{r['answer_entropy']:>8.3f} "
            f"{r['answer_correct_prob']:>8.4f} "
            f"{r.get('pass_at_1', 0):>7.4f} "
            f"{r.get('pass_at_64', 0):>7.4f} "
            f"{r['amplification']:>7.1f}x"
        )

    # Group by operation
    print(f"\n{'=' * 70}")
    print("Means by Operation")
    print(f"{'=' * 70}")

    by_op: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_op.setdefault(r["operation"], []).append(r)

    for op, runs in sorted(by_op.items()):
        ent = statistics.mean([r["answer_entropy"] for r in runs])
        pcorr = statistics.mean([r["answer_correct_prob"] for r in runs])
        p1 = statistics.mean([r.get("pass_at_1", 0) for r in runs])
        p64 = statistics.mean([r.get("pass_at_64", 0) for r in runs])
        amp = statistics.mean([r["amplification"] for r in runs])
        print(
            f"{op:<8} entropy={ent:.3f}, "
            f"P(corr)={pcorr:.4f}, "
            f"p@1={p1:.4f}, p@64={p64:.4f}, "
            f"amp={amp:.1f}x"
        )

    # Correlation analysis
    if len(results) >= 4:
        print(f"\n{'=' * 70}")
        print("Correlation Analysis (across all 6 runs)")
        print(f"{'=' * 70}")

        ents = [r["answer_entropy"] for r in results]
        amps = [r["amplification"] for r in results]
        pcorrs = [r["answer_correct_prob"] for r in results]
        p1s = [r.get("pass_at_1", 0) for r in results]

        r_ent_amp = _pearson(ents, amps)
        r_pcorr_amp = _pearson(pcorrs, amps)
        r_p1_amp = _pearson(p1s, amps)
        r_ent_p1 = _pearson(ents, p1s)

        print(f"  r(entropy, amp):      {r_ent_amp:+.3f}")
        print(f"  r(P(correct), amp):   {r_pcorr_amp:+.3f}")
        print(f"  r(pass@1, amp):       {r_p1_amp:+.3f}")
        print(f"  r(entropy, pass@1):   {r_ent_p1:+.3f}")

        print("\nInterpretation:")
        if abs(r_ent_amp) > 0.8:
            print(
                "  -> Entropy STRONGLY predicts "
                "amplification (H13-a supported)"
            )
        elif abs(r_pcorr_amp) > 0.8 > abs(r_ent_amp):
            print(
                "  -> P(correct) predicts amp better "
                "than entropy (H13-b supported)"
            )
        elif abs(r_ent_amp) > 0.5 and abs(r_pcorr_amp) > 0.5:
            print("  -> Both contribute (H13-c supported)")
        else:
            print("  -> No clean relationship (H13-d — null)")


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx_val = sum(x) / n
    my_val = sum(y) / n
    sx = sum((xi - mx_val) ** 2 for xi in x) ** 0.5
    sy = sum((yi - my_val) ** 2 for yi in y) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum(
        (xi - mx_val) * (yi - my_val) for xi, yi in zip(x, y, strict=True)
    )
    return cov / (sx * sy)


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-013 grid sweep."""
    parser = argparse.ArgumentParser(
        description=("HYP-013: Entropy predicts TTC amplification"),
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

    print("HYP-013: Entropy Predicts TTC Amplification")
    print(f"Grid: {len(grid)} runs")
    print(f"  Operations: {OPERATIONS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  K values: {K_VALUES}")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  Modulus: {MODULUS}")
    print(f"  FLOP budget: {flop_budget:.2e} per run")
    print(f"  Target steps: ~{args.target_steps}")
    print(f"  LR: {LEARNING_RATE}")
    print("  Dropout: 0.0 (fixed)")

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
    header = (
        f"{'Run':<15} {'Val':>7} {'p@1':>6} {'p@64':>6} {'Amp':>6} {'Ent':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['run']:<15} "
            f"{r['val_loss']:>7.4f} "
            f"{r.get('pass_at_1', 0):>6.3f} "
            f"{r.get('pass_at_64', 0):>6.3f} "
            f"{r['amplification']:>6.1f}x "
            f"{r['answer_entropy']:>6.3f}"
        )

    # Detailed analysis
    analyze_results(results)

    # Save results
    out = Path("experiments") / "hyp013_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
