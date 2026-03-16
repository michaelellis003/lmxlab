"""HYP-011: Per-token loss decomposition (ANOM-015).

Pre-registered hypothesis:
    Does the val_loss vs pass@k inversion (ANOM-015) arise
    because SSM/hybrid models predict prompt tokens better
    while pure attention (LLaMA) predicts the answer token
    better?

Competing hypotheses:
    H11-a (Prompt-token dominance): Hybrids have much lower
        prompt-token loss; LLaMA has much lower answer-token
        loss. Inversion is fully explained.
    H11-b (Calibration): Similar per-position losses, but
        LLaMA's answer-token logits are better calibrated.
    H11-c (Training dynamics): Inversion is an artifact of
        FLOP-matching (LLaMA gets 2000 steps vs ~1667).
    H11-d (Null): Inversion is within seed noise.

Design:
    4 architectures (LLaMA, Falcon-H1, Jamba, Bamba)
    x 3 seeds (42, 43, 44)
    = 12 runs, FLOP-matched via FLOPCounter
    dropout=0.0 (per HYP-007)
    Dataset: modular arithmetic (a+b) mod 97
    Primary new metric: per-position cross-entropy loss
    Also: pass@k curves for HYP-008 replication

Usage:
    uv run python recipes/hyp011_token_loss_decomp.py
    uv run python recipes/hyp011_token_loss_decomp.py --dry-run
    uv run python recipes/hyp011_token_loss_decomp.py --pilot
    uv run python recipes/hyp011_token_loss_decomp.py --max-runs 1
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

    Same as HYP-008 for replication.

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

    For each test example "a + b = c\\n", forward the full
    sequence and extract loss at each position. Groups into
    prompt_loss (predicting a, +, b, =, space tokens) and
    answer_loss (predicting the answer token c).

    Also computes answer-token entropy and top-5 logit mass.

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

    prompt_losses: list[float] = []
    answer_losses: list[float] = []
    answer_entropies: list[float] = []
    answer_top5_masses: list[float] = []
    answer_correct_probs: list[float] = []

    for a, b, c in dataset._pairs:
        # Full sequence: "a + b = c\n"
        text = f"{a} + {b} = {c}\n"
        tokens = tokenizer.encode(text)
        token_arr = mx.array([tokens])  # (1, seq_len)

        # Forward pass
        logits, _ = model(token_arr)
        mx.eval(logits)
        # logits shape: (1, seq_len, vocab_size)
        logits = logits[0]  # (seq_len, vocab_size)

        # Prompt: "a + b ="
        prompt_text = f"{a} + {b} ="
        prompt_toks = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_toks)

        # The answer token is predicted at position prompt_len-1
        # (the logits at position i predict token i+1)
        # So: positions 0..prompt_len-2 predict prompt tokens
        #     position prompt_len-1 predicts the answer token

        # Cross-entropy at each position for next-token
        # prediction: CE(position i) = -log P(token_{i+1})
        for i in range(len(tokens) - 1):
            target = tokens[i + 1]
            logit_row = logits[i]
            # log-softmax: log(softmax(x)) = x - logsumexp(x)
            log_probs = logit_row - mx.logsumexp(logit_row)
            ce = -log_probs[target].item()

            if i < prompt_len - 1:
                # Predicting a prompt token
                prompt_losses.append(ce)
            elif i == prompt_len - 1:
                # Predicting the answer token
                answer_losses.append(ce)

                # Answer-token entropy
                probs = mx.softmax(logits[i])
                mx.eval(probs)
                p = probs.tolist()
                entropy = -sum(pi * math.log(pi + 1e-30) for pi in p)
                answer_entropies.append(entropy)

                # Top-5 mass over valid answers
                answer_probs = [probs[aid].item() for aid in answer_ids]
                top5 = sorted(answer_probs, reverse=True)[:5]
                answer_top5_masses.append(sum(top5))

                # Probability of correct answer
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
        "loss_ratio": (
            (sum(answer_losses) / n_answer)
            / (sum(prompt_losses) / n_prompt + 1e-10)
            if n_prompt and n_answer
            else 0.0
        ),
    }


def run_single(
    arch_name: str,
    seed: int,
    flop_budget: int,
    dry_run: bool = False,
    target_steps: int = 2000,
) -> dict[str, Any] | None:
    """Train one model and evaluate per-token loss + pass@k.

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

    # Data
    train_ds = ModularArithmeticDataset(p=MODULUS, split="train", seed=42)
    test_ds = ModularArithmeticDataset(p=MODULUS, split="test", seed=42)
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
        name="HYP-011",
        description=run_name,
        time_budget_s=1200.0,
        seed=seed,
        output_dir="experiments",
    )
    runner = MLflowExperimentRunner(
        exp_config,
        tags={
            "architecture": arch_name,
            "hypothesis": "HYP-011",
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

    # ── Phase 2: Evaluate pass@k (HYP-008 replication) ──
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

    # ── Phase 3: Per-token loss decomposition (NEW) ──
    print("  Evaluating per-token loss decomposition...")
    decomp_start = time.monotonic()
    decomp_results = evaluate_per_token_loss(
        model=model,
        dataset=test_ds,
    )
    decomp_elapsed = time.monotonic() - decomp_start
    print(f"  decomposition eval: {decomp_elapsed:.1f}s")
    print(f"    prompt_loss: {decomp_results['prompt_loss']:.4f}")
    print(f"    answer_loss: {decomp_results['answer_loss']:.4f}")
    print(f"    loss_ratio: {decomp_results['loss_ratio']:.2f}")
    print(f"    answer_entropy: {decomp_results['answer_entropy']:.4f}")
    print(f"    answer_top5_mass: {decomp_results['answer_top5_mass']:.4f}")
    print(
        f"    answer_correct_prob: {decomp_results['answer_correct_prob']:.6f}"
    )

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
        **decomp_results,
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
        **decomp_results,
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
    """Print analysis tables with per-token decomposition."""
    import statistics

    print(f"\n{'=' * 70}")
    print("Per-Token Loss Decomposition by Architecture")
    print(f"{'=' * 70}")

    by_arch: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        arch = r["architecture"]
        by_arch.setdefault(arch, []).append(r)

    # Main decomposition table
    header = (
        f"{'Arch':<12} {'ValLoss':>8} {'Prompt':>8} "
        f"{'Answer':>8} {'Ratio':>6} "
        f"{'AnsEnt':>8} {'AnsP':>8} {'p@1':>7} "
        f"{'p@64':>7}"
    )
    print(header)
    print("-" * len(header))

    for arch, runs in sorted(by_arch.items()):
        val = statistics.mean([r["val_loss"] for r in runs])
        pl = statistics.mean([r["prompt_loss"] for r in runs])
        al = statistics.mean([r["answer_loss"] for r in runs])
        ratio = al / (pl + 1e-10)
        ent = statistics.mean([r["answer_entropy"] for r in runs])
        ap = statistics.mean([r["answer_correct_prob"] for r in runs])
        p1 = statistics.mean([r.get("pass_at_1", 0) for r in runs])
        p64 = statistics.mean([r.get("pass_at_64", 0) for r in runs])

        print(
            f"{arch:<12} {val:>8.4f} {pl:>8.4f} "
            f"{al:>8.4f} {ratio:>6.1f} "
            f"{ent:>8.4f} {ap:>8.6f} {p1:>7.4f} "
            f"{p64:>7.4f}"
        )

    # Hypothesis test summary
    print(f"\n{'=' * 70}")
    print("H11-a Test: Prompt vs Answer Loss by Architecture")
    print(f"{'=' * 70}")

    # Get LLaMA stats as reference
    if "llama" in by_arch:
        llama_runs = by_arch["llama"]
        llama_al = statistics.mean([r["answer_loss"] for r in llama_runs])
        llama_pl = statistics.mean([r["prompt_loss"] for r in llama_runs])
        print(f"LLaMA prompt_loss: {llama_pl:.4f}")
        print(f"LLaMA answer_loss: {llama_al:.4f}")
        print()

        for arch, runs in sorted(by_arch.items()):
            if arch == "llama":
                continue
            al = statistics.mean([r["answer_loss"] for r in runs])
            pl = statistics.mean([r["prompt_loss"] for r in runs])
            pl_diff = (pl - llama_pl) / llama_pl * 100
            al_diff = (al - llama_al) / llama_al * 100
            print(
                f"{arch:<12} prompt_loss: {pl:.4f} "
                f"({pl_diff:+.1f}% vs LLaMA)  "
                f"answer_loss: {al:.4f} "
                f"({al_diff:+.1f}% vs LLaMA)"
            )

    # Pass@k replication check
    print(f"\n{'=' * 70}")
    print("HYP-008 Replication Check")
    print(f"{'=' * 70}")
    rep_header = (
        f"{'Arch':<12} {'Val':>7} {'p@1':>7} "
        f"{'p@16':>7} {'p@64':>7} {'p64/p1':>7}"
    )
    print(rep_header)
    print("-" * len(rep_header))
    for arch, runs in sorted(by_arch.items()):
        val = statistics.mean([r["val_loss"] for r in runs])
        p1 = statistics.mean([r.get("pass_at_1", 0) for r in runs])
        p16 = statistics.mean([r.get("pass_at_16", 0) for r in runs])
        p64 = statistics.mean([r.get("pass_at_64", 0) for r in runs])
        ratio = p64 / p1 if p1 > 0 else 0
        print(
            f"{arch:<12} {val:>7.4f} "
            f"{p1:>7.4f} {p16:>7.4f} "
            f"{p64:>7.4f} {ratio:>7.1f}x"
        )


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-011 grid sweep."""
    parser = argparse.ArgumentParser(
        description=("HYP-011: Per-token loss decomposition (ANOM-015)"),
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

    print("HYP-011: Per-Token Loss Decomposition (ANOM-015)")
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
    print("  NEW: Per-token loss decomposition eval")

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

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    header = (
        f"{'Run':<25} {'Val':>7} {'Prompt':>7} "
        f"{'Answer':>7} {'p@1':>6} {'p@64':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['run']:<25} "
            f"{r['val_loss']:>7.4f} "
            f"{r.get('prompt_loss', 0):>7.4f} "
            f"{r.get('answer_loss', 0):>7.4f} "
            f"{r.get('pass_at_1', 0):>6.3f} "
            f"{r.get('pass_at_64', 0):>6.3f}"
        )

    # Detailed analysis
    analyze_results(results)

    # Save results
    out = Path("experiments") / "hyp011_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
