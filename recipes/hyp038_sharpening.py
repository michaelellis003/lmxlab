"""HYP-038: Answer-Token Sharpening Dynamics During Latent Knowledge Phase

Pre-registered hypothesis:
    B-011 shows TTC reveals latent generalization before greedy accuracy.
    B-015 shows grokking onset is stochastic (rho ~0.1 across seeds).
    Combined: there's a "latent knowledge phase" where pass@64 is ~100%
    but pass@1 is near 0%. How does P(correct answer token) evolve during
    this phase?

Competing hypotheses:
    H38-a: Gradual sigmoid sharpening (transition width > 10K steps).
    H38-b: Phase transition (transition width < 5K steps, bimodal P).
    H38-c: Oscillatory (>3 cycles before stabilizing).

Design:
    5 seeds (42-46) x 30K steps. Measure P(correct), pass@1, pass@64,
    val_acc, val_loss every 2K steps on full val set.

Usage:
    uv run python recipes/hyp038_sharpening.py
    uv run python recipes/hyp038_sharpening.py --pilot
    uv run python recipes/hyp038_sharpening.py --max-runs 1
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from lmxlab.core.config import ModelConfig
from lmxlab.data.modular_arithmetic import ModularArithmeticDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.jamba import jamba_config

# ── Constants ────────────────────────────────────────────

SEEDS = list(range(42, 47))  # 5 seeds: 42..46
MODULUS = 97
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
BATCH_SIZE = 64
MAX_STEPS = 30_000
EVAL_EVERY = 2000  # Evaluate every 2K steps
N_SAMPLES = 64  # For pass@k estimation

# Known grok steps from HYP-016
GROK_STEPS = {
    42: 18000, 43: 12000, 44: 60000, 45: 48000, 46: 22000,
}

WARMUP_STEPS = 100

# ── Jamba MoE config (same as HYP-015/016/037) ──────────

GROK_COMMON = dict(
    vocab_size=50257,
    d_model=128,
    n_heads=4,
    n_kv_heads=2,
    d_ff=512,
    max_seq_len=64,
    tie_embeddings=True,
)

GROK_MAMBA = dict(
    mamba_n_heads=8,
    mamba_head_dim=32,
    ssm_state_size=16,
    mamba_expand=2,
    mamba_n_groups=1,
    mamba_chunk_size=64,
    conv_kernel_size=4,
)


def _config() -> ModelConfig:
    """Build Jamba MoE config (same as HYP-015/016)."""
    return jamba_config(
        n_layers=2,
        attn_every=2,
        n_experts=4,
        top_k_experts=2,
        moe_every=2,
        rope_theta=10000.0,
        **GROK_COMMON,
        **GROK_MAMBA,
    )


CONFIG = _config()


# ── Data helpers ─────────────────────────────────────────


def build_example_batches(
    dataset: ModularArithmeticDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[mx.array, mx.array]]:
    """Build padded batches of complete examples."""
    tokenizer = TiktokenTokenizer("gpt2")
    pairs = dataset._pairs  # noqa: SLF001

    all_seqs: list[list[int]] = []
    for a, b, c in pairs:
        text = f"{a} + {b} = {c}\n"
        toks = tokenizer.encode(text)
        all_seqs.append(toks)

    indices = rng.permutation(len(all_seqs))

    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < 2:
            continue
        seqs = [all_seqs[i] for i in batch_idx]
        max_len = max(len(s) for s in seqs)

        inputs = []
        targets = []
        for s in seqs:
            padded = s + [0] * (max_len - len(s))
            inputs.append(padded[:-1])
            targets.append(padded[1:])

        batches.append((mx.array(inputs), mx.array(targets)))

    return batches


def build_val_examples(
    dataset: ModularArithmeticDataset,
) -> list[tuple[list[int], int, tuple[int, int, int]]]:
    """Build val examples with (input_tokens, answer_token_id, triple).

    Returns list of (prompt_tokens, correct_answer_token, (a, b, c)).
    The prompt is everything up to (not including) the answer token.
    """
    tokenizer = TiktokenTokenizer("gpt2")
    pairs = dataset._pairs  # noqa: SLF001

    examples = []
    for a, b, c in pairs:
        text = f"{a} + {b} = {c}\n"
        toks = tokenizer.encode(text)
        # The answer token is the token encoding c.
        # In "a + b = c\n", the answer is the token(s) for c.
        # For mod 97, c is 0-96. The tokenization of "= 42\n"
        # has the number token(s) after "= ".
        # We need the LAST meaningful token before newline.
        # Strategy: encode "a + b = " as prompt, get remaining
        prompt_text = f"{a} + {b} ="
        prompt_toks = tokenizer.encode(prompt_text)
        answer_toks = toks[len(prompt_toks):]
        # answer_toks should be [space_number, newline] or similar
        # The first answer token is what we care about
        if len(answer_toks) >= 1:
            answer_token_id = answer_toks[0]
            # Input for the model: everything up to answer position
            input_toks = toks[:len(prompt_toks)]
            examples.append((input_toks, answer_token_id, (a, b, c)))

    return examples


def evaluate_sharpening(
    model: LanguageModel,
    val_examples: list[tuple[list[int], int, tuple]],
    rng: np.random.Generator,
    n_samples: int = N_SAMPLES,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Evaluate P(correct answer token) and pass@k on val set.

    Returns dict with:
        p_correct_mean: Mean P(correct) across val examples
        p_correct_median: Median P(correct)
        p_correct_std: Std of P(correct)
        pass_at_1: Fraction of examples where greedy = correct
        pass_at_64: Estimated pass@64 via sampling
        val_acc: Same as pass_at_1
        answer_entropy_mean: Mean entropy of answer distribution
    """
    p_corrects = []
    greedy_correct = 0
    sample_correct_counts = []
    entropies = []

    # Process in batches for efficiency
    batch_size = 64
    for start in range(0, len(val_examples), batch_size):
        batch = val_examples[start:start + batch_size]

        # Pad inputs to same length
        max_len = max(len(ex[0]) for ex in batch)
        padded = []
        for toks, _, _ in batch:
            padded.append(toks + [0] * (max_len - len(toks)))

        input_ids = mx.array(padded)
        logits, _ = model(input_ids)
        mx.eval(logits)

        for i, (toks, answer_id, _) in enumerate(batch):
            # Get logits at the answer position (last real token)
            pos = len(toks) - 1
            logit_vec = logits[i, pos, :]

            # Softmax to get probabilities
            probs = mx.softmax(logit_vec)
            mx.eval(probs)

            # P(correct)
            p_correct = probs[answer_id].item()
            p_corrects.append(p_correct)

            # Greedy correct?
            greedy_id = mx.argmax(logit_vec).item()
            if greedy_id == answer_id:
                greedy_correct += 1

            # Entropy
            log_probs = mx.log(probs + 1e-10)
            entropy = -mx.sum(probs * log_probs).item()
            entropies.append(entropy)

            # Sample-based pass@k
            # Draw n_samples from the distribution
            sample_logits = logit_vec / temperature
            sample_probs = mx.softmax(sample_logits)
            mx.eval(sample_probs)
            # Use numpy for multinomial sampling
            sp = np.array(sample_probs.tolist(), dtype=np.float64)
            sp = np.maximum(sp, 0)
            sp_sum = sp.sum()
            if sp_sum > 0:
                sp /= sp_sum
            else:
                sp = np.ones_like(sp) / len(sp)
            samples = rng.choice(len(sp), size=n_samples, p=sp)
            n_correct = int(np.sum(samples == answer_id))
            sample_correct_counts.append(n_correct)

    n_total = len(val_examples)
    p_arr = np.array(p_corrects)

    # pass@k estimation (unbiased estimator)
    pass_at_1 = greedy_correct / n_total
    # pass@64: fraction of examples where at least 1 of 64 is correct
    pass_at_64 = sum(
        1 for c in sample_correct_counts if c > 0
    ) / n_total

    return {
        "p_correct_mean": float(np.mean(p_arr)),
        "p_correct_median": float(np.median(p_arr)),
        "p_correct_std": float(np.std(p_arr)),
        "p_correct_p10": float(np.percentile(p_arr, 10)),
        "p_correct_p90": float(np.percentile(p_arr, 90)),
        "pass_at_1": pass_at_1,
        "pass_at_64": pass_at_64,
        "val_acc": pass_at_1,
        "answer_entropy_mean": float(np.mean(entropies)),
        "n_examples": n_total,
    }


# ── Core run function ────────────────────────────────────


def run_single(
    seed: int,
    config: ModelConfig,
    max_steps: int,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Train one seed, measure sharpening at checkpoints."""
    run_name = f"jamba_moe_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  seed={seed}, max_steps={max_steps}")

    model = LanguageModel(config)
    mx.eval(model.parameters())
    n_params = model.count_parameters()
    print(f"  params={n_params:,}")

    if dry_run:
        print("  [DRY RUN] Skipping.")
        return None

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Data
    train_ds = ModularArithmeticDataset(
        p=MODULUS, split="train", seed=42
    )
    val_ds = ModularArithmeticDataset(
        p=MODULUS, split="test", seed=42
    )

    # Build val examples for P(correct) measurement
    val_examples = build_val_examples(val_ds)
    print(f"  val_examples: {len(val_examples)}")

    # Optimizer
    warmup = optim.linear_schedule(0.0, LEARNING_RATE, WARMUP_STEPS)
    constant = optim.linear_schedule(
        LEARNING_RATE, LEARNING_RATE, max_steps
    )
    schedule = optim.join_schedules(
        [warmup, constant], [WARMUP_STEPS]
    )
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=WEIGHT_DECAY,
    )

    def loss_fn(mdl, input_ids, target_ids):
        logits, _ = mdl(input_ids)
        return nn.losses.cross_entropy(
            logits, target_ids, reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    checkpoints: list[dict[str, Any]] = []
    start_time = time.monotonic()
    step = 0
    epoch = 0

    print(f"\n  Training ({max_steps} steps)...")

    while step < max_steps:
        batches = build_example_batches(train_ds, BATCH_SIZE, rng)
        epoch += 1

        for input_ids, target_ids in batches:
            if step >= max_steps:
                break

            loss, grads = loss_and_grad(model, input_ids, target_ids)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            step += 1

            if step % 1000 == 0:
                train_loss = loss.item()
                elapsed = time.monotonic() - start_time
                sps = step / elapsed
                print(
                    f"    step {step:>6d} | "
                    f"epoch {epoch:>4d} | "
                    f"loss {train_loss:.4f} | "
                    f"{sps:.1f} steps/s"
                )

            if step % EVAL_EVERY == 0:
                train_loss = loss.item()
                elapsed = time.monotonic() - start_time

                print(f"  ** Evaluating at step {step}...")

                eval_rng = np.random.default_rng(
                    seed * 10000 + step
                )
                metrics = evaluate_sharpening(
                    model, val_examples, eval_rng
                )

                cp = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "wall_time": elapsed,
                    **metrics,
                }
                checkpoints.append(cp)

                print(
                    f"     P(correct): mean={metrics['p_correct_mean']:.4f}"
                    f" median={metrics['p_correct_median']:.4f}"
                    f" std={metrics['p_correct_std']:.4f}"
                )
                print(
                    f"     pass@1={metrics['pass_at_1']:.4f}"
                    f" pass@64={metrics['pass_at_64']:.4f}"
                    f" entropy={metrics['answer_entropy_mean']:.3f}"
                )

    total_time = time.monotonic() - start_time

    print(f"\n  {'─' * 60}")
    print(f"  Completed {step} steps in {total_time:.1f}s")
    print(f"  Known grok_step: {GROK_STEPS.get(seed, 'unknown')}")

    # Print trajectory
    hdr = (
        f"{'Step':>6} {'Loss':>7} {'P(corr)':>8} "
        f"{'p@1':>6} {'p@64':>6} {'Entropy':>8}"
    )
    print(f"\n  {hdr}")
    print(f"  {'─' * 50}")
    for cp in checkpoints:
        print(
            f"  {cp['step']:>6d} "
            f"{cp['train_loss']:>7.4f} "
            f"{cp['p_correct_mean']:>8.4f} "
            f"{cp['pass_at_1']:>6.4f} "
            f"{cp['pass_at_64']:>6.4f} "
            f"{cp['answer_entropy_mean']:>8.3f}"
        )

    return {
        "run": run_name,
        "seed": seed,
        "grok_step": GROK_STEPS.get(seed),
        "total_steps": step,
        "total_time": total_time,
        "checkpoints": checkpoints,
    }


# ── Analysis ─────────────────────────────────────────────


def analyze(results: list[dict]) -> None:
    """Analyze sharpening trajectories across seeds."""
    from scipy.stats import spearmanr

    print("\n" + "=" * 70)
    print("ANALYSIS: Answer-Token Sharpening Dynamics")
    print("=" * 70)

    # 1. Trajectory summary
    print("\n## P(correct) Trajectories")
    for r in results:
        seed = r["seed"]
        grok = r["grok_step"]
        cps = r["checkpoints"]
        vals = [cp["p_correct_mean"] for cp in cps]
        steps = [cp["step"] for cp in cps]
        print(
            f"\n  Seed {seed} (grok@{grok}):"
        )
        for s, v in zip(steps, vals):
            bar = "#" * int(v * 50)
            marker = " <-- GROK" if grok and s == (
                (grok // EVAL_EVERY) * EVAL_EVERY
            ) else ""
            print(f"    step {s:>6d}: {v:.4f} {bar}{marker}")

    # 2. Transition width analysis
    print("\n## Transition Width (10% → 90% P(correct))")
    for r in results:
        seed = r["seed"]
        cps = r["checkpoints"]
        vals = [cp["p_correct_mean"] for cp in cps]
        steps = [cp["step"] for cp in cps]

        # Find step where P(correct) first exceeds 10%
        step_10 = None
        step_90 = None
        for s, v in zip(steps, vals):
            if v >= 0.10 and step_10 is None:
                step_10 = s
            if v >= 0.90 and step_90 is None:
                step_90 = s

        if step_10 and step_90:
            width = step_90 - step_10
            print(
                f"  Seed {seed}: 10%@{step_10} → "
                f"90%@{step_90} = width {width}"
            )
        else:
            print(
                f"  Seed {seed}: 10%@{step_10} → "
                f"90%@{step_90} (incomplete)"
            )

    # 3. Oscillation detection
    print("\n## Oscillation Check")
    for r in results:
        seed = r["seed"]
        cps = r["checkpoints"]
        vals = [cp["p_correct_mean"] for cp in cps]

        # Count direction changes
        if len(vals) < 3:
            continue
        changes = 0
        for i in range(2, len(vals)):
            d1 = vals[i - 1] - vals[i - 2]
            d2 = vals[i] - vals[i - 1]
            if d1 * d2 < 0 and abs(d1) > 0.01 and abs(d2) > 0.01:
                changes += 1
        print(
            f"  Seed {seed}: {changes} direction changes "
            f"(>0.01 magnitude)"
        )

    # 4. Correlation: sharpening rate vs grok_step
    print("\n## Sharpening Rate vs Grok Step")
    rates = []
    grok_steps = []
    for r in results:
        seed = r["seed"]
        grok = r["grok_step"]
        if grok is None or grok > MAX_STEPS:
            continue
        cps = r["checkpoints"]
        vals = [cp["p_correct_mean"] for cp in cps]
        steps = [cp["step"] for cp in cps]

        # Rate = max derivative of P(correct) between checkpoints
        max_rate = 0
        for i in range(1, len(vals)):
            rate = (vals[i] - vals[i - 1]) / (
                steps[i] - steps[i - 1]
            )
            max_rate = max(max_rate, rate)

        rates.append(max_rate)
        grok_steps.append(grok)
        print(
            f"  Seed {seed}: max_rate={max_rate:.6f}/step, "
            f"grok@{grok}"
        )

    if len(rates) >= 3:
        rho, p = spearmanr(rates, grok_steps)
        print(
            f"\n  Spearman(max_rate, grok_step): "
            f"rho={rho:.3f}, p={p:.3f}"
        )

    # 5. Hypothesis adjudication
    print("\n## Hypothesis Adjudication")
    # Collect transition widths
    widths = []
    for r in results:
        cps = r["checkpoints"]
        vals = [cp["p_correct_mean"] for cp in cps]
        steps = [cp["step"] for cp in cps]
        step_10 = step_90 = None
        for s, v in zip(steps, vals):
            if v >= 0.10 and step_10 is None:
                step_10 = s
            if v >= 0.90 and step_90 is None:
                step_90 = s
        if step_10 is not None and step_90 is not None:
            widths.append(step_90 - step_10)

    if widths:
        mean_width = np.mean(widths)
        print(f"  Mean transition width: {mean_width:.0f} steps")
        print(f"  Individual widths: {widths}")

        if mean_width < 5000:
            print("  → H38-b SUPPORTED (phase transition)")
        elif mean_width > 10000:
            print("  → H38-a SUPPORTED (gradual sharpening)")
        else:
            print("  → INCONCLUSIVE (intermediate width)")


# ── Main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="HYP-038: Sharpening dynamics"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config only",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run 1 seed, 4K steps",
    )
    parser.add_argument(
        "--max-runs", type=int, default=len(SEEDS),
        help="Max seeds to run",
    )
    args = parser.parse_args()

    max_steps = MAX_STEPS
    seeds = SEEDS[:args.max_runs]

    if args.pilot:
        seeds = [SEEDS[0]]
        max_steps = 4000

    print("HYP-038: Answer-Token Sharpening Dynamics")
    print(f"  Seeds: {seeds}")
    print(f"  Max steps: {max_steps}")
    print(f"  Eval every: {EVAL_EVERY}")
    print(f"  Config: {CONFIG.n_layers}L, d={CONFIG.block.d_model}")

    results = []
    for seed in seeds:
        result = run_single(
            seed=seed,
            config=CONFIG,
            max_steps=max_steps,
            dry_run=args.dry_run,
        )
        if result is not None:
            results.append(result)

    if results and not args.dry_run:
        # Save results
        out_path = Path("experiments/hyp038_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

        # Run analysis
        analyze(results)


if __name__ == "__main__":
    main()
