"""HYP-016: Can early training signals predict grokking onset?

Pre-registered hypothesis:
    B-014 showed pass@64 at step 2K predicts grokking ORDER across
    4 architectures (rank correlation = 1.0). B-015 showed grokking
    onset varies 10x across seeds (4K-40K). Can early TTC signal
    predict WHICH SEEDS will grok within a single architecture?

    If pass@64 is truly an early indicator of grokking propensity,
    it should predict within-architecture seed variation, not just
    between-architecture differences. This moves from a confounded
    correlation (architecture -> both) to a clean test.

Competing hypotheses:
    H16-a (TTC predicts grokking): Spearman rank correlation
        between pass@64 at step 2K and grok_step is >= 0.6 (or
        pass@64 separates grokking from non-grokking seeds).
    H16-b (Loss predicts better): val_loss at step 2K is a
        stronger predictor than pass@64 (higher rank correlation
        with grok_step).
    H16-c (No early signal): Neither metric at step 2K has
        Spearman >= 0.4 with grok_step. Grokking onset is
        essentially random given fixed architecture.

Design:
    10 seeds x 1 architecture = 10 runs:
    - MoE-Jamba (7.6M params), seeds 42..51
    - 50K steps, eval every 2K steps
    - Same training setup as HYP-015

    Key metrics at each checkpoint:
    - val_accuracy (greedy)
    - pass@1 through pass@64
    - train_loss

    Analysis:
    - Compute Spearman correlation between early metrics (step 2K)
      and grokking onset step
    - Seeds that never grok get grok_step = 60K (censored)
    - ROC analysis: can pass@64 at step 2K separate grokkers
      from non-grokkers?

Usage:
    uv run python recipes/hyp016_early_grokking_prediction.py
    uv run python recipes/hyp016_early_grokking_prediction.py --dry-run
    uv run python recipes/hyp016_early_grokking_prediction.py --pilot
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
from lmxlab.eval.metrics import pass_at_k
from lmxlab.models.base import LanguageModel
from lmxlab.models.jamba import jamba_config

# ── Constants ────────────────────────────────────────────

SEEDS = list(range(42, 52))  # 10 seeds: 42..51
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
N_SAMPLES = 64
MODULUS = 97
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
BATCH_SIZE = 64
TEMPERATURE = 0.8
MAX_STEPS = 50_000
EVAL_INTERVAL = 2_000
WARMUP_STEPS = 100

# ── Jamba MoE config (same as HYP-015) ───────────────────

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
    """Build Jamba MoE config (same as HYP-015 jamba_moe)."""
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


# ── Data helpers (reused from HYP-015) ──────────────────


def build_example_batches(
    dataset: ModularArithmeticDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[mx.array, mx.array]]:
    """Build padded batches of complete examples.

    Each example is "a + b = c\\n" with next-token prediction.

    Args:
        dataset: ModularArithmeticDataset instance.
        batch_size: Examples per batch.
        rng: NumPy random generator for shuffling.

    Returns:
        List of (input_ids, target_ids) tuples.
    """
    from lmxlab.data.tokenizer import TiktokenTokenizer

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


# ── Evaluation ───────────────────────────────────────────


def evaluate_pass_at_k_modular(
    model: LanguageModel,
    dataset: ModularArithmeticDataset,
    k_values: list[int],
    n_samples: int,
    temperature: float,
) -> dict[str, float]:
    """Evaluate pass@k on modular arithmetic test set.

    Args:
        model: Trained language model.
        dataset: Test split of ModularArithmeticDataset.
        k_values: Values of k for pass@k.
        n_samples: Number of samples per prompt (N).
        temperature: Sampling temperature.

    Returns:
        Dict with pass@k scores and val_accuracy.
    """
    model.eval()
    prompts = dataset.get_prompts()
    answer_ids = dataset.answer_token_ids

    results: dict[str, list[float]] = {f"pass_at_{k}": [] for k in k_values}
    correct_greedy = 0
    total = 0

    for prompt_tokens, answer in prompts:
        prompt_batch = mx.broadcast_to(
            prompt_tokens[None, :],
            (n_samples, prompt_tokens.shape[0]),
        )
        logits, _ = model(prompt_batch)
        mx.eval(logits)

        next_logits = logits[:, -1, :]

        greedy_pred = mx.argmax(next_logits[0]).item()
        correct_id = answer_ids[answer]
        if greedy_pred == correct_id:
            correct_greedy += 1
        total += 1

        scaled_logits = next_logits / temperature
        sampled = mx.random.categorical(scaled_logits)
        mx.eval(sampled)

        sampled_list = sampled.tolist()
        c = sum(1 for s in sampled_list if s == correct_id)

        for k in k_values:
            if k <= n_samples:
                score = pass_at_k(n_samples, c, k)
                results[f"pass_at_{k}"].append(score)

    model.train()

    out = {
        key: sum(vals) / len(vals) if vals else 0.0
        for key, vals in results.items()
    }
    out["val_accuracy"] = correct_greedy / total if total else 0.0
    return out


# ── Core run function ────────────────────────────────────


def run_single(
    seed: int,
    config: ModelConfig,
    max_steps: int,
    eval_interval: int,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Train one seed, track grokking + TTC trajectory.

    Args:
        seed: Random seed.
        config: Model configuration.
        max_steps: Maximum training steps.
        eval_interval: Steps between pass@k evaluations.
        dry_run: If True, skip actual training.

    Returns:
        Result dict with checkpoint data, or None.
    """
    run_name = f"jamba_moe_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  seed={seed}")

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
    train_ds = ModularArithmeticDataset(p=MODULUS, split="train", seed=42)
    test_ds = ModularArithmeticDataset(p=MODULUS, split="test", seed=42)

    n_train = train_ds.num_examples
    steps_per_epoch = n_train // BATCH_SIZE
    print(f"  train_examples={n_train}")
    print(f"  steps_per_epoch~={steps_per_epoch}")

    # Optimizer
    warmup = optim.linear_schedule(0.0, LEARNING_RATE, WARMUP_STEPS)
    constant = optim.linear_schedule(LEARNING_RATE, LEARNING_RATE, max_steps)
    schedule = optim.join_schedules([warmup, constant], [WARMUP_STEPS])
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=WEIGHT_DECAY,
    )

    def loss_fn(model, input_ids, target_ids):
        logits, _ = model(input_ids)
        return nn.losses.cross_entropy(logits, target_ids, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    checkpoints: list[dict[str, Any]] = []
    start_time = time.monotonic()
    step = 0
    grokked = False
    grok_step = -1
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

            if step % eval_interval == 0:
                elapsed = time.monotonic() - start_time
                train_loss = loss.item()

                pk = evaluate_pass_at_k_modular(
                    model=model,
                    dataset=test_ds,
                    k_values=K_VALUES,
                    n_samples=N_SAMPLES,
                    temperature=TEMPERATURE,
                )

                val_acc = pk.pop("val_accuracy")

                checkpoint = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_accuracy": val_acc,
                    "wall_time": elapsed,
                    **pk,
                }
                checkpoints.append(checkpoint)

                p1 = pk.get("pass_at_1", 0)
                p64 = pk.get("pass_at_64", 0)
                ratio = p64 / p1 if p1 > 0 else 0

                print(
                    f"  ** CP step={step} "
                    f"epoch={epoch} | "
                    f"val_acc={val_acc:.3f} | "
                    f"p@1={p1:.4f} | "
                    f"p@64={p64:.4f} | "
                    f"p@64/p@1={ratio:.1f}x"
                )

                if val_acc > 0.95 and not grokked:
                    grokked = True
                    grok_step = step
                    print(
                        f"  ** GROKKED at step {step}! val_acc={val_acc:.4f}"
                    )

    total_time = time.monotonic() - start_time

    print(f"\n  {'─' * 50}")
    print(f"  Completed {step} steps in {total_time:.1f}s")
    print(f"  Grokked: {grokked} (step {grok_step})")

    # Print trajectory
    print(f"\n  {'Step':>6} {'Epoch':>5} {'VAcc':>6} {'p@1':>7} {'p@64':>7}")
    print(f"  {'─' * 38}")
    for cp in checkpoints:
        p1 = cp.get("pass_at_1", 0)
        p64 = cp.get("pass_at_64", 0)
        print(
            f"  {cp['step']:>6d} "
            f"{cp['epoch']:>5d} "
            f"{cp['val_accuracy']:>6.3f} "
            f"{p1:>7.4f} "
            f"{p64:>7.4f}"
        )

    return {
        "run": run_name,
        "seed": seed,
        "params": n_params,
        "max_steps": step,
        "epochs": epoch,
        "wall_time": total_time,
        "grokked": grokked,
        "grok_step": grok_step,
        "checkpoints": checkpoints,
    }


# ── Analysis ─────────────────────────────────────────────


def analyze_results(results: list[dict[str, Any]]) -> None:
    """Analyze early signal prediction of grokking onset."""
    print(f"\n{'=' * 70}")
    print("HYP-016: Early Grokking Prediction Analysis")
    print(f"{'=' * 70}")

    # Summary table
    header = (
        f"{'Seed':>4} {'Params':>8} {'Grokked':>7} "
        f"{'GrokStep':>8} {'p@64@2K':>8} "
        f"{'VAcc@2K':>8} {'Loss@2K':>8}"
    )
    print(header)
    print("-" * len(header))

    early_data = []
    for r in results:
        grok_str = "Yes" if r["grokked"] else "No"
        grok_step = str(r["grok_step"]) if r["grokked"] else ">50K"

        # Get step 2K checkpoint
        cp_2k = None
        for cp in r["checkpoints"]:
            if cp["step"] == 2000:
                cp_2k = cp
                break

        p64 = cp_2k["pass_at_64"] if cp_2k else 0
        vacc = cp_2k["val_accuracy"] if cp_2k else 0
        tloss = cp_2k["train_loss"] if cp_2k else 0

        print(
            f"{r['seed']:>4} "
            f"{r['params']:>8,} "
            f"{grok_str:>7} "
            f"{grok_step:>8} "
            f"{p64:>8.4f} "
            f"{vacc:>8.4f} "
            f"{tloss:>8.4f}"
        )

        # Use censored value for non-grokkers
        effective_grok = r["grok_step"] if r["grokked"] else 60_000
        early_data.append(
            {
                "seed": r["seed"],
                "grokked": r["grokked"],
                "grok_step": effective_grok,
                "p64_2k": p64,
                "vacc_2k": vacc,
                "loss_2k": tloss,
            }
        )

    if len(early_data) < 3:
        print("\nToo few runs for correlation analysis.")
        return

    # Spearman rank correlation
    from scipy.stats import spearmanr  # type: ignore

    grok_steps = [d["grok_step"] for d in early_data]
    p64_vals = [d["p64_2k"] for d in early_data]
    loss_vals = [d["loss_2k"] for d in early_data]
    vacc_vals = [d["vacc_2k"] for d in early_data]

    # Higher p@64 should predict EARLIER grokking (lower step)
    # So we expect negative correlation with grok_step
    rho_p64, p_p64 = spearmanr(p64_vals, grok_steps)
    rho_loss, p_loss = spearmanr(loss_vals, grok_steps)
    rho_vacc, p_vacc = spearmanr(vacc_vals, grok_steps)

    print(f"\n{'─' * 50}")
    print("Spearman correlations (early metric vs grok_step):")
    print(f"  p@64 at 2K:    rho={rho_p64:.3f}, p={p_p64:.4f}")
    print(f"  val_loss at 2K: rho={rho_loss:.3f}, p={p_loss:.4f}")
    print(f"  val_acc at 2K: rho={rho_vacc:.3f}, p={p_vacc:.4f}")

    # Grokker vs non-grokker separation
    grokkers = [d for d in early_data if d["grokked"]]
    nongrokkers = [d for d in early_data if not d["grokked"]]

    print(f"\n{'─' * 50}")
    print("Grokker vs Non-grokker separation:")
    print(f"  Grokkers: {len(grokkers)}, Non-grokkers: {len(nongrokkers)}")

    if grokkers and nongrokkers:
        avg_p64_grok = sum(d["p64_2k"] for d in grokkers) / len(grokkers)
        avg_p64_nongrok = sum(d["p64_2k"] for d in nongrokkers) / len(
            nongrokkers
        )
        avg_loss_grok = sum(d["loss_2k"] for d in grokkers) / len(grokkers)
        avg_loss_nongrok = sum(d["loss_2k"] for d in nongrokkers) / len(
            nongrokkers
        )
        print(
            f"  p@64 at 2K: grokkers={avg_p64_grok:.4f}, "
            f"non-grokkers={avg_p64_nongrok:.4f}"
        )
        print(
            f"  loss at 2K: grokkers={avg_loss_grok:.4f}, "
            f"non-grokkers={avg_loss_nongrok:.4f}"
        )

    # Adjudication
    print(f"\n{'─' * 50}")
    print("Hypothesis adjudication:")
    abs_rho_p64 = abs(rho_p64) if not np.isnan(rho_p64) else 0
    abs_rho_loss = abs(rho_loss) if not np.isnan(rho_loss) else 0

    if abs_rho_p64 >= 0.6:
        print(
            "  H16-a (TTC predicts): SUPPORTED "
            f"(|rho|={abs_rho_p64:.3f} >= 0.6)"
        )
    elif abs_rho_p64 >= 0.4:
        print(
            "  H16-a (TTC predicts): PARTIALLY SUPPORTED "
            f"(|rho|={abs_rho_p64:.3f}, 0.4-0.6)"
        )
    else:
        print(
            "  H16-a (TTC predicts): NOT SUPPORTED "
            f"(|rho|={abs_rho_p64:.3f} < 0.4)"
        )

    if abs_rho_loss > abs_rho_p64 and abs_rho_loss >= 0.4:
        print(
            "  H16-b (Loss better): SUPPORTED "
            f"(|rho_loss|={abs_rho_loss:.3f} > "
            f"|rho_p64|={abs_rho_p64:.3f})"
        )
    else:
        print(
            "  H16-b (Loss better): NOT SUPPORTED "
            f"(|rho_loss|={abs_rho_loss:.3f})"
        )

    if abs_rho_p64 < 0.4 and abs_rho_loss < 0.4:
        print("  H16-c (No early signal): SUPPORTED (both |rho| < 0.4)")
    else:
        print("  H16-c (No early signal): NOT SUPPORTED")


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-016 early grokking prediction experiment."""
    parser = argparse.ArgumentParser(
        description="HYP-016: Early grokking prediction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Quick single run (seed 45, 10K steps)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Max training steps (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit number of runs",
    )
    args = parser.parse_args()

    if args.pilot:
        # Use seed 45 (not in HYP-015) for a fresh pilot
        print("PILOT: jamba_moe, seed=45, 10K steps")
        result = run_single(
            seed=45,
            config=CONFIG,
            max_steps=10000,
            eval_interval=2000,
        )
        if result:
            print(
                f"\nPilot: grokked={result['grokked']}, "
                f"grok_step={result['grok_step']}"
            )
        return

    # Full experiment: 10 seeds
    seeds = SEEDS[: args.max_runs] if args.max_runs else SEEDS

    print("HYP-016: Early Grokking Prediction")
    print(f"Seeds: {seeds}")
    print(f"Max steps: {args.max_steps}")

    results: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i + 1}/{len(seeds)}]", end="")
        result = run_single(
            seed=seed,
            config=CONFIG,
            max_steps=args.max_steps,
            eval_interval=EVAL_INTERVAL,
            dry_run=args.dry_run,
        )
        if result:
            results.append(result)

    if not results:
        return

    analyze_results(results)

    out = Path("experiments") / "hyp016_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
