"""HYP-014: Grokking dynamics across architectures.

Pre-registered hypothesis:
    Do different architecture families (LLaMA, Falcon-H1, Jamba,
    Bamba) grok modular arithmetic at different rates? Does
    TTC (pass@64) reveal latent generalization earlier in some
    architectures than others?

Competing hypotheses:
    H14-a (SSM advantage): SSM-containing architectures
        (Falcon-H1, Jamba, Bamba) grok faster than pure
        attention (LLaMA) because SSM rotational dynamics
        provide an inductive bias for cyclic group structure.
    H14-b (Attention advantage): Pure attention (LLaMA)
        groks faster because attention can implement the
        Fourier circuit (Nanda et al.) directly.
    H14-c (Architecture-independent): All architectures
        grok at similar rates — grokking dynamics depend on
        optimization (wd, lr) not architecture.
    H14-d (Hybrid advantage): Hybrids grok faster than
        both pure attention and pure SSM because they
        combine the SSM's rotational bias with attention's
        compositionality.

Design:
    4 architectures (LLaMA, Falcon-H1, Jamba, Bamba)
    x 1 seed (42 — the seed that grokked in HYP-009)
    = 4 runs with per-example training
    50K max steps, eval every 2K steps
    wd=0.1, lr=1e-3, constant LR, batch_size=64
    Small models (~7M params each) with d=128, tied embeddings
    Grokking detection: val_accuracy > 0.95

Usage:
    uv run python recipes/hyp014_grokking_architectures.py
    uv run python recipes/hyp014_grokking_architectures.py --dry-run
    uv run python recipes/hyp014_grokking_architectures.py --pilot
    uv run python recipes/hyp014_grokking_architectures.py --max-steps 10000
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

from lmxlab.data.modular_arithmetic import ModularArithmeticDataset
from lmxlab.eval.metrics import pass_at_k
from lmxlab.models.bamba import bamba_config
from lmxlab.models.base import LanguageModel, ModelConfig
from lmxlab.models.falcon import falcon_h1_config
from lmxlab.models.jamba import jamba_config
from lmxlab.models.llama import llama_config

# ── Constants ────────────────────────────────────────────

SEED = 42
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

# ── Grokking-scale model configs ─────────────────────────

# All models: d=128, BPE vocab (50257), tied embeddings,
# max_seq_len=64. Target ~7M params (dominated by embedding).
# Each hybrid has 1 SSM + 1 attention layer minimum.

GROK_COMMON = dict(
    vocab_size=50257,
    d_model=128,
    n_heads=4,
    n_kv_heads=2,
    d_ff=512,
    max_seq_len=64,
    tie_embeddings=True,
)

GROK_MAMBA_COMMON = dict(
    mamba_n_heads=8,
    mamba_head_dim=32,
    ssm_state_size=16,
    mamba_expand=2,
    mamba_n_groups=1,
    mamba_chunk_size=64,
    conv_kernel_size=4,
)


def _arch_configs() -> dict[str, ModelConfig]:
    """Build grokking-scale configs for all 4 architectures."""
    return {
        "llama": llama_config(
            n_layers=2,
            **GROK_COMMON,
        ),
        "falcon_h1": falcon_h1_config(
            hybrid_pattern="M*",
            rope_theta=10000.0,
            **GROK_COMMON,
            **GROK_MAMBA_COMMON,
        ),
        "jamba": jamba_config(
            n_layers=2,
            attn_every=2,
            n_experts=4,
            top_k_experts=2,
            moe_every=2,
            rope_theta=10000.0,
            **GROK_COMMON,
            **GROK_MAMBA_COMMON,
        ),
        "bamba": bamba_config(
            hybrid_pattern="M*",
            rope_theta=10000.0,
            **GROK_COMMON,
            **GROK_MAMBA_COMMON,
        ),
    }


ARCHITECTURES = _arch_configs()


# ── Data helpers ─────────────────────────────────────────


def build_example_batches(
    dataset: ModularArithmeticDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[mx.array, mx.array]]:
    """Build padded batches of complete examples.

    Each example is the full tokenized "a + b = c\\n" with
    next-token prediction on ALL tokens.

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
    arch_name: str,
    config: ModelConfig,
    max_steps: int,
    eval_interval: int,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Train one architecture, track grokking + pass@k.

    Args:
        arch_name: Architecture name (for logging).
        config: Model configuration.
        max_steps: Maximum training steps.
        eval_interval: Steps between pass@k evaluations.
        dry_run: If True, skip actual training.

    Returns:
        Result dict with checkpoint data, or None.
    """
    run_name = f"{arch_name}_s{SEED}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  arch={arch_name}, seed={SEED}")
    print(f"  wd={WEIGHT_DECAY}, lr={LEARNING_RATE}")
    print(f"  max_steps={max_steps}, eval_interval={eval_interval}")

    model = LanguageModel(config)
    mx.eval(model.parameters())
    n_params = model.count_parameters()
    print(f"  params={n_params:,}")

    if dry_run:
        print("  [DRY RUN] Skipping.")
        return None

    mx.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Data
    train_ds = ModularArithmeticDataset(p=MODULUS, split="train", seed=42)
    test_ds = ModularArithmeticDataset(p=MODULUS, split="test", seed=42)

    n_train = train_ds.num_examples
    steps_per_epoch = n_train // BATCH_SIZE
    print(f"  train_examples={n_train}")
    print(f"  steps_per_epoch~={steps_per_epoch}")
    est_epochs = max_steps / max(steps_per_epoch, 1)
    print(f"  total_epochs~={est_epochs:.0f}")

    # Optimizer — constant LR + weight decay
    warmup_schedule = optim.linear_schedule(0.0, LEARNING_RATE, WARMUP_STEPS)
    constant_schedule = optim.linear_schedule(
        LEARNING_RATE, LEARNING_RATE, max_steps
    )
    schedule = optim.join_schedules(
        [warmup_schedule, constant_schedule], [WARMUP_STEPS]
    )
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=WEIGHT_DECAY,
    )

    # Loss
    def loss_fn(model, input_ids, target_ids):
        logits, _ = model(input_ids)
        return nn.losses.cross_entropy(logits, target_ids, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # ── Training loop with checkpoint evals ──
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

            # Periodic logging
            if step % 1000 == 0:
                train_loss = loss.item()
                elapsed = time.monotonic() - start_time
                steps_per_sec = step / elapsed
                print(
                    f"    step {step:>6d} | "
                    f"epoch {epoch:>4d} | "
                    f"loss {train_loss:.4f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

            # Checkpoint evaluation
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
                    f"  ** CHECKPOINT step={step} "
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

                # Early stop: 3 post-grok checkpoints
                if grokked:
                    post_grok = [
                        c for c in checkpoints if c["val_accuracy"] > 0.95
                    ]
                    if len(post_grok) >= 3:
                        print("  ** 3+ post-grok checkpoints. Stopping.")
                        step = max_steps
                        break

    total_time = time.monotonic() - start_time

    # Summary
    print(f"\n  {'─' * 50}")
    print(f"  Completed {step} steps in {total_time:.1f}s")
    print(f"  Epochs: {epoch}")
    print(f"  Grokked: {grokked}")
    if grokked:
        print(f"  Grokking step: {grok_step}")

    print(
        f"\n  {'Step':>6} {'Epoch':>5} {'VAcc':>6} "
        f"{'p@1':>7} {'p@64':>7} {'p64/p1':>7}"
    )
    print(f"  {'─' * 45}")
    for cp in checkpoints:
        p1 = cp.get("pass_at_1", 0)
        p64 = cp.get("pass_at_64", 0)
        ratio = p64 / p1 if p1 > 0 else 0
        print(
            f"  {cp['step']:>6d} "
            f"{cp['epoch']:>5d} "
            f"{cp['val_accuracy']:>6.3f} "
            f"{p1:>7.4f} "
            f"{p64:>7.4f} "
            f"{ratio:>7.1f}x"
        )

    return {
        "run": run_name,
        "arch": arch_name,
        "seed": SEED,
        "params": n_params,
        "max_steps": step,
        "epochs": epoch,
        "wall_time": total_time,
        "grokked": grokked,
        "grok_step": grok_step,
        "checkpoints": checkpoints,
    }


def run_pilot() -> None:
    """Quick run: LLaMA only, 5K steps."""
    print("=" * 60)
    print("PILOT RUN: LLaMA, 5K steps")
    print("=" * 60)
    config = ARCHITECTURES["llama"]
    result = run_single(
        arch_name="llama",
        config=config,
        max_steps=5000,
        eval_interval=1000,
    )
    if result:
        n_cp = len(result["checkpoints"])
        print(f"\nPilot: {n_cp} checkpoints collected")


# ── Analysis ─────────────────────────────────────────────


def analyze_results(results: list[dict[str, Any]]) -> None:
    """Print cross-architecture grokking comparison."""
    print(f"\n{'=' * 70}")
    print("Cross-Architecture Grokking Comparison")
    print(f"{'=' * 70}")

    header = (
        f"{'Arch':<12} {'Params':>8} {'Grokked':>8} "
        f"{'Grok Step':>10} {'Time':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        grok_str = "Yes" if r["grokked"] else "No"
        grok_step = str(r["grok_step"]) if r["grokked"] else "N/A"
        print(
            f"{r['arch']:<12} "
            f"{r['params']:>8,} "
            f"{grok_str:>8} "
            f"{grok_step:>10} "
            f"{r['wall_time']:>7.0f}s"
        )

    # TTC signature comparison at specific checkpoints
    print(f"\n{'=' * 70}")
    print("TTC Signature at Early Checkpoints")
    print(f"{'=' * 70}")

    # Find the first checkpoint where any arch has p@64 > 0.5
    early_steps = [2000, 4000, 6000, 10000]
    for target_step in early_steps:
        print(f"\n  Step {target_step}:")
        for r in results:
            cp = None
            for c in r["checkpoints"]:
                if c["step"] == target_step:
                    cp = c
                    break
            if cp:
                p1 = cp.get("pass_at_1", 0)
                p64 = cp.get("pass_at_64", 0)
                ratio = p64 / p1 if p1 > 0 else 0
                print(
                    f"    {r['arch']:<12} "
                    f"val_acc={cp['val_accuracy']:.3f} "
                    f"p@1={p1:.4f} "
                    f"p@64={p64:.4f} "
                    f"p64/p1={ratio:.1f}x"
                )
            else:
                print(f"    {r['arch']:<12} (no checkpoint)")


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-014 grokking comparison."""
    parser = argparse.ArgumentParser(
        description=("HYP-014: Grokking dynamics across architectures"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Quick LLaMA-only 5K-step run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Max training steps (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL,
        help=f"Steps between evals (default: {EVAL_INTERVAL})",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit number of architecture runs",
    )
    args = parser.parse_args()

    if args.pilot:
        run_pilot()
        return

    arch_names = list(ARCHITECTURES.keys())
    if args.max_runs:
        arch_names = arch_names[: args.max_runs]

    print("HYP-014: Grokking Dynamics Across Architectures")
    print(f"Architectures: {arch_names}")
    print(f"Seed: {SEED}")
    print(f"Max steps: {args.max_steps}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"LR: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"K values: {K_VALUES}")
    print(f"N samples: {N_SAMPLES}")
    print(f"Temperature: {TEMPERATURE}")

    results: list[dict[str, Any]] = []
    for i, arch_name in enumerate(arch_names):
        print(f"\n[{i + 1}/{len(arch_names)}]", end="")
        config = ARCHITECTURES[arch_name]
        result = run_single(
            arch_name=arch_name,
            config=config,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            dry_run=args.dry_run,
        )
        if result:
            results.append(result)

    if not results:
        return

    # Analysis
    analyze_results(results)

    # Save
    out = Path("experiments") / "hyp014_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
