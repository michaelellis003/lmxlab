"""HYP-009: Grokking x Test-Time Compute interaction.

Pre-registered hypothesis:
    How does test-time compute effectiveness (pass@k curves)
    change as a model transitions from memorization to
    generalization (grokking) on modular arithmetic?

Competing hypotheses:
    H9-a (Early indicator): pass@64 improves before val
        accuracy jumps at grokking
    H9-b (Simultaneous): pass@k jumps with val accuracy
    H9-c (Diversity peak): p@64/p@1 peaks pre-grok
    H9-d (Post-grok explosion): pass@64 jumps >10x at grok

Design:
    LLaMA-grok (~7M params: d=128, 2 layers, BPE vocab)
    Per-example training with answer-token-only loss
    dropout=0.0, weight_decay=1.0, constant LR=1e-3
    3 seeds (42, 43, 44)
    50K steps max, eval every 1K steps
    pass@k at each checkpoint

    Uses per-example training (not token-stream) to match
    the grokking literature setup. Each batch contains
    complete "a + b = c" examples. Loss computed on the
    answer token only.

Usage:
    uv run python recipes/hyp009_grokking_ttc.py
    uv run python recipes/hyp009_grokking_ttc.py --dry-run
    uv run python recipes/hyp009_grokking_ttc.py --pilot
    uv run python recipes/hyp009_grokking_ttc.py --max-runs 1
    uv run python recipes/hyp009_grokking_ttc.py --max-steps 10000
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
from lmxlab.models.base import LanguageModel
from lmxlab.models.llama import llama_config

# ── Constants ────────────────────────────────────────────

SEEDS = [42, 43, 44]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
N_SAMPLES = 64
MODULUS = 97
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
BATCH_SIZE = 64
TEMPERATURE = 0.8
MAX_STEPS = 50_000
EVAL_INTERVAL = 1_000
WARMUP_STEPS = 100


def llama_grok():
    """Small LLaMA for grokking experiments.

    d=128, 2 layers, 4 heads, BPE vocab, tied embeddings.
    ~7M params (6.4M in embedding, ~500K in transformer).
    """
    return llama_config(
        vocab_size=50257,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        tie_embeddings=True,
    )


# ── Data helpers ─────────────────────────────────────────


def build_example_batches(
    dataset: ModularArithmeticDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[mx.array, mx.array]]:
    """Build padded batches of complete examples.

    Each example is the full tokenized "a + b = c\\n" with
    next-token prediction on ALL tokens (not just answer).
    This matches the grokking literature setup.

    Args:
        dataset: ModularArithmeticDataset instance.
        batch_size: Examples per batch.
        rng: NumPy random generator for shuffling.

    Returns:
        List of (input_ids, target_ids) tuples.
        input_ids: (batch, seq_len) — tokens [:-1]
        target_ids: (batch, seq_len) — tokens [1:]
    """
    from lmxlab.data.tokenizer import TiktokenTokenizer

    tokenizer = TiktokenTokenizer("gpt2")
    pairs = dataset._pairs  # noqa: SLF001

    # Tokenize all examples
    all_seqs: list[list[int]] = []
    for a, b, c in pairs:
        text = f"{a} + {b} = {c}\n"
        toks = tokenizer.encode(text)
        all_seqs.append(toks)

    # Shuffle
    indices = rng.permutation(len(all_seqs))

    # Batch with padding
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < 2:
            continue

        seqs = [all_seqs[i] for i in batch_idx]
        max_len = max(len(s) for s in seqs)

        # Pad and split into input/target
        inputs = []
        targets = []
        for s in seqs:
            padded = s + [0] * (max_len - len(s))
            inputs.append(padded[:-1])
            targets.append(padded[1:])

        batches.append((mx.array(inputs), mx.array(targets)))

    return batches


# ── Core functions ───────────────────────────────────────


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

        # Greedy accuracy
        greedy_pred = mx.argmax(next_logits[0]).item()
        correct_id = answer_ids[answer]
        if greedy_pred == correct_id:
            correct_greedy += 1
        total += 1

        # Sampled pass@k
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


def run_single(
    seed: int,
    max_steps: int,
    eval_interval: int,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Train one model, evaluate pass@k at checkpoints.

    Uses per-example training with answer-token-only loss
    to match the grokking literature setup.

    Args:
        seed: Random seed.
        max_steps: Maximum training steps.
        eval_interval: Steps between pass@k evaluations.
        dry_run: If True, skip actual training.

    Returns:
        Result dict with checkpoint data, or None.
    """
    run_name = f"llama_grok_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  seed={seed}, wd={WEIGHT_DECAY}, lr={LEARNING_RATE}")
    print(f"  max_steps={max_steps}, eval_interval={eval_interval}")

    if dry_run:
        config = llama_grok()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        n_params = model.count_parameters()
        print(f"  params={n_params:,}")
        print("  [DRY RUN] Skipping.")
        return None

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Model — small LLaMA for grokking
    model = LanguageModel(llama_grok())
    mx.eval(model.parameters())
    n_params = model.count_parameters()
    print(f"  params={n_params:,}")

    # Data
    train_ds = ModularArithmeticDataset(p=MODULUS, split="train", seed=42)
    test_ds = ModularArithmeticDataset(p=MODULUS, split="test", seed=42)

    n_train = train_ds.num_examples
    n_test = test_ds.num_examples
    steps_per_epoch = n_train // BATCH_SIZE
    print(f"  train_examples={n_train}")
    print(f"  test_examples={n_test}")
    print(f"  batch_size={BATCH_SIZE}")
    print(f"  steps_per_epoch~={steps_per_epoch}")
    est_epochs = max_steps / max(steps_per_epoch, 1)
    print(f"  total_epochs~={est_epochs:.0f}")

    # Optimizer — constant LR + high weight decay
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

    # Loss: full-sequence next-token prediction
    def loss_fn(model, input_ids, target_ids):
        logits, _ = model(input_ids)
        return nn.losses.cross_entropy(logits, target_ids, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # ── Training loop with checkpoint evals ──
    checkpoints: list[dict[str, Any]] = []
    start_time = time.monotonic()
    step = 0
    grokked = False
    epoch = 0

    print(f"\n  Training ({max_steps} steps)...")

    while step < max_steps:
        # Build shuffled batches for this epoch
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
            if step % 500 == 0:
                train_loss = loss.item()
                elapsed = time.monotonic() - start_time
                steps_per_sec = step / elapsed
                print(
                    f"    step {step:>6d} | "
                    f"epoch {epoch:>4d} | "
                    f"train_loss {train_loss:.4f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

            # Checkpoint evaluation
            if step % eval_interval == 0:
                elapsed = time.monotonic() - start_time

                # Compute train accuracy on a sample
                train_loss = loss.item()

                # pass@k on test set
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

                # Grokking detection
                if val_acc > 0.95 and not grokked:
                    grokked = True
                    print(
                        f"  ** GROKKED at step {step}! val_acc={val_acc:.4f}"
                    )

                # Early stop: 3+ post-grok checkpoints
                if grokked:
                    post_grok = [
                        c for c in checkpoints if c["val_accuracy"] > 0.95
                    ]
                    if len(post_grok) >= 3:
                        print("  ** 3+ post-grok checkpoints. Stopping.")
                        step = max_steps
                        break

    total_time = time.monotonic() - start_time

    # ── Summary ──
    print(f"\n  {'─' * 50}")
    print(f"  Completed {step} steps in {total_time:.1f}s")
    print(f"  Epochs: {epoch}")
    print(f"  Grokked: {grokked}")
    if grokked:
        grok_step = next(
            c["step"] for c in checkpoints if c["val_accuracy"] > 0.95
        )
        print(f"  Grokking step: {grok_step}")

    # Checkpoint summary table
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
        "seed": seed,
        "params": n_params,
        "max_steps": step,
        "epochs": epoch,
        "wall_time": total_time,
        "grokked": grokked,
        "checkpoints": checkpoints,
    }


def run_pilot() -> None:
    """Quick run to verify setup — 5K steps, 1 seed."""
    print("=" * 60)
    print("PILOT RUN: 5K steps, 1 seed")
    print("=" * 60)
    result = run_single(
        seed=42,
        max_steps=5000,
        eval_interval=1000,
    )
    if result:
        print(f"\nPilot: {len(result['checkpoints'])} checkpoints collected")


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-009 grokking experiment."""
    parser = argparse.ArgumentParser(
        description="HYP-009: Grokking x TTC interaction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Quick 5K-step single-seed run",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit number of seed runs",
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
    args = parser.parse_args()

    if args.pilot:
        run_pilot()
        return

    seeds = SEEDS[:]
    if args.max_runs:
        seeds = seeds[: args.max_runs]

    print("HYP-009: Grokking x TTC Interaction")
    print(f"Seeds: {seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"LR: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"K values: {K_VALUES}")
    print(f"N samples: {N_SAMPLES}")
    print(f"Temperature: {TEMPERATURE}")

    results: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i + 1}/{len(seeds)}]", end="")
        result = run_single(
            seed=seed,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            dry_run=args.dry_run,
        )
        if result:
            results.append(result)

    if not results:
        return

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for r in results:
        grok_step = "N/A"
        if r["grokked"]:
            grok_step = str(
                next(
                    c["step"]
                    for c in r["checkpoints"]
                    if c["val_accuracy"] > 0.95
                )
            )
        print(
            f"  {r['run']}: grokked={r['grokked']}, "
            f"grok_step={grok_step}, "
            f"epochs={r['epochs']}, "
            f"wall_time={r['wall_time']:.0f}s"
        )

    # Save results
    out = Path("experiments") / "hyp009_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
