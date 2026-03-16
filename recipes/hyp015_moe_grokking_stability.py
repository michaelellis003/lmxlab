"""HYP-015: Does MoE cause grokking instability?

Pre-registered hypothesis:
    Jamba (MoE+SSM+attention) showed grokking instability in
    HYP-014: it grokked at step 36K then un-grokked (val_acc
    dropped to 65%). Bamba (SSM+attention, no MoE) oscillated
    but stabilized. Falcon-H1 (SSM+attention, no MoE) was
    completely stable. Is MoE the cause?

Competing hypotheses:
    H15-a (MoE destabilizes): Jamba-noMoE groks stably (no
        un-grokking within 50K steps). MoE routing creates
        unstable grokking dynamics.
    H15-b (SSM+attn destabilizes): Jamba-noMoE also shows
        instability. The SSM+attention mixing itself causes
        oscillation; MoE is incidental.
    H15-c (Jamba-specific): Jamba-noMoE is stable but the
        interaction of ALL three (SSM+attention+MoE) creates
        the instability — removing any one fixes it.
    H15-d (Seed-dependent): The instability in HYP-014 was
        seed-specific. Multi-seed Jamba-MoE doesn't consistently
        show instability.

Design:
    2 conditions x 3 seeds = 6 runs:
    - Jamba-noMoE: Same as HYP-014 Jamba but moe_every=999
      (all dense FFN). 3 seeds (42, 43, 44).
    - Jamba-MoE: Same as HYP-014 Jamba (with MoE). 3 seeds
      (42, 43, 44). Seed 42 replicated from HYP-014.

    Same training setup as HYP-014:
    50K steps, per-example training, wd=0.1, lr=1e-3,
    constant LR, batch_size=64, eval every 2K steps.

    Grokking detection: val_accuracy > 0.95.
    Stability criterion: 3+ consecutive post-grok checkpoints
    with val_accuracy > 0.90.

Usage:
    uv run python recipes/hyp015_moe_grokking_stability.py
    uv run python recipes/hyp015_moe_grokking_stability.py --dry-run
    uv run python recipes/hyp015_moe_grokking_stability.py --pilot
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

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.data.modular_arithmetic import ModularArithmeticDataset
from lmxlab.eval.metrics import pass_at_k
from lmxlab.models.base import LanguageModel
from lmxlab.models.jamba import jamba_config

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
EVAL_INTERVAL = 2_000
WARMUP_STEPS = 100

# ── Grokking-scale Jamba configs ─────────────────────────

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


def _configs() -> dict[str, ModelConfig]:
    """Build Jamba configs with and without MoE.

    Note: jamba_config() places MoE on attention layer index 0
    (attn_count % moe_every == 0), so moe_every=999 does NOT
    prevent MoE when there is only 1 attention layer. For the
    noMoE condition we manually replace the MoE block_config
    with a dense FFN block_config.
    """
    moe_cfg = jamba_config(
        n_layers=2,
        attn_every=2,
        n_experts=4,
        top_k_experts=2,
        moe_every=2,
        rope_theta=10000.0,
        **GROK_COMMON,
        **GROK_MAMBA,
    )

    # Build noMoE: replace any MoE FFN blocks with dense
    nomoe_blocks = []
    for bc in moe_cfg.block_configs:
        if bc.ffn == "moe":
            nomoe_blocks.append(
                BlockConfig(
                    attention=bc.attention,
                    ffn="gated",
                    position=bc.position,
                    d_model=bc.d_model,
                    n_heads=bc.n_heads,
                    n_kv_heads=bc.n_kv_heads,
                    d_ff=bc.d_ff,
                    bias=bc.bias,
                    dropout=bc.dropout,
                    norm=bc.norm,
                    norm_eps=bc.norm_eps,
                    rope_theta=bc.rope_theta,
                    max_seq_len=bc.max_seq_len,
                    pre_norm=bc.pre_norm,
                )
            )
        else:
            nomoe_blocks.append(bc)

    nomoe_cfg = ModelConfig(
        block=nomoe_blocks[0],
        block_configs=tuple(nomoe_blocks),
        vocab_size=moe_cfg.vocab_size,
        n_layers=moe_cfg.n_layers,
        tie_embeddings=moe_cfg.tie_embeddings,
    )

    return {
        "jamba_moe": moe_cfg,
        "jamba_nomoe": nomoe_cfg,
    }


CONFIGS = _configs()


# ── Data helpers (reused from HYP-014) ──────────────────


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
    condition: str,
    config: ModelConfig,
    seed: int,
    max_steps: int,
    eval_interval: int,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Train one condition+seed, track grokking + stability.

    Args:
        condition: Condition name (jamba_moe/jamba_nomoe).
        config: Model configuration.
        seed: Random seed.
        max_steps: Maximum training steps.
        eval_interval: Steps between pass@k evaluations.
        dry_run: If True, skip actual training.

    Returns:
        Result dict with checkpoint data, or None.
    """
    run_name = f"{condition}_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  condition={condition}, seed={seed}")

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
    stable = False
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

                # Check stability: 3 consecutive >0.90
                if grokked and not stable:
                    recent = checkpoints[-3:]
                    if len(recent) >= 3 and all(
                        c["val_accuracy"] > 0.90 for c in recent
                    ):
                        stable = True
                        print("  ** STABLE: 3 consecutive >0.90")

                # No early stopping — run all 50K to observe
                # instability patterns

    total_time = time.monotonic() - start_time

    # Detect un-grokking
    ungrokked = False
    if grokked:
        post_grok_cps = [c for c in checkpoints if c["step"] > grok_step]
        for c in post_grok_cps:
            if c["val_accuracy"] < 0.70:
                ungrokked = True
                break

    print(f"\n  {'─' * 50}")
    print(f"  Completed {step} steps in {total_time:.1f}s")
    print(f"  Grokked: {grokked} (step {grok_step})")
    print(f"  Stable: {stable}")
    print(f"  Un-grokked: {ungrokked}")

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
        "condition": condition,
        "seed": seed,
        "params": n_params,
        "max_steps": step,
        "epochs": epoch,
        "wall_time": total_time,
        "grokked": grokked,
        "grok_step": grok_step,
        "stable": stable,
        "ungrokked": ungrokked,
        "checkpoints": checkpoints,
    }


# ── Analysis ─────────────────────────────────────────────


def analyze_results(results: list[dict[str, Any]]) -> None:
    """Print MoE vs no-MoE grokking stability comparison."""
    print(f"\n{'=' * 70}")
    print("MoE vs No-MoE Grokking Stability")
    print(f"{'=' * 70}")

    header = (
        f"{'Condition':<15} {'Seed':>4} {'Params':>8} "
        f"{'Grokked':>7} {'Step':>6} "
        f"{'Stable':>6} {'Ungrok':>6} {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        grok_str = "Yes" if r["grokked"] else "No"
        grok_step = str(r["grok_step"]) if r["grokked"] else "-"
        stable_str = "Yes" if r["stable"] else "No"
        ungrok_str = "Yes" if r["ungrokked"] else "No"
        print(
            f"{r['condition']:<15} "
            f"{r['seed']:>4} "
            f"{r['params']:>8,} "
            f"{grok_str:>7} "
            f"{grok_step:>6} "
            f"{stable_str:>6} "
            f"{ungrok_str:>6} "
            f"{r['wall_time']:>5.0f}s"
        )

    # Summary by condition
    for cond in ["jamba_moe", "jamba_nomoe"]:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        n_grok = sum(1 for r in runs if r["grokked"])
        n_stable = sum(1 for r in runs if r["stable"])
        n_ungrok = sum(1 for r in runs if r["ungrokked"])
        grok_steps = [r["grok_step"] for r in runs if r["grokked"]]
        avg_step = sum(grok_steps) / len(grok_steps) if grok_steps else 0
        print(
            f"\n  {cond}: {n_grok}/{len(runs)} grokked, "
            f"{n_stable}/{len(runs)} stable, "
            f"{n_ungrok}/{len(runs)} un-grokked, "
            f"avg grok step={avg_step:.0f}"
        )


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-015 MoE grokking stability experiment."""
    parser = argparse.ArgumentParser(
        description="HYP-015: MoE grokking stability",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Quick single run (jamba_nomoe, seed 42, 10K)",
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
        print("PILOT: jamba_nomoe, seed=42, 10K steps")
        config = CONFIGS["jamba_nomoe"]
        result = run_single(
            condition="jamba_nomoe",
            config=config,
            seed=42,
            max_steps=10000,
            eval_interval=2000,
        )
        if result:
            print(
                f"\nPilot: grokked={result['grokked']}, "
                f"stable={result['stable']}"
            )
        return

    # Full experiment: 2 conditions x 3 seeds
    run_list = []
    for cond in ["jamba_moe", "jamba_nomoe"]:
        for seed in SEEDS:
            run_list.append((cond, seed))

    if args.max_runs:
        run_list = run_list[: args.max_runs]

    print("HYP-015: MoE Grokking Stability")
    print(f"Runs: {len(run_list)}")
    print(f"Max steps: {args.max_steps}")

    results: list[dict[str, Any]] = []
    for i, (cond, seed) in enumerate(run_list):
        print(f"\n[{i + 1}/{len(run_list)}]", end="")
        config = CONFIGS[cond]
        result = run_single(
            condition=cond,
            config=config,
            seed=seed,
            max_steps=args.max_steps,
            eval_interval=EVAL_INTERVAL,
            dry_run=args.dry_run,
        )
        if result:
            results.append(result)

    if not results:
        return

    analyze_results(results)

    out = Path("experiments") / "hyp015_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
