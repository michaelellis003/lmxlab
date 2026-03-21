"""HYP-037: Commutator Defect Predicts Grokking Onset Across Seeds

Pre-registered hypothesis:
    HYP-016 showed no aggregate metric (p@64, val_loss, val_acc) at step
    2K predicts grokking onset across 10 seeds. LIT-137 (arXiv 2602.16967)
    introduces the commutator defect -- a curvature measure from non-
    commuting gradient updates -- that rises before generalization.

    Novel angle: does the defect predict grokking ACROSS seeds (not just
    within runs)?

Competing hypotheses:
    H37-a: Defect at step 2K correlates with grok_step (|rho| >= 0.6).
    H37-b: Defect separates grokkers/non-grokkers (Cohen's d > 0.5).
    H37-c: No cross-seed signal (both |rho| < 0.4 and d < 0.3).

Design:
    10 seeds x 1 architecture (MoE-Jamba 7M) = 10 runs.
    10K steps, defect measured at steps 1K, 2K, 5K, 10K.
    K=5 mini-batch pairs per measurement, eta_comm=1e-3.
    Grok_step values from HYP-016 (known).

Usage:
    uv run python recipes/hyp037_commutator_defect.py
    uv run python recipes/hyp037_commutator_defect.py --dry-run
    uv run python recipes/hyp037_commutator_defect.py --pilot
    uv run python recipes/hyp037_commutator_defect.py --max-runs 2
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
from lmxlab.models.base import LanguageModel
from lmxlab.models.jamba import jamba_config

# ── Constants ────────────────────────────────────────────

SEEDS = list(range(42, 52))  # 10 seeds: 42..51
MODULUS = 97
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
BATCH_SIZE = 64
MAX_STEPS = 10_000
DEFECT_STEPS = [1000, 2000, 5000, 10000]  # When to measure defect
K_MEASUREMENTS = 5  # Independent measurements per checkpoint
ETA_COMM = 1e-3  # Perturbation step size for defect

# Known grok steps from HYP-016 (censored at 60K for non-grokkers)
GROK_STEPS = {
    42: 18000, 43: 12000, 44: 60000, 45: 48000, 46: 22000,
    47: 12000, 48: 4000, 49: 48000, 50: 12000, 51: 36000,
}

WARMUP_STEPS = 100

# ── Jamba MoE config (same as HYP-015/016) ────────────────

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


# ── Data helpers (reused from HYP-016) ───────────────────


def build_example_batches(
    dataset: ModularArithmeticDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[mx.array, mx.array]]:
    """Build padded batches of complete examples."""
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


# ── Commutator Defect ────────────────────────────────────


def _flatten_params(model: LanguageModel) -> mx.array:
    """Flatten all model parameters into a single 1D array."""
    leaves = model.parameters()
    flat_parts = []
    for leaf in nn.utils.tree_flatten(leaves):
        flat_parts.append(leaf[1].reshape(-1))
    return mx.concatenate(flat_parts)


def _load_flat_params(model: LanguageModel, flat: mx.array) -> None:
    """Load a flat parameter vector back into the model."""
    leaves = nn.utils.tree_flatten(model.parameters())
    offset = 0
    updates = {}
    for name, param in leaves:
        n = param.size
        new_val = flat[offset : offset + n].reshape(param.shape)
        updates[name] = new_val
        offset += n
    model.load_weights(list(updates.items()))


def compute_commutator_defect(
    model: LanguageModel,
    batches: list[tuple[mx.array, mx.array]],
    rng: np.random.Generator,
    eta: float = ETA_COMM,
    k: int = K_MEASUREMENTS,
) -> dict[str, float]:
    """Compute the commutator defect metric.

    The defect measures non-commutativity of gradient updates:
    D = ||theta_AB - theta_BA|| / (||eta*g_A|| * ||eta*g_B||)

    Args:
        model: Current model (weights will be temporarily modified).
        batches: Available training batches to sample from.
        rng: Random generator for selecting batch pairs.
        eta: Perturbation step size.
        k: Number of independent measurements.

    Returns:
        Dict with median, p25, p75, and all measurements.
    """
    defects = []

    def loss_fn(mdl, inp, tgt):
        logits, _ = mdl(inp)
        return nn.losses.cross_entropy(logits, tgt, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Save original params
    theta_0 = _flatten_params(model)
    mx.eval(theta_0)

    n_batches = len(batches)

    for _ in range(k):
        # Pick 2 random batches
        idx = rng.choice(n_batches, size=2, replace=False)
        inp_a, tgt_a = batches[idx[0]]
        inp_b, tgt_b = batches[idx[1]]

        # Compute gradients at theta_0
        _load_flat_params(model, theta_0)
        mx.eval(model.parameters())

        _, grads_a = loss_and_grad(model, inp_a, tgt_a)
        flat_ga = mx.concatenate(
            [g[1].reshape(-1) for g in nn.utils.tree_flatten(grads_a)]
        )
        mx.eval(flat_ga)

        _, grads_b = loss_and_grad(model, inp_b, tgt_b)
        flat_gb = mx.concatenate(
            [g[1].reshape(-1) for g in nn.utils.tree_flatten(grads_b)]
        )
        mx.eval(flat_gb)

        # Order A then B: theta_A = theta_0 - eta*g_A
        theta_a = theta_0 - eta * flat_ga
        _load_flat_params(model, theta_a)
        mx.eval(model.parameters())

        _, grads_b_at_a = loss_and_grad(model, inp_b, tgt_b)
        flat_gb_at_a = mx.concatenate(
            [g[1].reshape(-1) for g in nn.utils.tree_flatten(grads_b_at_a)]
        )
        mx.eval(flat_gb_at_a)
        theta_ab = theta_a - eta * flat_gb_at_a

        # Order B then A: theta_B = theta_0 - eta*g_B
        theta_b = theta_0 - eta * flat_gb
        _load_flat_params(model, theta_b)
        mx.eval(model.parameters())

        _, grads_a_at_b = loss_and_grad(model, inp_a, tgt_a)
        flat_ga_at_b = mx.concatenate(
            [g[1].reshape(-1) for g in nn.utils.tree_flatten(grads_a_at_b)]
        )
        mx.eval(flat_ga_at_b)
        theta_ba = theta_b - eta * flat_ga_at_b

        # Commutator defect
        diff = theta_ab - theta_ba
        mx.eval(diff)

        norm_diff = mx.sqrt(mx.sum(diff * diff)).item()
        norm_ga = mx.sqrt(mx.sum(flat_ga * flat_ga)).item()
        norm_gb = mx.sqrt(mx.sum(flat_gb * flat_gb)).item()

        denom = eta * norm_ga * eta * norm_gb
        defect = norm_diff / denom if denom > 1e-12 else 0.0
        defects.append(defect)

    # Restore original params
    _load_flat_params(model, theta_0)
    mx.eval(model.parameters())

    defects_sorted = sorted(defects)
    median = float(np.median(defects_sorted))
    p25 = float(np.percentile(defects_sorted, 25))
    p75 = float(np.percentile(defects_sorted, 75))

    return {
        "defect_median": median,
        "defect_p25": p25,
        "defect_p75": p75,
        "defect_all": defects,
    }


# ── Core run function ────────────────────────────────────


def run_single(
    seed: int,
    config: ModelConfig,
    max_steps: int,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Train one seed, measure commutator defect at checkpoints.

    Args:
        seed: Random seed.
        config: Model configuration.
        max_steps: Maximum training steps.
        dry_run: If True, skip actual training.

    Returns:
        Result dict with defect measurements, or None.
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

    # Optimizer
    warmup = optim.linear_schedule(0.0, LEARNING_RATE, WARMUP_STEPS)
    constant = optim.linear_schedule(LEARNING_RATE, LEARNING_RATE, max_steps)
    schedule = optim.join_schedules([warmup, constant], [WARMUP_STEPS])
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=WEIGHT_DECAY,
    )

    def loss_fn(mdl, input_ids, target_ids):
        logits, _ = mdl(input_ids)
        return nn.losses.cross_entropy(logits, target_ids, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    defect_checkpoints: list[dict[str, Any]] = []
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

            if step in DEFECT_STEPS:
                elapsed = time.monotonic() - start_time
                train_loss = loss.item()

                print(f"  ** Measuring defect at step {step}...")

                # Get fresh batches for defect computation
                defect_rng = np.random.default_rng(seed * 1000 + step)
                defect_batches = build_example_batches(
                    train_ds, BATCH_SIZE, defect_rng
                )

                defect = compute_commutator_defect(
                    model=model,
                    batches=defect_batches,
                    rng=defect_rng,
                    eta=ETA_COMM,
                    k=K_MEASUREMENTS,
                )

                checkpoint = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "wall_time": elapsed,
                    **defect,
                }
                defect_checkpoints.append(checkpoint)

                print(
                    f"     defect: median={defect['defect_median']:.6f} "
                    f"[{defect['defect_p25']:.6f}, "
                    f"{defect['defect_p75']:.6f}]"
                )

    total_time = time.monotonic() - start_time

    print(f"\n  {'─' * 50}")
    print(f"  Completed {step} steps in {total_time:.1f}s")
    print(f"  Known grok_step: {GROK_STEPS.get(seed, 'unknown')}")

    # Print defect trajectory
    hdr = f"{'Step':>6} {'Loss':>8} {'Defect':>10} {'P25':>10} {'P75':>10}"
    print(f"\n  {hdr}")
    print(f"  {'─' * 48}")
    for cp in defect_checkpoints:
        print(
            f"  {cp['step']:>6d} "
            f"{cp['train_loss']:>8.4f} "
            f"{cp['defect_median']:>10.6f} "
            f"{cp['defect_p25']:>10.6f} "
            f"{cp['defect_p75']:>10.6f}"
        )

    return {
        "run": run_name,
        "seed": seed,
        "params": n_params,
        "max_steps": step,
        "epochs": epoch,
        "wall_time": total_time,
        "grok_step": GROK_STEPS.get(seed, -1),
        "defect_checkpoints": defect_checkpoints,
    }


# ── Analysis ─────────────────────────────────────────────


def analyze_results(results: list[dict[str, Any]]) -> None:
    """Analyze cross-seed defect prediction of grokking."""
    print(f"\n{'=' * 70}")
    print("HYP-037: Commutator Defect as Cross-Seed Grokking Predictor")
    print(f"{'=' * 70}")

    # Summary table for defect at step 2K
    header = (
        f"{'Seed':>4} {'GrokStep':>8} {'Defect@1K':>10} "
        f"{'Defect@2K':>10} {'Defect@5K':>10} {'Defect@10K':>10}"
    )
    print(header)
    print("-" * len(header))

    seed_data = []
    for r in results:
        grok_step = r["grok_step"]
        grok_str = str(grok_step) if grok_step < 60000 else ">50K"

        defects_by_step = {}
        for cp in r["defect_checkpoints"]:
            defects_by_step[cp["step"]] = cp["defect_median"]

        d1k = defects_by_step.get(1000, float("nan"))
        d2k = defects_by_step.get(2000, float("nan"))
        d5k = defects_by_step.get(5000, float("nan"))
        d10k = defects_by_step.get(10000, float("nan"))

        print(
            f"{r['seed']:>4} {grok_str:>8} "
            f"{d1k:>10.6f} {d2k:>10.6f} "
            f"{d5k:>10.6f} {d10k:>10.6f}"
        )

        seed_data.append({
            "seed": r["seed"],
            "grok_step": grok_step,
            "defect_1k": d1k,
            "defect_2k": d2k,
            "defect_5k": d5k,
            "defect_10k": d10k,
        })

    if len(seed_data) < 3:
        print("\nToo few runs for correlation analysis.")
        return

    # Spearman rank correlation at each checkpoint
    from scipy.stats import spearmanr

    grok_steps = [d["grok_step"] for d in seed_data]

    print(f"\n{'─' * 50}")
    print("Spearman correlations (defect vs grok_step):")

    for step_name, key in [
        ("1K", "defect_1k"),
        ("2K", "defect_2k"),
        ("5K", "defect_5k"),
        ("10K", "defect_10k"),
    ]:
        vals = [d[key] for d in seed_data]
        if any(np.isnan(v) for v in vals):
            print(f"  defect@{step_name}: SKIPPED (missing data)")
            continue
        rho, p = spearmanr(vals, grok_steps)
        print(f"  defect@{step_name}: rho={rho:.3f}, p={p:.4f}")

    # Grokker vs non-grokker separation at step 2K
    grokkers = [d for d in seed_data if d["grok_step"] < 60000]
    nongrokkers = [d for d in seed_data if d["grok_step"] >= 60000]

    print(f"\n{'─' * 50}")
    print(
        f"Grokker/non-grokker separation "
        f"({len(grokkers)} vs {len(nongrokkers)}):"
    )

    if grokkers and nongrokkers:
        avg_grok = np.mean([d["defect_2k"] for d in grokkers])
        avg_nongrok = np.mean([d["defect_2k"] for d in nongrokkers])
        std_grok = np.std([d["defect_2k"] for d in grokkers])
        pooled_std = std_grok if std_grok > 1e-12 else 1e-12
        cohens_d = abs(avg_grok - avg_nongrok) / pooled_std
        print(
            f"  defect@2K: grokkers={avg_grok:.6f}, "
            f"non-grokkers={avg_nongrok:.6f}"
        )
        print(f"  Cohen's d = {cohens_d:.3f}")
    else:
        print("  Cannot compute (need both groups)")

    # Adjudication
    print(f"\n{'─' * 50}")
    print("Hypothesis adjudication:")

    d2k_vals = [d["defect_2k"] for d in seed_data]
    if not any(np.isnan(v) for v in d2k_vals):
        rho_2k, _ = spearmanr(d2k_vals, grok_steps)
        abs_rho = abs(rho_2k) if not np.isnan(rho_2k) else 0

        if abs_rho >= 0.6:
            print(
                f"  H37-a (defect predicts): SUPPORTED "
                f"(|rho|={abs_rho:.3f} >= 0.6)"
            )
        elif abs_rho >= 0.4:
            print(
                f"  H37-a (defect predicts): PARTIALLY SUPPORTED "
                f"(|rho|={abs_rho:.3f}, 0.4-0.6)"
            )
        else:
            print(
                f"  H37-a (defect predicts): NOT SUPPORTED "
                f"(|rho|={abs_rho:.3f} < 0.4)"
            )

        if grokkers and nongrokkers:
            if cohens_d > 0.5:
                print(
                    f"  H37-b (separates groups): SUPPORTED "
                    f"(d={cohens_d:.3f} > 0.5)"
                )
            else:
                print(
                    f"  H37-b (separates groups): NOT SUPPORTED "
                    f"(d={cohens_d:.3f} <= 0.5)"
                )
        else:
            print(
                "  H37-b: CANNOT ASSESS "
                "(insufficient groups)"
            )

        if abs_rho < 0.4 and (not nongrokkers or cohens_d < 0.3):
            print(
                "  H37-c (no cross-seed signal): SUPPORTED "
                "(both |rho| < 0.4 and d < 0.3)"
            )
        elif abs_rho >= 0.4 or (nongrokkers and cohens_d > 0.3):
            print("  H37-c (no cross-seed signal): NOT SUPPORTED")
        else:
            print("  H37-c (no cross-seed signal): INCONCLUSIVE")


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    """Run the HYP-037 commutator defect experiment."""
    parser = argparse.ArgumentParser(
        description="HYP-037: Commutator defect as grokking predictor",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print config without running",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Quick single run (seed 48, 2K steps)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=MAX_STEPS,
        help=f"Max training steps (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None, help="Limit number of runs",
    )
    args = parser.parse_args()

    if args.pilot:
        print("PILOT: jamba_moe, seed=48, 2K steps, defect at 1K/2K")
        global DEFECT_STEPS
        DEFECT_STEPS = [1000, 2000]
        result = run_single(
            seed=48, config=CONFIG, max_steps=2000,
        )
        if result:
            for cp in result["defect_checkpoints"]:
                print(
                    f"  step={cp['step']} "
                    f"defect={cp['defect_median']:.6f}"
                )
        return

    # Full experiment: 10 seeds
    seeds = SEEDS[: args.max_runs] if args.max_runs else SEEDS

    print("HYP-037: Commutator Defect as Cross-Seed Grokking Predictor")
    print(f"Seeds: {seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Defect checkpoints: {DEFECT_STEPS}")
    print(f"K measurements per checkpoint: {K_MEASUREMENTS}")

    results: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i + 1}/{len(seeds)}]", end="")
        result = run_single(
            seed=seed, config=CONFIG, max_steps=args.max_steps,
            dry_run=args.dry_run,
        )
        if result:
            results.append(result)

    if not results:
        return

    analyze_results(results)

    out = Path("experiments") / "hyp037_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
