"""HYP-001d: Dropout regularization x architecture interaction.

Tests whether dropout reduces overfitting and changes GPT vs LLaMA
rankings when training on repeated Shakespeare data at 1 PFLOPs.

Key findings motivating this experiment (HYP-001c):
  - All configs overfit severely (train-val gap 0.83-0.93)
  - GPT baseline (val 1.609) beats LLaMA (val 1.670) on val loss
  - RMSNorm lacks implicit regularization (ANOM-007, LIT-020)

Literature basis (see memory/literature.md):
  - LIT-017: Dropout 0.1 standard for <100M params
  - LIT-019: Dropout 0.2 optimal on Tiny Shakespeare (val 1.5531)
  - LIT-020: RMSNorm lacks LayerNorm's implicit regularization
  - nanoGPT: dropout=0.2 on Shakespeare char-level, val 1.4697

Design: 2 archs x 3 dropout rates x 3 seeds = 18 runs, 1 PFLOPs each.
Estimated total: ~9-12 hours.

Usage:
    uv run python recipes/ablation_hyp001d.py
    uv run python recipes/ablation_hyp001d.py --flop-budget 1e12 --seeds 1
    uv run python recipes/ablation_hyp001d.py --mlflow
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.experiments.analysis import (
    cohens_d,
    compute_statistics,
    confidence_interval,
)
from lmxlab.experiments.flops import estimate_flops_per_step
from lmxlab.experiments.runner import (
    ExperimentConfig,
    ExperimentRunner,
)
from lmxlab.experiments.tracking import ExperimentLog
from lmxlab.models.base import LanguageModel
from lmxlab.training.callbacks import FLOPCounter, MetricsLogger
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_PATH = Path("data/shakespeare.txt")

EXPERIMENT_NAME = "HYP-001d-dropout"
OUTPUT_DIR = "experiments"
RESULTS_FILE = Path(OUTPUT_DIR) / "results.jsonl"

FLOP_BUDGET = 1e15  # 1 PFLOPs

# Best LRs from HYP-001b
BEST_LRS = {
    "GPT": 3e-4,
    "LLaMA": 1e-4,
}

DROPOUT_RATES = [0.0, 0.1, 0.2]
SEEDS = [42, 43, 44]

# M3 Pro theoretical peak FP32 TFLOP/s
HARDWARE_PEAK_TFLOPS = 6.5


def download_shakespeare() -> str:
    """Download Shakespeare text if not cached."""
    if DATA_PATH.exists():
        return DATA_PATH.read_text()

    print("Downloading Shakespeare text...")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text()


def build_configs(
    vocab_size: int,
    dropout: float,
) -> list[tuple[str, ModelConfig]]:
    """Build GPT and LLaMA configs with specified dropout.

    Args:
        vocab_size: Vocabulary size for the model.
        dropout: Dropout rate to apply.

    Returns:
        List of (name, ModelConfig) tuples.
    """
    d_model = 256
    n_heads = 8
    n_kv_heads = 4
    d_ff_standard = 512
    d_ff_gated = 341  # SwiGLU: d_ff * 2/3 (LIT-002)
    n_layers = 6
    max_seq_len = 256

    # GPT baseline: LayerNorm, MHA, standard FFN, sinusoidal
    gpt_block = BlockConfig(
        attention="mha",
        ffn="standard",
        norm="layer_norm",
        position="sinusoidal",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff_standard,
        bias=True,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    # Full LLaMA: RMSNorm, GQA, SwiGLU, RoPE, no bias
    llama_block = BlockConfig(
        attention="gqa",
        ffn="gated",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff_gated,
        bias=False,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    return [
        (
            "GPT",
            ModelConfig(
                block=gpt_block,
                vocab_size=vocab_size,
                n_layers=n_layers,
                tie_embeddings=True,
            ),
        ),
        (
            "LLaMA",
            ModelConfig(
                block=llama_block,
                vocab_size=vocab_size,
                n_layers=n_layers,
                tie_embeddings=True,
            ),
        ),
    ]


def evaluate(
    model: LanguageModel,
    val_tokens: mx.array,
    batch_size: int,
    seq_len: int,
) -> dict[str, float]:
    """Run a full evaluation pass over validation data.

    Args:
        model: The language model to evaluate.
        val_tokens: Validation token array.
        batch_size: Batch size for evaluation.
        seq_len: Sequence length.

    Returns:
        Dict with eval_loss, eval_perplexity, and
        eval_accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0
    for x, y in batch_iterator(
        val_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle=False,
    ):
        logits, _ = model(x)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = y.reshape(-1)
        loss = nn.losses.cross_entropy(
            logits,
            targets,
            reduction="mean",
        )
        preds = mx.argmax(logits, axis=-1)
        correct = mx.sum(preds == targets)
        mx.eval(loss, correct)
        total_loss += loss.item()
        total_correct += correct.item()
        total_tokens += targets.size
        n_batches += 1
    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": math.exp(min(avg_loss, 20.0)),
        "eval_accuracy": (total_correct / max(total_tokens, 1)),
    }


def compute_weight_norm(model: LanguageModel) -> float:
    """Compute L2 norm of all model parameters.

    Args:
        model: The language model.

    Returns:
        Global L2 weight norm.
    """
    flat = tree_flatten(model.parameters())
    total = sum(mx.sum(p * p).item() for _, p in flat)
    return math.sqrt(total)


def snapshot_weights(
    model: LanguageModel,
) -> list[tuple[str, mx.array]]:
    """Snapshot current model weights for update tracking.

    Args:
        model: The language model.

    Returns:
        Flattened list of (name, array_copy) tuples.
    """
    return [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]


def compute_update_ratio(
    model: LanguageModel,
    prev_flat: list[tuple[str, mx.array]],
) -> float:
    """Compute ||delta_w|| / ||w|| parameter update ratio.

    Args:
        model: Current model.
        prev_flat: Flattened previous parameters.

    Returns:
        Ratio of update norm to weight norm.
    """
    cur_flat = tree_flatten(model.parameters())
    delta_sq = 0.0
    weight_sq = 0.0
    for (_, c), (_, p) in zip(
        cur_flat,
        prev_flat,
        strict=True,
    ):
        diff = c - p
        mx.eval(diff)
        delta_sq += mx.sum(diff * diff).item()
        weight_sq += mx.sum(c * c).item()
    return math.sqrt(delta_sq) / max(math.sqrt(weight_sq), 1e-8)


def train_one(
    arch_name: str,
    dropout: float,
    config: ModelConfig,
    train_tokens: mx.array,
    val_tokens: mx.array,
    batch_size: int,
    seq_len: int,
    flop_budget: float,
    seed: int,
    learning_rate: float,
    log: ExperimentLog,
    eval_interval: int = 500,
    use_mlflow: bool = False,
) -> dict:
    """Train a single config under a FLOP budget.

    Args:
        arch_name: Architecture name (GPT or LLaMA).
        dropout: Dropout rate for this run.
        config: Model configuration.
        train_tokens: Token array for training.
        val_tokens: Token array for validation.
        batch_size: Training batch size.
        seq_len: Sequence length.
        flop_budget: FLOP budget for this run.
        seed: Random seed.
        learning_rate: Learning rate for this run.
        log: Shared experiment log.
        eval_interval: Steps between validation evaluations.
        use_mlflow: Whether to log to MLflow.

    Returns:
        Dict with run results.
    """
    run_name = f"{arch_name} drop={dropout}"
    description = f"{run_name}, lr={learning_rate:.0e}, seed={seed}"
    exp_config = ExperimentConfig(
        name=EXPERIMENT_NAME,
        description=description,
        time_budget_s=86400.0,  # effectively unlimited
        flop_budget=flop_budget,
        seed=seed,
        output_dir=OUTPUT_DIR,
    )

    if use_mlflow:
        from lmxlab.experiments.mlflow import (
            MLflowCallback,
            MLflowExperimentRunner,
        )

        runner = MLflowExperimentRunner(
            exp_config,
            experiment_name=EXPERIMENT_NAME,
            tags={
                "arch": arch_name,
                "dropout": str(dropout),
                "seed": str(seed),
                "learning_rate": str(learning_rate),
                "flop_budget": str(flop_budget),
            },
            log=log,
        )
    else:
        runner = ExperimentRunner(exp_config, log=log)
    runner.start()

    mx.random.seed(seed)
    model = LanguageModel(config)
    model.train()
    mx.eval(model.parameters())
    param_count = model.count_parameters()

    train_config = TrainConfig(
        learning_rate=learning_rate,
        max_steps=1_000_000,
        batch_size=batch_size,
        compile_step=False,
        warmup_steps=20,
        log_interval=500,
    )

    # FLOP-based stopping
    flops_per_step = estimate_flops_per_step(
        config,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    flop_counter = FLOPCounter(
        flops_per_step,
        flop_budget=flop_budget,
        log_interval=500,
    )

    callbacks = [flop_counter, MetricsLogger(log_interval=500)]
    if use_mlflow:
        callbacks.append(MLflowCallback(log_interval=500))

    trainer = Trainer(model, train_config, callbacks=callbacks)
    flop_counter.on_train_begin(None)

    tokens_per_step = batch_size * seq_len
    history: list[dict] = []
    start_time = time.monotonic()
    best_val_loss = float("inf")
    step_count = 0
    tokens_processed = 0
    ema_loss = None
    spike_count = 0
    prev_weights: list | None = None

    while not flop_counter.should_stop:
        for batch in batch_iterator(
            train_tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle=True,
        ):
            if flop_counter.should_stop:
                break
            metrics = trainer.train_step(batch)
            history.append(metrics)
            step_count += 1
            tokens_processed += tokens_per_step
            loss = metrics["loss"]

            # Loss spike detection (EMA-based)
            if ema_loss is None:
                ema_loss = loss
            else:
                ema_loss = 0.99 * ema_loss + 0.01 * loss
                if loss > 3.0 * ema_loss:
                    spike_count += 1

            # Periodic evaluation
            if step_count % eval_interval == 0:
                eval_result = evaluate(
                    model,
                    val_tokens,
                    batch_size,
                    seq_len,
                )
                eval_loss = eval_result["eval_loss"]
                best_val_loss = min(
                    best_val_loss,
                    eval_loss,
                )

                elapsed = time.monotonic() - start_time
                tflops_s = flop_counter.total_flops / max(elapsed, 1e-6) / 1e12
                mfu = tflops_s / HARDWARE_PEAK_TFLOPS

                w_norm = compute_weight_norm(model)
                peak_mem = mx.metal.get_peak_memory() / 1e6

                update_ratio = 0.0
                if prev_weights is not None:
                    update_ratio = compute_update_ratio(
                        model,
                        prev_weights,
                    )
                prev_weights = snapshot_weights(model)

                eval_result.update(
                    {
                        "weight_norm": w_norm,
                        "peak_memory_mb": peak_mem,
                        "mfu": mfu,
                        "tflops_per_sec": tflops_s,
                        "tokens_processed": float(
                            tokens_processed,
                        ),
                        "train_val_gap": loss - eval_loss,
                        "param_update_ratio": update_ratio,
                        "loss_spikes": float(spike_count),
                    }
                )
                for cb in callbacks:
                    cb.on_eval_end(
                        step_count,
                        eval_result,
                    )

    # Final evaluation
    final_result = evaluate(
        model,
        val_tokens,
        batch_size,
        seq_len,
    )
    final_eval_loss = final_result["eval_loss"]
    best_val_loss = min(best_val_loss, final_eval_loss)

    wall_time = time.monotonic() - start_time
    final_loss = history[-1]["loss"] if history else float("inf")
    total_flops = flop_counter.total_flops
    final_tflops = total_flops / max(wall_time, 1e-6) / 1e12
    final_mfu = final_tflops / HARDWARE_PEAK_TFLOPS
    final_w_norm = compute_weight_norm(model)
    peak_mem = mx.metal.get_peak_memory() / 1e6
    final_grad_norm = history[-1].get("grad_norm", 0.0) if history else 0.0

    config_dict = {
        "arch": arch_name,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "flop_budget": flop_budget,
        "attention": config.block.attention,
        "ffn": config.block.ffn,
        "norm": config.block.norm,
        "position": config.block.position,
        "bias": config.block.bias,
        "n_kv_heads": config.block.n_kv_heads,
        "d_model": config.block.d_model,
        "n_heads": config.block.n_heads,
        "d_ff": config.block.d_ff,
        "n_layers": config.n_layers,
    }

    runner.finish(
        metrics={
            "val_loss": best_val_loss,
            "train_loss": final_loss,
            "final_val_loss": final_eval_loss,
            "val_perplexity": math.exp(
                min(best_val_loss, 20.0),
            ),
            "train_perplexity": math.exp(
                min(final_loss, 20.0),
            ),
            "final_val_accuracy": (final_result["eval_accuracy"]),
            "final_weight_norm": final_w_norm,
            "final_grad_norm": final_grad_norm,
            "peak_memory_mb": peak_mem,
            "mfu": final_mfu,
            "tokens_processed": float(
                tokens_processed,
            ),
            "loss_spikes": float(spike_count),
            "steps": len(history),
            "total_flops": total_flops,
            "arch": arch_name,
            "dropout": dropout,
            "learning_rate": learning_rate,
        },
        param_count=param_count,
        config_dict=config_dict,
        status="keep",
    )

    pflops = total_flops / 1e15
    gap = final_eval_loss - final_loss
    print(
        f"    {run_name:<20} lr={learning_rate:.0e} "
        f"seed={seed}: train={final_loss:.4f}, "
        f"val={best_val_loss:.4f} (gap={gap:+.4f}), "
        f"acc={final_result['eval_accuracy']:.1%}, "
        f"mfu={final_mfu:.1%}, "
        f"{len(history)} steps, {pflops:.3f} PFLOPs, "
        f"{param_count:,} params, {wall_time:.1f}s"
    )

    return {
        "arch": arch_name,
        "dropout": dropout,
        "seed": seed,
        "learning_rate": learning_rate,
        "final_loss": final_loss,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_eval_loss,
        "train_val_gap": final_eval_loss - final_loss,
        "val_perplexity": math.exp(
            min(best_val_loss, 20.0),
        ),
        "final_accuracy": (final_result["eval_accuracy"]),
        "mfu": final_mfu,
        "steps": len(history),
        "total_flops": total_flops,
        "wall_time": wall_time,
        "param_count": param_count,
    }


def print_results_table(
    results_by_key: dict[tuple[str, float], list[dict]],
) -> None:
    """Print results table with per-config statistics.

    Args:
        results_by_key: (arch, dropout) -> list of result
            dicts.
    """
    print("\n" + "=" * 78)
    print("HYP-001d: Dropout x Architecture Results")
    print("=" * 78)
    print(
        f"{'Config':<20} {'Drop':>5} {'LR':>6}"
        f" {'Val':>7} {'Train':>7} {'Gap':>7}"
        f" {'Std':>7} {'95% CI':>16} {'N':>3}"
    )
    print("-" * 78)

    for (arch, drop), runs in sorted(
        results_by_key.items(),
    ):
        val_losses = [r["best_val_loss"] for r in runs]
        train_losses = [r["final_loss"] for r in runs]
        stats = compute_statistics(val_losses)
        ci_lo, ci_hi = confidence_interval(val_losses)
        lr = BEST_LRS[arch]
        mean_train = sum(train_losses) / len(train_losses)
        gap = stats["mean"] - mean_train

        print(
            f"{arch:<20} {drop:>5.1f} {lr:>6.0e}"
            f" {stats['mean']:>7.4f}"
            f" {mean_train:>7.4f}"
            f" {gap:>+7.4f}"
            f" {stats['std']:>7.4f}"
            f" [{ci_lo:>7.4f},{ci_hi:>7.4f}]"
            f" {len(val_losses):>3}"
        )


def print_comparison(
    results_by_key: dict[tuple[str, float], list[dict]],
) -> None:
    """Compare architectures and dropout effects.

    Args:
        results_by_key: (arch, dropout) -> list of result
            dicts.
    """
    print("\n" + "=" * 78)
    print("Pairwise Comparisons (Cohen's d)")
    print("=" * 78)

    # GPT vs LLaMA at each dropout rate
    print("\nGPT vs LLaMA at each dropout rate:")
    print(
        f"  {'Dropout':>7} {'GPT val':>8} {'LLaMA val':>9}"
        f" {'d':>8} {'Interpretation':>15}"
    )
    print("  " + "-" * 50)
    for drop in DROPOUT_RATES:
        gpt_key = ("GPT", drop)
        llama_key = ("LLaMA", drop)
        if gpt_key in results_by_key and llama_key in results_by_key:
            gpt_vals = [r["best_val_loss"] for r in results_by_key[gpt_key]]
            llama_vals = [
                r["best_val_loss"] for r in results_by_key[llama_key]
            ]
            gpt_mean = sum(gpt_vals) / len(gpt_vals)
            llama_mean = sum(llama_vals) / len(llama_vals)
            d = cohens_d(gpt_vals, llama_vals)
            interp = (
                "negligible"
                if abs(d) < 0.2
                else "small"
                if abs(d) < 0.5
                else "medium"
                if abs(d) < 0.8
                else "large"
            )
            print(
                f"  {drop:>7.1f} {gpt_mean:>8.4f}"
                f" {llama_mean:>9.4f}"
                f" {d:>+8.2f} {interp:>15}"
            )

    # Dropout effect within each architecture
    print("\nDropout effect within each architecture:")
    print(f"  {'Arch':<8} {'Comparison':>14} {'d':>8} {'Gap reduction':>14}")
    print("  " + "-" * 50)
    for arch in ["GPT", "LLaMA"]:
        base_key = (arch, 0.0)
        if base_key not in results_by_key:
            continue
        base_vals = [r["best_val_loss"] for r in results_by_key[base_key]]
        base_gaps = [r["train_val_gap"] for r in results_by_key[base_key]]
        base_gap_mean = sum(base_gaps) / len(base_gaps)
        for drop in [0.1, 0.2]:
            key = (arch, drop)
            if key not in results_by_key:
                continue
            vals = [r["best_val_loss"] for r in results_by_key[key]]
            gaps = [r["train_val_gap"] for r in results_by_key[key]]
            gap_mean = sum(gaps) / len(gaps)
            d = cohens_d(base_vals, vals)
            gap_reduction = base_gap_mean - gap_mean
            print(
                f"  {arch:<8} 0.0 vs {drop:.1f}"
                f" {d:>+8.2f}"
                f" {gap_reduction:>+14.4f}"
            )


def main() -> None:
    """Run HYP-001d: Dropout x Architecture ablation."""
    parser = argparse.ArgumentParser(
        description=("HYP-001d: Dropout x Architecture ablation"),
    )
    parser.add_argument(
        "--flop-budget",
        type=float,
        default=FLOP_BUDGET,
        help=("FLOP budget per run (default: 1e15 = 1 PFLOPs)"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds (default: 3)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length (default: 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help=("Steps between validation evaluations (default: 500)"),
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking",
    )
    args = parser.parse_args()

    seeds = [42 + i for i in range(args.seeds)]

    # --- Data (90/10 sequential split) ---
    text = download_shakespeare()
    print(f"Shakespeare: {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    tokens = mx.array(
        tokenizer.encode(text),
        dtype=mx.int32,
    )
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"Tokenizer: char-level, {tokenizer.vocab_size} tokens")
    print(
        f"Tokens: {len(tokens):,} total, "
        f"{len(train_tokens):,} train, "
        f"{len(val_tokens):,} val (90/10 split)"
    )

    total_runs = 2 * len(DROPOUT_RATES) * len(seeds)
    pflops_each = args.flop_budget / 1e15

    print("\n" + "=" * 78)
    print("HYP-001d: Dropout x Architecture Ablation")
    print(
        f"2 archs x {len(DROPOUT_RATES)} dropout rates "
        f"x {len(seeds)} seeds = {total_runs} runs"
    )
    print(f"FLOP budget: {pflops_each:.3f} PFLOPs per run")
    print(
        f"Batch size: {args.batch_size}, "
        f"Seq len: {args.seq_len}, "
        f"Eval every {args.eval_interval} steps"
    )
    print("Dropout rates:", DROPOUT_RATES)
    print("Per-arch LRs from HYP-001b:")
    for arch, lr in BEST_LRS.items():
        print(f"  {arch:<8} LR={lr:.0e}")
    if args.mlflow:
        print("MLflow tracking: enabled")
    print("=" * 78)

    # --- Training ---
    log = ExperimentLog(RESULTS_FILE)
    results_by_key: dict[tuple[str, float], list[dict]] = {}
    run_count = 0

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for dropout in DROPOUT_RATES:
            configs = build_configs(
                tokenizer.vocab_size,
                dropout,
            )
            for arch_name, config in configs:
                run_count += 1
                lr = BEST_LRS[arch_name]
                print(
                    f"  [{run_count}/{total_runs}]",
                    end="",
                )
                result = train_one(
                    arch_name=arch_name,
                    dropout=dropout,
                    config=config,
                    train_tokens=train_tokens,
                    val_tokens=val_tokens,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    flop_budget=args.flop_budget,
                    seed=seed,
                    learning_rate=lr,
                    log=log,
                    eval_interval=args.eval_interval,
                    use_mlflow=args.mlflow,
                )
                key = (arch_name, dropout)
                results_by_key.setdefault(
                    key,
                    [],
                ).append(result)

    # --- Analysis ---
    print_results_table(results_by_key)
    print_comparison(results_by_key)

    # --- Summary ---
    print("\n" + "=" * 78)
    print("Summary:")
    for (arch, drop), runs in sorted(
        results_by_key.items(),
    ):
        val_losses = [r["best_val_loss"] for r in runs]
        train_losses = [r["final_loss"] for r in runs]
        val_mean = sum(val_losses) / len(val_losses)
        train_mean = sum(train_losses) / len(train_losses)
        lr = BEST_LRS[arch]
        print(
            f"  {arch:<8} drop={drop:.1f} LR={lr:.0e}"
            f"  val={val_mean:.4f}"
            f"  train={train_mean:.4f}"
            f"  gap={val_mean - train_mean:+.4f}"
            f"  n={len(val_losses)}"
        )

    print(f"\nResults logged to {RESULTS_FILE}")
    log_summary = log.summary()
    print(f"Total entries in log: {log_summary['total']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
