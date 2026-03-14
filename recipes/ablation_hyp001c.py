"""HYP-001c: FLOP-matched GPT-to-LLaMA progressive ablation.

Replicates HYP-001b's progressive ablation (GPT -> LLaMA) but with
FLOP-matched training (1 PFLOPs per run via FLOPCounter) instead of
5-minute time budgets.  This resolves the undertrained-model confound:
HYP-001b gave only ~2 tokens/param, whereas 1 PFLOPs gives ~14
tokens/param.

Key differences from HYP-001b:
  1. FLOP budget replaces time budget (FLOPCounter callback)
  2. Per-config best LR from HYP-001b (no LR sweep)
  3. 5 seeds instead of 3
  4. 90/10 train/val split with periodic evaluation

Literature basis (see memory/literature.md):
  - LIT-002 (Shazeer 2020): SwiGLU needs d_ff * 2/3 for fair comparison
  - LIT-001 (Narang et al. 2021): mods may not transfer across scales

Design: 6 configs x 5 seeds = 30 runs, 1 PFLOPs each.
Estimated total: ~15-20 hours.

Usage:
    uv run python recipes/ablation_hyp001c.py
    uv run python recipes/ablation_hyp001c.py --flop-budget 1e12 --seeds 1
    uv run python recipes/ablation_hyp001c.py --mlflow
"""

import argparse
import math
import time
import urllib.request
from dataclasses import replace
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
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner
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

EXPERIMENT_NAME = "HYP-001c-flop-matched"
OUTPUT_DIR = "experiments"
RESULTS_FILE = Path(OUTPUT_DIR) / "results.jsonl"

FLOP_BUDGET = 1e15  # 1 PFLOPs

# Best LRs from HYP-001b
BEST_LRS = {
    "GPT baseline": 3e-4,
    "+ RMSNorm": 1e-4,
    "+ RoPE": 3e-4,
    "+ SwiGLU FFN": 3e-4,
    "+ GQA": 3e-4,
    "+ No bias (=LLaMA)": 1e-4,
}

SEEDS = [42, 43, 44, 45, 46]

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


def build_ablation_configs(
    vocab_size: int,
) -> list[tuple[str, ModelConfig]]:
    """Build 6 configs adding LLaMA features one at a time.

    Key fix from HYP-001: SwiGLU configs use d_ff=341
    for parameter matching (LIT-002, Shazeer 2020).

    Args:
        vocab_size: Vocabulary size for the model.

    Returns:
        List of (name, ModelConfig) tuples.
    """
    d_model = 256
    n_heads = 8
    n_kv_heads = 4
    d_ff_standard = 512
    # SwiGLU: 3 projections vs 2, so d_ff * 2/3 for
    # parameter matching (LIT-002)
    d_ff_gated = 341
    n_layers = 6
    max_seq_len = 256

    # 1. GPT baseline
    gpt_block = BlockConfig(
        attention="mha",
        ffn="standard",
        norm="layer_norm",
        position="sinusoidal",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff_standard,
        bias=True,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    # 2. + RMSNorm
    rmsnorm_block = replace(gpt_block, norm="rms_norm")

    # 3. + RoPE
    rope_block = replace(rmsnorm_block, position="rope")

    # 4. + Gated FFN (SwiGLU) — d_ff reduced for param matching
    gated_block = replace(
        rope_block,
        ffn="gated",
        d_ff=d_ff_gated,
    )

    # 5. + GQA (4 KV heads for 8 query heads)
    gqa_block = replace(
        gated_block,
        attention="gqa",
        n_kv_heads=n_kv_heads,
    )

    # 6. + No bias = full LLaMA
    llama_block = replace(gqa_block, bias=False)

    configs = [
        ("GPT baseline", gpt_block),
        ("+ RMSNorm", rmsnorm_block),
        ("+ RoPE", rope_block),
        ("+ SwiGLU FFN", gated_block),
        ("+ GQA", gqa_block),
        ("+ No bias (=LLaMA)", llama_block),
    ]

    return [
        (
            name,
            ModelConfig(
                block=block,
                vocab_size=vocab_size,
                n_layers=n_layers,
                tie_embeddings=True,
            ),
        )
        for name, block in configs
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
        eval_accuracy (top-1 token accuracy).
    """
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


def train_one(
    name: str,
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
    config_order: int = 0,
    use_mlflow: bool = False,
) -> dict:
    """Train a single config under a FLOP budget.

    Args:
        name: Config name for logging.
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
        config_order: Ordering index for MLflow display.
        use_mlflow: Whether to log to MLflow.

    Returns:
        Dict with run results.
    """
    description = f"{name}, lr={learning_rate:.0e}, seed={seed}"
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
                "config_name": name,
                "config_order": str(config_order),
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
    mx.eval(model.parameters())
    param_count = model.count_parameters()

    train_config = TrainConfig(
        learning_rate=learning_rate,
        max_steps=1_000_000,  # high cap; FLOP budget stops us
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

    # Outer loop checks FLOP budget; Trainer.train_step() only
    # checks max_steps internally, so we manage stopping here.
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

                # Compute efficiency metrics
                elapsed = time.monotonic() - start_time
                tflops_s = flop_counter.total_flops / max(elapsed, 1e-6) / 1e12
                mfu = tflops_s / HARDWARE_PEAK_TFLOPS

                # Weight norm + memory
                w_norm = compute_weight_norm(model)
                peak_mem = mx.metal.get_peak_memory() / 1e6

                # Parameter update ratio
                update_ratio = 0.0
                if prev_weights is not None:
                    update_ratio = compute_update_ratio(
                        model,
                        prev_weights,
                    )
                prev_weights = snapshot_weights(model)

                # Build eval metrics dict
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
        "config_name": name,
        "config_order": config_order,
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
            "config_name": name,
            "learning_rate": learning_rate,
        },
        param_count=param_count,
        config_dict=config_dict,
        status="keep",
    )

    pflops = total_flops / 1e15
    gap = final_eval_loss - final_loss
    print(
        f"    {name:<22} lr={learning_rate:.0e} "
        f"seed={seed}: train={final_loss:.4f}, "
        f"val={best_val_loss:.4f} (gap={gap:+.4f}), "
        f"acc={final_result['eval_accuracy']:.1%}, "
        f"mfu={final_mfu:.1%}, "
        f"{len(history)} steps, {pflops:.3f} PFLOPs, "
        f"{param_count:,} params, {wall_time:.1f}s"
    )

    return {
        "name": name,
        "seed": seed,
        "learning_rate": learning_rate,
        "final_loss": final_loss,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_eval_loss,
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
    config_names: list[str],
    val_by_config: dict[str, list[float]],
    train_by_config: dict[str, list[float]],
) -> None:
    """Print results table with per-config statistics.

    Args:
        config_names: Ordered list of config names.
        val_by_config: Config name -> list of best val losses.
        train_by_config: Config name -> list of train losses.
    """
    print("\n" + "=" * 78)
    print("HYP-001c: FLOP-Matched Ablation Results")
    print("=" * 78)
    print(
        f"{'Config':<22} {'LR':>6}"
        f" {'Val':>7} {'Train':>7} {'Gap':>7}"
        f" {'Std':>7} {'95% CI':>16} {'N':>3}"
    )
    print("-" * 78)

    for name in config_names:
        val_losses = val_by_config[name]
        train_losses = train_by_config[name]
        stats = compute_statistics(val_losses)
        ci_lo, ci_hi = confidence_interval(val_losses)
        lr = BEST_LRS[name]
        mean_train = sum(train_losses) / len(train_losses)
        gap = stats["mean"] - mean_train

        print(
            f"{name:<22} {lr:>6.0e}"
            f" {stats['mean']:>7.4f}"
            f" {mean_train:>7.4f}"
            f" {gap:>+7.4f}"
            f" {stats['std']:>7.4f}"
            f" [{ci_lo:>7.4f},{ci_hi:>7.4f}]"
            f" {len(val_losses):>3}"
        )


def print_comparison(
    config_names: list[str],
    val_by_config: dict[str, list[float]],
) -> None:
    """Compare configs vs GPT baseline with effect sizes.

    Uses best val loss as the primary comparison metric.

    Args:
        config_names: Ordered list of config names.
        val_by_config: Config name -> list of best val losses.
    """
    print("\n" + "=" * 78)
    print("Comparison vs GPT Baseline (best val loss, at best LR per config)")
    print("=" * 78)
    print(
        f"{'Config':<22} {'Best LR':>8} {'Mean':>8}"
        f" {'Std':>8} {'Cohen d':>9} {'vs BL':>8}"
    )
    print("-" * 78)

    baseline_name = config_names[0]
    bl_losses = val_by_config[baseline_name]
    bl_mean = sum(bl_losses) / len(bl_losses)

    for name in config_names:
        losses = val_by_config[name]
        stats = compute_statistics(losses)
        lr = BEST_LRS[name]

        if name == baseline_name:
            d_str = "   ---"
            imp_str = "   ---"
        else:
            d = cohens_d(bl_losses, losses)
            cur_mean = stats["mean"]
            imp_pct = (bl_mean - cur_mean) / bl_mean * 100
            d_str = f"{d:>+8.2f}"
            imp_str = f"{imp_pct:>+7.1f}%"

        print(
            f"{name:<22} {lr:>8.0e}"
            f" {stats['mean']:>8.4f}"
            f" {stats['std']:>8.4f}"
            f" {d_str} {imp_str}"
        )


def main() -> None:
    """Run HYP-001c: FLOP-matched GPT-to-LLaMA ablation."""
    parser = argparse.ArgumentParser(
        description=("HYP-001c: FLOP-matched GPT->LLaMA ablation"),
    )
    parser.add_argument(
        "--flop-budget",
        type=float,
        default=FLOP_BUDGET,
        help="FLOP budget per run (default: 1e15 = 1 PFLOPs)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds (default: 5)",
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
        help="Steps between validation evaluations (default: 500)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking",
    )
    args = parser.parse_args()

    seeds = [42 + i for i in range(args.seeds)]

    # --- Data (90/10 sequential split, matching nanoGPT) ---
    text = download_shakespeare()
    print(f"Shakespeare: {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    tokens = mx.array(tokenizer.encode(text), dtype=mx.int32)
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"Tokenizer: char-level, {tokenizer.vocab_size} tokens")
    print(
        f"Tokens: {len(tokens):,} total, "
        f"{len(train_tokens):,} train, "
        f"{len(val_tokens):,} val (90/10 split)"
    )

    # --- Configs ---
    configs = build_ablation_configs(tokenizer.vocab_size)
    config_names = [name for name, _ in configs]
    config_order_map = {name: i for i, (name, _) in enumerate(configs)}

    total_runs = len(configs) * len(seeds)
    pflops_each = args.flop_budget / 1e15

    print("\n" + "=" * 78)
    print("HYP-001c: FLOP-Matched GPT-to-LLaMA Ablation")
    print(f"{len(configs)} configs x {len(seeds)} seeds = {total_runs} runs")
    print(f"FLOP budget: {pflops_each:.3f} PFLOPs per run")
    print(
        f"Batch size: {args.batch_size}, Seq len: {args.seq_len}, "
        f"Eval every {args.eval_interval} steps"
    )
    print("Per-config LRs from HYP-001b:")
    for name in config_names:
        print(f"  {name:<22} LR={BEST_LRS[name]:.0e}")
    print("Design fix: SwiGLU d_ff=341 (was 512 in HYP-001, per LIT-002)")
    if args.mlflow:
        print("MLflow tracking: enabled")
    print("=" * 78)

    # --- Training ---
    log = ExperimentLog(RESULTS_FILE)
    # config -> list of val/train losses
    val_by_config: dict[str, list[float]] = {name: [] for name in config_names}
    train_by_config: dict[str, list[float]] = {
        name: [] for name in config_names
    }
    run_count = 0

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for name, config in configs:
            run_count += 1
            lr = BEST_LRS[name]
            print(
                f"  [{run_count}/{total_runs}]",
                end="",
            )
            result = train_one(
                name=name,
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
                config_order=config_order_map[name],
                use_mlflow=args.mlflow,
            )
            val_by_config[name].append(
                result["best_val_loss"],
            )
            train_by_config[name].append(
                result["final_loss"],
            )

    # --- Analysis ---
    print_results_table(
        config_names,
        val_by_config,
        train_by_config,
    )
    print_comparison(config_names, val_by_config)

    # --- Summary ---
    print("\n" + "=" * 78)
    print("Summary (per-config best LR from HYP-001b):")
    for name in config_names:
        val_losses = val_by_config[name]
        train_losses = train_by_config[name]
        val_mean = sum(val_losses) / len(val_losses)
        train_mean = sum(train_losses) / len(train_losses)
        lr = BEST_LRS[name]
        print(
            f"  {name:<22} LR={lr:.0e}  "
            f"val={val_mean:.4f}  "
            f"train={train_mean:.4f}  "
            f"n={len(val_losses)}"
        )

    print(f"\nResults logged to {RESULTS_FILE}")
    log_summary = log.summary()
    print(f"Total entries in log: {log_summary['total']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
