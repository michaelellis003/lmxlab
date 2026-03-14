"""HYP-001b Sub-experiment A: LR sweep across GPT->LLaMA configs.

Resolves confounds from HYP-001 by:
  1. Sweeping LR {1e-4, 3e-4, 1e-3} per config (addresses ANOM-001)
  2. Fixing d_ff for SwiGLU: 341 instead of 512 (parameter matching)
  3. MLflow tracking for real-time loss curves

Literature basis (see memory/literature.md):
  - LIT-002 (Shazeer 2020): SwiGLU needs d_ff * 2/3 for fair comparison
  - LIT-003 (Yang et al. 2022 muP): fixed-LR comparison is confounded
  - LIT-001 (Narang et al. 2021): mods may not transfer across scales

Design: 6 configs x 3 LRs x 3 seeds = 54 runs, 5-min budget each.
Estimated total: ~4.5 hours.

Usage:
    uv run python recipes/ablation_hyp001b.py
    uv run python recipes/ablation_hyp001b.py --time-budget 30 --seeds 1
    uv run python recipes/ablation_hyp001b.py --time-budget 300 --seeds 3
"""

import argparse
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.experiments.analysis import (
    cohens_d,
    compute_statistics,
    confidence_interval,
)
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner
from lmxlab.experiments.tracking import ExperimentLog
from lmxlab.models.base import LanguageModel
from lmxlab.training.callbacks import MetricsLogger
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_PATH = Path("data/shakespeare.txt")

EXPERIMENT_NAME = "HYP-001b-lr-sweep"
OUTPUT_DIR = "experiments"
RESULTS_FILE = Path(OUTPUT_DIR) / "results.jsonl"

# LR sweep values (from HYP-001b pre-registration)
LEARNING_RATES = [1e-4, 3e-4, 1e-3]


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


def train_one(
    name: str,
    config: ModelConfig,
    tokens: mx.array,
    seq_len: int,
    time_budget_s: float,
    seed: int,
    learning_rate: float,
    log: ExperimentLog,
    use_mlflow: bool = False,
) -> dict:
    """Train a single config at a given LR.

    Args:
        name: Config name for logging.
        config: Model configuration.
        tokens: Token array for training.
        seq_len: Sequence length.
        time_budget_s: Wall-clock time budget in seconds.
        seed: Random seed.
        learning_rate: Learning rate for this run.
        log: Shared experiment log.
        use_mlflow: Whether to log to MLflow.

    Returns:
        Dict with run results.
    """
    description = f"{name}, lr={learning_rate:.0e}, seed={seed}"
    exp_config = ExperimentConfig(
        name=EXPERIMENT_NAME,
        description=description,
        time_budget_s=time_budget_s,
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
                "seed": str(seed),
                "learning_rate": str(learning_rate),
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
        max_steps=100_000,
        batch_size=8,
        compile_step=False,
        warmup_steps=20,
        log_interval=50,
    )

    callbacks = [MetricsLogger(log_interval=100)]
    if use_mlflow:
        callbacks.append(MLflowCallback(log_interval=50))

    trainer = Trainer(model, train_config, callbacks=callbacks)

    # Train with time budget
    history: list[dict] = []
    start_time = time.monotonic()

    while not runner.is_time_up():
        for batch in batch_iterator(
            tokens,
            batch_size=8,
            seq_len=seq_len,
            shuffle=True,
        ):
            if runner.is_time_up():
                break
            metrics = trainer.train_step(batch)
            history.append(metrics)

    wall_time = time.monotonic() - start_time
    final_loss = history[-1]["loss"] if history else float("inf")

    config_dict = {
        "config_name": name,
        "learning_rate": learning_rate,
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
            "val_loss": final_loss,
            "train_loss": final_loss,
            "steps": len(history),
            "config_name": name,
            "learning_rate": learning_rate,
        },
        param_count=param_count,
        config_dict=config_dict,
        status="keep",
    )

    print(
        f"    {name:<22} lr={learning_rate:.0e} seed={seed}: "
        f"loss={final_loss:.4f}, "
        f"{len(history)} steps, "
        f"{param_count:,} params, "
        f"{wall_time:.1f}s"
    )

    return {
        "name": name,
        "seed": seed,
        "learning_rate": learning_rate,
        "final_loss": final_loss,
        "steps": len(history),
        "wall_time": wall_time,
        "param_count": param_count,
    }


def print_results_table(
    config_names: list[str],
    results_by_config_lr: dict[str, dict[float, list[float]]],
) -> dict[str, float]:
    """Print results table and find best LR per config.

    Args:
        config_names: Ordered list of config names.
        results_by_config_lr: Nested dict of
            config -> LR -> list of final losses.

    Returns:
        Dict mapping config name to best learning rate.
    """
    print("\n" + "=" * 78)
    print("HYP-001b Sub-experiment A: LR Sweep Results")
    print("=" * 78)
    print(
        f"{'Config':<22} {'LR':>8} {'Mean':>8} {'Std':>8}"
        f" {'95% CI':>16} {'Best?':>6}"
    )
    print("-" * 78)

    best_lr: dict[str, float] = {}

    for name in config_names:
        lr_results = results_by_config_lr[name]
        best_mean = float("inf")
        best_lr_val = 0.0

        for lr in sorted(lr_results.keys()):
            losses = lr_results[lr]
            stats = compute_statistics(losses)
            ci_lo, ci_hi = confidence_interval(losses)
            mean = stats["mean"]

            if mean < best_mean:
                best_mean = mean
                best_lr_val = lr

        best_lr[name] = best_lr_val

        for lr in sorted(lr_results.keys()):
            losses = lr_results[lr]
            stats = compute_statistics(losses)
            ci_lo, ci_hi = confidence_interval(losses)
            is_best = " ***" if lr == best_lr_val else ""

            print(
                f"{name:<22} {lr:>8.0e}"
                f" {stats['mean']:>8.4f}"
                f" {stats['std']:>8.4f}"
                f" [{ci_lo:>7.4f},{ci_hi:>7.4f}]"
                f"{is_best}"
            )

        print()

    return best_lr


def print_best_lr_comparison(
    config_names: list[str],
    results_by_config_lr: dict[str, dict[float, list[float]]],
    best_lr: dict[str, float],
) -> None:
    """Compare configs at their best LR.

    Args:
        config_names: Ordered list of config names.
        results_by_config_lr: Nested dict of results.
        best_lr: Best LR per config.
    """
    print("=" * 78)
    print("Best-LR Comparison (each config at its optimal LR)")
    print("=" * 78)
    print(
        f"{'Config':<22} {'Best LR':>8} {'Mean':>8}"
        f" {'Std':>8} {'Cohen d':>9} {'vs BL':>8}"
    )
    print("-" * 78)

    baseline_name = config_names[0]
    bl_lr = best_lr[baseline_name]
    bl_losses = results_by_config_lr[baseline_name][bl_lr]

    for name in config_names:
        lr = best_lr[name]
        losses = results_by_config_lr[name][lr]
        stats = compute_statistics(losses)

        if name == baseline_name:
            d_str = "   ---"
            imp_str = "   ---"
        else:
            d = cohens_d(bl_losses, losses)
            bl_mean = sum(bl_losses) / len(bl_losses)
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
    """Run HYP-001b Sub-experiment A: LR sweep."""
    parser = argparse.ArgumentParser(
        description="HYP-001b: LR sweep across GPT->LLaMA configs"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=300.0,
        help="Time budget per run in seconds (default: 300)",
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
        default=128,
        help="Sequence length (default: 128)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking (requires mlflow-skinny)",
    )
    args = parser.parse_args()

    seeds = [42 + i for i in range(args.seeds)]

    # --- Data ---
    text = download_shakespeare()
    print(f"Shakespeare: {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    tokens = mx.array(tokenizer.encode(text), dtype=mx.int32)
    print(f"Tokenizer: char-level, {tokenizer.vocab_size} tokens")
    print(f"Token array: {len(tokens):,} tokens")

    # --- Configs ---
    configs = build_ablation_configs(tokenizer.vocab_size)
    config_names = [name for name, _ in configs]

    total_runs = len(configs) * len(LEARNING_RATES) * len(seeds)
    total_time_est = total_runs * args.time_budget / 60

    print("\n" + "=" * 78)
    print("HYP-001b Sub-experiment A: LR Sweep")
    print(
        f"{len(configs)} configs x {len(LEARNING_RATES)} LRs "
        f"x {len(seeds)} seeds = {total_runs} runs"
    )
    print(f"LRs: {LEARNING_RATES}")
    print(f"Time budget: {args.time_budget:.0f}s per run")
    print(f"Estimated total: ~{total_time_est:.0f} minutes")
    print(f"Seq len: {args.seq_len}")
    print("Design fix: SwiGLU d_ff=341 (was 512 in HYP-001, per LIT-002)")
    if args.mlflow:
        print("MLflow tracking: enabled")
    print("=" * 78)

    # --- Training ---
    log = ExperimentLog(RESULTS_FILE)
    # config -> LR -> list of final losses
    results_by_config_lr: dict[str, dict[float, list[float]]] = {
        name: {lr: [] for lr in LEARNING_RATES} for name in config_names
    }
    run_count = 0

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for name, config in configs:
            for lr in LEARNING_RATES:
                run_count += 1
                print(
                    f"  [{run_count}/{total_runs}]",
                    end="",
                )
                result = train_one(
                    name=name,
                    config=config,
                    tokens=tokens,
                    seq_len=args.seq_len,
                    time_budget_s=args.time_budget,
                    seed=seed,
                    learning_rate=lr,
                    log=log,
                    use_mlflow=args.mlflow,
                )
                results_by_config_lr[name][lr].append(
                    result["final_loss"],
                )

    # --- Analysis ---
    best_lr = print_results_table(
        config_names,
        results_by_config_lr,
    )

    print_best_lr_comparison(
        config_names,
        results_by_config_lr,
        best_lr,
    )

    # --- Summary ---
    print("\n" + "=" * 78)
    print("Best LR per config:")
    for name in config_names:
        lr = best_lr[name]
        losses = results_by_config_lr[name][lr]
        mean = sum(losses) / len(losses)
        print(f"  {name:<22} LR={lr:.0e}  mean={mean:.4f}")

    print(f"\nResults logged to {RESULTS_FILE}")
    log_summary = log.summary()
    print(f"Total entries in log: {log_summary['total']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
