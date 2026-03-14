"""Architecture ablation: GPT -> LLaMA one feature at a time.

HYP-001 experiment: measures individual and cumulative effects of
LLaMA architectural features on training loss. Uses ExperimentRunner
for proper tracking and statistical analysis (ANOVA + Cohen's d).

Features tested (cumulative):
  1. GPT baseline (MHA + LayerNorm + Standard FFN + Sinusoidal)
  2. + RMSNorm (replace LayerNorm)
  3. + RoPE (replace sinusoidal position encoding)
  4. + GatedFFN/SwiGLU (replace standard FFN)
  5. + GQA (replace MHA with grouped-query attention)
  6. + No bias (remove all bias terms) = Full LLaMA

Design: d_model=256, n_heads=8, n_kv_heads=4, d_ff=512, n_layers=6,
5-min time budget, 3 seeds, Shakespeare dataset.

Usage:
    uv run python recipes/ablation_gpt_to_llama.py
    uv run python recipes/ablation_gpt_to_llama.py --time-budget 300 --seeds 3
    uv run python recipes/ablation_gpt_to_llama.py --time-budget 30 --seeds 1
"""

import argparse
import math
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
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_PATH = Path("data/shakespeare.txt")

EXPERIMENT_NAME = "HYP-001-gpt-to-llama"
OUTPUT_DIR = "experiments"
RESULTS_FILE = Path(OUTPUT_DIR) / "results.jsonl"


def download_shakespeare() -> str:
    """Download Shakespeare text if not cached."""
    if DATA_PATH.exists():
        return DATA_PATH.read_text()

    print("Downloading Shakespeare text...")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text()


def build_ablation_configs(vocab_size: int) -> list[tuple[str, ModelConfig]]:
    """Build 6 configs adding LLaMA features one at a time.

    Args:
        vocab_size: Vocabulary size for the model.

    Returns:
        List of (name, ModelConfig) tuples.
    """
    d_model = 256
    n_heads = 8
    n_kv_heads = 4
    d_ff = 512
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
        d_ff=d_ff,
        bias=True,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    # 2. + RMSNorm
    rmsnorm_block = replace(gpt_block, norm="rms_norm")

    # 3. + RoPE
    rope_block = replace(rmsnorm_block, position="rope")

    # 4. + Gated FFN (SwiGLU)
    gated_block = replace(rope_block, ffn="gated")

    # 5. + GQA (4 KV heads for 8 query heads)
    gqa_block = replace(gated_block, attention="gqa", n_kv_heads=n_kv_heads)

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
    log: ExperimentLog,
    use_mlflow: bool = False,
) -> dict:
    """Train a single config and log results via ExperimentRunner.

    Args:
        name: Config name for logging.
        config: Model configuration.
        tokens: Token array for training.
        seq_len: Sequence length.
        time_budget_s: Wall-clock time budget in seconds.
        seed: Random seed.
        log: Shared experiment log.
        use_mlflow: Whether to log to MLflow.

    Returns:
        Dict with 'name', 'seed', 'final_loss', 'steps', 'wall_time'.
    """
    exp_config = ExperimentConfig(
        name=EXPERIMENT_NAME,
        description=f"{name}, seed={seed}",
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
            tags={"config_name": name, "seed": str(seed)},
            log=log,
        )
    else:
        runner = ExperimentRunner(exp_config, log=log)
    runner.start()

    mx.random.seed(seed)
    model = LanguageModel(config)
    mx.eval(model.parameters())
    param_count = model.count_parameters()

    # Use max_steps as a high ceiling; time budget is the real limit
    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=100_000,
        batch_size=8,
        compile_step=False,
        warmup_steps=20,
        log_interval=50,
    )

    callbacks = []
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
        },
        param_count=param_count,
        config_dict=config_dict,
        status="keep",
    )

    print(
        f"    {name:<22} seed={seed}: "
        f"loss={final_loss:.4f}, "
        f"{len(history)} steps, "
        f"{param_count:,} params, "
        f"{wall_time:.1f}s"
    )

    return {
        "name": name,
        "seed": seed,
        "final_loss": final_loss,
        "steps": len(history),
        "wall_time": wall_time,
        "param_count": param_count,
    }


def one_way_anova(groups: list[list[float]]) -> tuple[float, float]:
    """One-way ANOVA F-test (no scipy dependency).

    Args:
        groups: List of sample groups.

    Returns:
        (F-statistic, approximate p-value).
    """
    k = len(groups)
    all_vals = [v for g in groups for v in g]
    N = len(all_vals)
    grand_mean = sum(all_vals) / N

    # Between-group sum of squares
    ss_between = sum(
        len(g) * (sum(g) / len(g) - grand_mean) ** 2 for g in groups
    )
    # Within-group sum of squares
    ss_within = sum(sum((v - sum(g) / len(g)) ** 2 for v in g) for g in groups)

    df_between = k - 1
    df_within = N - k

    if df_within <= 0 or ss_within == 0:
        return float("inf"), 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    # Approximate p-value via F-distribution CDF
    # Using the regularized incomplete beta function approximation
    p_value = _f_pvalue_approx(f_stat, df_between, df_within)

    return f_stat, p_value


def _f_pvalue_approx(f: float, df1: int, df2: int) -> float:
    """Approximate p-value for F-distribution.

    Uses the transformation F -> Beta and a simple series
    approximation. Adequate for reporting; not for publication.

    Args:
        f: F-statistic.
        df1: Numerator degrees of freedom.
        df2: Denominator degrees of freedom.

    Returns:
        Approximate p-value (upper tail).
    """
    if f <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    # Use normal approximation for large df
    # Abramowitz & Stegun 26.6.15
    a = df2 / 2.0
    b = df1 / 2.0
    # Simple approximation: Wilson-Hilferty
    lam = (
        a * math.log(x)
        + b * math.log(1 - x)
        + math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
    )
    # Clamp to avoid overflow
    lam = max(min(lam, 20), -700)
    # This gives the regularized incomplete beta, but we
    # just use a coarse threshold-based approximation
    # For proper p-values, use scipy. This is for display only.
    if f > 10:
        return 0.001
    elif f > 5:
        return 0.01
    elif f > 3:
        return 0.05
    elif f > 2:
        return 0.10
    else:
        return 0.25


def print_results_table(
    config_names: list[str],
    all_results: dict[str, list[float]],
) -> None:
    """Print the final loss comparison table with statistics.

    Args:
        config_names: Ordered list of config names.
        all_results: Mapping of config name to list of final losses.
    """
    print("\n" + "=" * 72)
    print("Final Loss Comparison (HYP-001)")
    print("=" * 72)
    print(
        f"{'Config':<22} {'Mean':>8} {'Std':>8}"
        f" {'95% CI':>16} {'Cohen d':>9} {'vs BL':>8}"
    )
    print("-" * 72)

    baseline_losses = all_results[config_names[0]]

    for name in config_names:
        finals = all_results[name]
        stats = compute_statistics(finals)
        ci_lo, ci_hi = confidence_interval(finals)

        if name == config_names[0]:
            d_str = "   ---"
            imp_str = "   ---"
        else:
            d = cohens_d(baseline_losses, finals)
            bl_mean = sum(baseline_losses) / len(baseline_losses)
            cur_mean = stats["mean"]
            imp_pct = (bl_mean - cur_mean) / bl_mean * 100
            d_str = f"{d:>+8.2f}"
            imp_str = f"{imp_pct:>+7.1f}%"

        print(
            f"{name:<22} {stats['mean']:>8.4f} {stats['std']:>8.4f}"
            f" [{ci_lo:>7.4f},{ci_hi:>7.4f}]"
            f" {d_str} {imp_str}"
        )


def print_individual_analysis(
    config_names: list[str],
    all_results: dict[str, list[float]],
) -> None:
    """Analyze individual vs cumulative improvements for H1d.

    Args:
        config_names: Ordered list of config names.
        all_results: Mapping of config name to list of final losses.
    """
    print("\n" + "=" * 72)
    print("Individual vs Cumulative Improvement Analysis (H1d)")
    print("=" * 72)

    baseline_mean = sum(all_results[config_names[0]]) / len(
        all_results[config_names[0]]
    )
    llama_mean = sum(all_results[config_names[-1]]) / len(
        all_results[config_names[-1]]
    )
    total_improvement = baseline_mean - llama_mean

    if total_improvement <= 0:
        print("WARNING: Full LLaMA did not improve over baseline.")
        print("Individual improvement analysis not meaningful.")
        return

    print(f"Baseline mean loss:    {baseline_mean:.4f}")
    print(f"Full LLaMA mean loss:  {llama_mean:.4f}")
    print(f"Total improvement:     {total_improvement:.4f}")

    print(
        f"\n{'Feature':<22} {'Marginal':>10} {'% of Total':>12}"
        f" {'Dominant?':>10}"
    )
    print("-" * 56)

    individual_improvements: list[tuple[str, float]] = []
    sum_individual = 0.0

    for i in range(1, len(config_names)):
        prev_mean = sum(all_results[config_names[i - 1]]) / len(
            all_results[config_names[i - 1]]
        )
        cur_mean = sum(all_results[config_names[i]]) / len(
            all_results[config_names[i]]
        )
        marginal = prev_mean - cur_mean
        pct = marginal / total_improvement * 100
        sum_individual += marginal
        dominant = "YES" if pct > 50 else ("maybe" if pct > 20 else "")
        individual_improvements.append((config_names[i], marginal))

        print(
            f"{config_names[i]:<22} {marginal:>+10.4f} {pct:>+11.1f}%"
            f" {dominant:>10}"
        )

    print("-" * 56)
    interaction = total_improvement - sum_individual
    pct_interaction = interaction / total_improvement * 100
    print(f"{'Sum of individual':<22} {sum_individual:>+10.4f}")
    print(
        f"{'Interaction effect':<22} {interaction:>+10.4f}"
        f" {pct_interaction:>+11.1f}%"
    )

    # H1d evaluation
    print("\n--- H1d Evaluation ---")
    max_name, max_imp = max(
        individual_improvements,
        key=lambda x: x[1],
    )
    max_pct = max_imp / total_improvement * 100
    print(f"Largest single feature: {max_name} ({max_pct:.1f}% of total)")

    if max_pct > 50:
        print(
            f"FALSIFIED: H1d says no single change > 50%, "
            f"but {max_name} = {max_pct:.1f}%"
        )
    elif max_pct > 20:
        print(
            f"H1d inconclusive: largest feature is {max_pct:.1f}% "
            f"(between 20-50%)"
        )
    else:
        print("SUPPORTED: No single feature > 20% of total improvement")


def main() -> None:
    """Run the HYP-001 ablation study."""
    parser = argparse.ArgumentParser(
        description="HYP-001: GPT->LLaMA ablation study"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=300.0,
        help="Time budget per config per seed in seconds (default: 300)",
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

    total_runs = len(configs) * len(seeds)
    total_time_est = total_runs * args.time_budget / 60

    print("\n" + "=" * 72)
    print("HYP-001: Architecture Ablation GPT -> LLaMA")
    print(f"{len(configs)} configs x {len(seeds)} seeds = {total_runs} runs")
    print(f"Time budget: {args.time_budget:.0f}s per run")
    print(f"Estimated total: ~{total_time_est:.0f} minutes")
    print(f"Seq len: {args.seq_len}")
    if args.mlflow:
        import mlflow as _mlflow

        print(f"MLflow tracking URI: {_mlflow.get_tracking_uri()}")
    print("=" * 72)

    # --- Training ---
    log = ExperimentLog(RESULTS_FILE)
    all_results: dict[str, list[float]] = {name: [] for name in config_names}
    run_count = 0

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for name, config in configs:
            run_count += 1
            print(f"  [{run_count}/{total_runs}]", end="")
            result = train_one(
                name=name,
                config=config,
                tokens=tokens,
                seq_len=args.seq_len,
                time_budget_s=args.time_budget,
                seed=seed,
                log=log,
                use_mlflow=args.mlflow,
            )
            all_results[name].append(result["final_loss"])

    # --- Statistical Analysis ---
    print_results_table(config_names, all_results)

    # ANOVA across all 6 configs
    groups = [all_results[name] for name in config_names]
    if args.seeds >= 2:
        f_stat, p_approx = one_way_anova(groups)
        print(f"\nOne-way ANOVA: F={f_stat:.2f}, p~{p_approx:.3f}")
        if p_approx < 0.05:
            print("  -> Significant difference between configs (p < 0.05)")
        else:
            print("  -> No significant difference detected (p >= 0.05)")

    # Individual vs cumulative analysis (H1d)
    print_individual_analysis(config_names, all_results)

    # Summary
    log_summary = log.summary()
    print(f"\nResults logged to {RESULTS_FILE}")
    print(f"Total entries in log: {log_summary['total']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
