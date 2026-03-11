"""Analyze experiment results with statistical tools.

Demonstrates the analysis utilities from the experiment framework:

- compare_experiments: rank experiments by metric
- compute_statistics: mean, std, min, max
- cohens_d: effect size between two groups
- confidence_interval: 95% CI for the mean
- simplicity_score: metric improvement weighted by param efficiency

Generates synthetic experiment data so this recipe runs standalone
without needing prior training runs.

Usage:
    uv run python recipes/analyze_experiments.py
"""

from lmxlab.experiments.analysis import (
    cohens_d,
    compare_experiments,
    compute_statistics,
    confidence_interval,
    simplicity_score,
)
from lmxlab.experiments.tracking import ExperimentLog, LogEntry


def create_synthetic_log(log: ExperimentLog) -> None:
    """Populate an experiment log with synthetic multi-seed results.

    Simulates two architectures (GPT and LLaMA) each run with
    5 seeds, where LLaMA has lower loss and fewer parameters.
    """
    import random

    rng = random.Random(42)

    # GPT baseline: higher loss, more parameters
    for seed in range(5):
        noise = rng.gauss(0, 0.05)
        log.log(
            LogEntry(
                experiment="gpt-tiny",
                val_loss=2.8 + noise,
                val_bpb=2.8 + noise,
                train_loss=2.6 + noise,
                param_count=50_000,
                wall_time_s=12.0 + rng.uniform(-1, 1),
                description=f"GPT tiny, seed={seed + 42}",
                seed=seed + 42,
                status="keep",
            )
        )

    # LLaMA: lower loss, fewer parameters (the "better" model)
    for seed in range(5):
        noise = rng.gauss(0, 0.04)
        log.log(
            LogEntry(
                experiment="llama-tiny",
                val_loss=2.4 + noise,
                val_bpb=2.4 + noise,
                train_loss=2.2 + noise,
                param_count=42_000,
                wall_time_s=14.0 + rng.uniform(-1, 1),
                description=f"LLaMA tiny, seed={seed + 42}",
                seed=seed + 42,
                status="keep",
            )
        )

    # A crashed run (excluded from analysis)
    log.log(
        LogEntry(
            experiment="mixtral-tiny",
            val_loss=float("inf"),
            status="crash",
            description="OOM on MoE routing",
        )
    )


def main() -> None:
    """Run the full analysis pipeline on synthetic data."""
    import tempfile
    from pathlib import Path

    # Create a temporary experiment log
    tmp_dir = Path(tempfile.mkdtemp())
    log_path = tmp_dir / "results.jsonl"
    log = ExperimentLog(log_path)

    print("=== Generating synthetic experiment data ===\n")
    create_synthetic_log(log)

    entries = log.load()
    summary = log.summary()
    print(
        f"Logged {summary['total']} runs: "
        f"{summary['kept']} kept, "
        f"{summary['crashed']} crashed\n"
    )

    # --- 1. Compare experiments ---
    print("=== 1. Compare Experiments (ranked by val_bpb) ===\n")
    ranked = compare_experiments(log, metric="val_bpb")
    print(f"{'Experiment':<16} {'val_bpb':<10} {'Params':<10} {'Time':<8}")
    print("-" * 48)
    for r in ranked:
        print(
            f"{r['experiment']:<16} "
            f"{r['val_bpb']:<10.4f} "
            f"{r['param_count']:<10,} "
            f"{r['wall_time_s']:<8.1f}"
        )

    # --- 2. Per-group statistics ---
    print("\n=== 2. Per-Group Statistics ===\n")
    kept = [e for e in entries if e.status == "keep"]
    groups: dict[str, list[float]] = {}
    for e in kept:
        groups.setdefault(e.experiment, []).append(e.val_bpb)

    for name, values in sorted(groups.items()):
        stats = compute_statistics(values)
        print(
            f"{name:<16} "
            f"mean={stats['mean']:.4f}  "
            f"std={stats['std']:.4f}  "
            f"min={stats['min']:.4f}  "
            f"max={stats['max']:.4f}  "
            f"n={stats['n']}"
        )

    # --- 3. Confidence intervals ---
    print("\n=== 3. Confidence Intervals (95%) ===\n")
    for name, values in sorted(groups.items()):
        lo, hi = confidence_interval(values, confidence=0.95)
        mean = sum(values) / len(values)
        print(f"{name:<16} {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    # --- 4. Cohen's d effect size ---
    print("\n=== 4. Cohen's d Effect Size ===\n")
    group_names = sorted(groups.keys())
    if len(group_names) >= 2:
        a_name, b_name = group_names[0], group_names[1]
        d = cohens_d(groups[a_name], groups[b_name])
        # Interpret
        if abs(d) < 0.2:
            magnitude = "negligible"
        elif abs(d) < 0.5:
            magnitude = "small"
        elif abs(d) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        direction = f"{a_name} > {b_name}" if d > 0 else f"{b_name} > {a_name}"
        print(
            f"{a_name} vs {b_name}: d = {d:.3f} ({magnitude})\n"
            f"  Interpretation: {direction}"
        )

    # --- 5. Simplicity score ---
    print("\n=== 5. Simplicity Score ===\n")
    print("(Rewards metric improvement * parameter efficiency)\n")

    # Use GPT as baseline
    gpt_entries = [e for e in kept if e.experiment == "gpt-tiny"]
    baseline_bpb = sum(e.val_bpb for e in gpt_entries) / len(gpt_entries)
    baseline_params = gpt_entries[0].param_count

    print(
        f"Baseline: gpt-tiny (val_bpb={baseline_bpb:.4f}, "
        f"params={baseline_params:,})\n"
    )

    for e in kept:
        score = simplicity_score(
            e,
            baseline_params=baseline_params,
            baseline_metric=baseline_bpb,
        )
        print(
            f"  {e.experiment:<16} seed={e.seed:<4} "
            f"val_bpb={e.val_bpb:.4f}  "
            f"params={e.param_count:<8,}  "
            f"score={score:+.4f}"
        )

    print("\n=== Summary ===\n")
    best = log.best(metric="val_bpb")
    if best:
        print(
            f"Best overall: {best.experiment} "
            f"(val_bpb={best.val_bpb:.4f}, seed={best.seed})"
        )

    # Clean up
    log_path.unlink(missing_ok=True)
    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
