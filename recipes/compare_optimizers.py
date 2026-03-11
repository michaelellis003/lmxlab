"""Compare optimizers on unified memory (Pre-registered Experiment 3).

Tests whether Apple Silicon's unified memory architecture changes
which optimizers work best, compared to published CUDA results.

Competing hypotheses:
  H1 (Same story): AdamW dominates regardless of hardware.
  H2 (Memory-efficient wins): SGD/Adafactor do comparatively
     better because less optimizer state = more room for batches.
  H3 (Bandwidth matters): SGD gains disproportionate advantage
     because fewer memory accesses per step.

Protocol: Train LLaMA-small on Shakespeare with each optimizer
across a learning rate sweep. Fixed step budget, multiple seeds.

Usage:
    uv run python recipes/compare_optimizers.py
    uv run python recipes/compare_optimizers.py --steps 300 --seeds 3
"""

import argparse
import time

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.experiments.analysis import (
    compare_experiments,
    compute_statistics,
)
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner
from lmxlab.experiments.tracking import ExperimentLog
from lmxlab.models.base import LanguageModel
from lmxlab.models.llama import llama_config
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

TEXT = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles, "
    "And by opposing end them. To die, to sleep; "
    "No more; and by a sleep to say we end "
    "The heart-ache and the thousand natural shocks "
    "That flesh is heir to: 'tis a consummation "
    "Devoutly to be wish'd. To die, to sleep; "
    "To sleep, perchance to dream. "
    "Ay, there's the rub; For in that sleep of death "
    "what dreams may come When we have shuffled off "
    "this mortal coil, Must give us pause. "
) * 10

# Optimizers and their learning rate sweeps (log-scale)
OPTIMIZERS = {
    "adamw": [1e-4, 3e-4, 1e-3, 3e-3],
    "sgd": [1e-3, 3e-3, 1e-2, 3e-2],
    "adafactor": [1e-4, 3e-4, 1e-3, 3e-3],
    "lion": [1e-5, 3e-5, 1e-4, 3e-4],
}


def run_trial(
    optimizer_name: str,
    lr: float,
    seed: int,
    tokens: mx.array,
    vocab_size: int,
    max_steps: int,
    d_model: int,
    n_layers: int,
) -> dict:
    """Run a single optimizer trial."""
    mx.random.seed(seed)

    model_config = llama_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=max(2, d_model // 32),
        n_kv_heads=max(1, d_model // 64),
        n_layers=n_layers,
        d_ff=d_model * 2,
        max_seq_len=128,
        tie_embeddings=True,
    )

    model = LanguageModel(model_config)
    mx.eval(model.parameters())
    param_count = model.count_parameters()

    train_config = TrainConfig(
        learning_rate=lr,
        max_steps=max_steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        optimizer=optimizer_name,
        lr_schedule="cosine",
    )

    trainer = Trainer(model, train_config)

    exp_config = ExperimentConfig(
        name=f"opt-{optimizer_name}-lr{lr:.0e}-s{seed}",
        description=(f"{optimizer_name} lr={lr:.1e} seed={seed}"),
        time_budget_s=300.0,
        seed=seed,
    )
    runner = ExperimentRunner(exp_config)
    runner.start()

    def data_iter():
        yield from batch_iterator(
            tokens,
            batch_size=4,
            seq_len=32,
            shuffle=True,
        )

    start = time.perf_counter()

    try:
        history = trainer.train(data_iter())
        elapsed = time.perf_counter() - start
        final_loss = history[-1]["loss"] if history else float("inf")
        steps_done = len(history)
        steps_per_sec = steps_done / elapsed if elapsed > 0 else 0

        entry = runner.finish(
            metrics={
                "val_loss": final_loss,
                "train_loss": final_loss,
                "steps": steps_done,
                "steps_per_sec": steps_per_sec,
            },
            param_count=param_count,
            config_dict={
                "optimizer": optimizer_name,
                "lr": lr,
                "d_model": d_model,
                "n_layers": n_layers,
                "seed": seed,
            },
            status="keep",
        )

        return {
            "optimizer": optimizer_name,
            "lr": lr,
            "seed": seed,
            "loss": final_loss,
            "steps_per_sec": steps_per_sec,
            "params": param_count,
            "time": entry.wall_time_s,
        }

    except Exception as e:
        runner.finish(
            metrics={"error": str(e)},
            param_count=param_count,
            status="crash",
        )
        return {
            "optimizer": optimizer_name,
            "lr": lr,
            "seed": seed,
            "loss": float("inf"),
            "steps_per_sec": 0,
            "params": param_count,
            "time": 0,
            "error": str(e),
        }


def main() -> None:
    """Run optimizer comparison experiment."""
    parser = argparse.ArgumentParser(
        description="Optimizer comparison (Experiment 3)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Training steps per trial",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds per configuration",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model dimension",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=list(OPTIMIZERS.keys()),
        choices=list(OPTIMIZERS.keys()),
        help="Optimizers to compare",
    )
    args = parser.parse_args()

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    vocab = tokenizer.vocab_size

    print("=== Optimizer Comparison (Experiment 3) ===")
    print(f"Model: LLaMA d={args.d_model}, L={args.n_layers}")
    print(f"Data: {len(tokens)} tokens, vocab={vocab}")
    print(f"Steps: {args.steps}, Seeds: {args.seeds}")
    print(f"Optimizers: {', '.join(args.optimizers)}\n")

    # --- Run all trials ---
    results = []
    total = sum(len(OPTIMIZERS[o]) * args.seeds for o in args.optimizers)
    trial_num = 0

    for opt_name in args.optimizers:
        lrs = OPTIMIZERS[opt_name]
        print(f"--- {opt_name.upper()} ({len(lrs)} LRs) ---")

        for lr in lrs:
            for seed in range(42, 42 + args.seeds):
                trial_num += 1
                print(
                    f"  [{trial_num}/{total}] lr={lr:.1e} seed={seed}...",
                    end=" ",
                    flush=True,
                )

                result = run_trial(
                    opt_name,
                    lr,
                    seed,
                    tokens,
                    vocab,
                    args.steps,
                    args.d_model,
                    args.n_layers,
                )
                results.append(result)

                if "error" in result:
                    print(f"CRASH: {result['error']}")
                else:
                    print(
                        f"loss={result['loss']:.4f} "
                        f"({result['steps_per_sec']:.0f} steps/s)"
                    )

    # --- Per-optimizer best results ---
    print(f"\n{'=' * 60}")
    print("Best result per optimizer (across LR sweep)")
    print(f"{'=' * 60}")
    print(f"{'Optimizer':<12} {'Best LR':>10} {'Loss':>10} {'Steps/s':>10}")
    print("-" * 45)

    best_per_opt = {}
    for opt_name in args.optimizers:
        opt_results = [r for r in results if r["optimizer"] == opt_name]
        if not opt_results:
            continue

        best = min(opt_results, key=lambda r: r["loss"])
        best_per_opt[opt_name] = best
        print(
            f"{opt_name:<12} {best['lr']:>10.1e} "
            f"{best['loss']:>10.4f} "
            f"{best['steps_per_sec']:>10.0f}"
        )

    # --- Multi-seed statistics (if seeds > 1) ---
    if args.seeds > 1:
        print(f"\n{'=' * 60}")
        print("Statistics across seeds (best LR per optimizer)")
        print(f"{'=' * 60}")

        for opt_name in args.optimizers:
            best_lr = best_per_opt.get(opt_name, {}).get("lr")
            if best_lr is None:
                continue

            seed_losses = [
                r["loss"]
                for r in results
                if r["optimizer"] == opt_name and r["lr"] == best_lr
            ]
            stats = compute_statistics(seed_losses)
            print(
                f"  {opt_name:<12} "
                f"mean={stats['mean']:.4f} "
                f"+/- {stats['std']:.4f} "
                f"(n={stats['n']})"
            )

    # --- Hypothesis evaluation ---
    print(f"\n{'=' * 60}")
    print("Hypothesis evaluation")
    print(f"{'=' * 60}")

    if len(best_per_opt) >= 2:
        sorted_opts = sorted(
            best_per_opt.items(),
            key=lambda x: x[1]["loss"],
        )
        winner = sorted_opts[0]
        runner_up = sorted_opts[1]

        print(f"  Winner: {winner[0]} (loss={winner[1]['loss']:.4f})")
        print(f"  Runner-up: {runner_up[0]} (loss={runner_up[1]['loss']:.4f})")

        gap = runner_up[1]["loss"] - winner[1]["loss"]
        print(f"  Gap: {gap:.4f}")

        if winner[0] == "adamw":
            print("  -> Supports H1 (AdamW dominates)")
        elif winner[0] in ("sgd", "adafactor"):
            print("  -> Supports H2 (memory-efficient wins)")
        else:
            print(f"  -> {winner[0]} wins — novel finding")

        # Check throughput hypothesis (H3)
        if "sgd" in best_per_opt:
            sgd_speed = best_per_opt["sgd"]["steps_per_sec"]
            adamw_speed = best_per_opt.get(
                "adamw",
                {},
            ).get("steps_per_sec", 0)
            if adamw_speed > 0:
                ratio = sgd_speed / adamw_speed
                print(f"\n  Throughput ratio (SGD/AdamW): {ratio:.2f}x")
                if ratio > 1.2:
                    print("  -> Supports H3 (bandwidth advantage for SGD)")
                else:
                    print("  -> Does not support H3 (similar throughput)")

    # --- Show experiment log ---
    log = ExperimentLog("experiments/results.jsonl")
    entries = log.load()
    opt_entries = [
        {"experiment": e.experiment, "val_loss": e.val_loss}
        for e in entries
        if e.experiment.startswith("opt-")
    ]
    if opt_entries:
        print(f"\nLogged {len(opt_entries)} optimizer trials")
        comparison = compare_experiments(log, metric="val_bpb")
        for row in comparison:
            print(f"  {row['experiment']}: val_bpb={row['val_bpb']:.4f}")

    print(
        "\nNote: Run with --seeds 3 for statistical rigor. "
        "Tiny models may not reflect large-scale trends."
    )


if __name__ == "__main__":
    main()
