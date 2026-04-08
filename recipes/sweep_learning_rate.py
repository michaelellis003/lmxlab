"""Hyperparameter sweep: find the best learning rate.

Grid search over learning rates (and optionally model dimensions).
Each config is trained and results logged to experiments/results.jsonl.

Usage:
    uv run python recipes/sweep_learning_rate.py
    uv run python recipes/sweep_learning_rate.py --steps 200
    uv run python recipes/sweep_learning_rate.py --random --trials 8
"""

import argparse

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner
from lmxlab.experiments.sweep import grid_sweep, random_sweep
from lmxlab.experiments.tracking import ExperimentLog
from lmxlab.models.base import LanguageModel
from lmxlab.models.llama import llama_config
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. "
    "The only thing we have to fear is fear itself. "
    "In the middle of difficulty lies opportunity. "
    "Life is what happens when you're busy making other plans. "
    "The best time to plant a tree was twenty years ago. "
    "The second best time is now. "
    "Not all who wander are lost. "
) * 10


def run_trial(
    trial_params: dict,
    tokens: mx.array,
    vocab_size: int,
    max_steps: int,
    trial_idx: int,
) -> dict:
    """Run a single training trial."""
    lr = trial_params.get("lr", 1e-3)
    d_model = int(trial_params.get("d_model", 64))
    n_layers = int(trial_params.get("n_layers", 2))

    config = llama_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=max(2, d_model // 32),
        n_kv_heads=max(1, d_model // 64),
        n_layers=n_layers,
        d_ff=d_model * 2,
        max_seq_len=128,
        tie_embeddings=True,
    )

    mx.random.seed(42)
    model = LanguageModel(config)
    mx.eval(model.parameters())
    params = model.count_parameters()

    train_config = TrainConfig(
        learning_rate=lr,
        max_steps=max_steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=5,
    )

    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens, batch_size=4, seq_len=32, shuffle=True
        )

    exp_config = ExperimentConfig(
        name=f"sweep-{trial_idx}",
        description=(f"lr={lr:.1e}, d_model={d_model}, n_layers={n_layers}"),
        time_budget_s=120.0,
        seed=42,
    )

    runner = ExperimentRunner(exp_config)
    runner.start()

    try:
        history = trainer.train(data_iter())
        final_loss = history[-1]["loss"] if history else float("inf")

        entry = runner.finish(
            metrics={
                "val_loss": final_loss,
                "train_loss": final_loss,
                "steps": len(history),
            },
            param_count=params,
            config_dict={
                "lr": lr,
                "d_model": d_model,
                "n_layers": n_layers,
                "max_steps": max_steps,
            },
            status="keep",
        )

        return {
            "trial": trial_idx,
            "lr": lr,
            "d_model": d_model,
            "n_layers": n_layers,
            "loss": final_loss,
            "params": params,
            "time": entry.wall_time_s,
        }

    except Exception as e:
        runner.finish(
            metrics={"error": str(e)},
            param_count=params,
            status="crash",
        )
        return {
            "trial": trial_idx,
            "lr": lr,
            "d_model": d_model,
            "n_layers": n_layers,
            "loss": float("inf"),
            "params": params,
            "time": 0,
            "error": str(e),
        }


def main() -> None:
    """Run learning rate sweep."""
    parser = argparse.ArgumentParser(description="Learning rate sweep")
    parser.add_argument(
        "--steps", type=int, default=100, help="Steps per trial"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random sweep instead of grid",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=6,
        help="Number of random trials",
    )
    args = parser.parse_args()

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    vocab = tokenizer.vocab_size
    print(f"Data: {len(tokens)} tokens, vocab={vocab}")

    if args.random:
        print(f"\nRandom sweep: {args.trials} trials")
        configs = list(
            random_sweep(
                param_ranges={
                    "lr": (1e-4, 5e-3),
                    "d_model": (32, 128),
                    "n_layers": (1, 4),
                },
                n_trials=args.trials,
                log_scale={"lr"},
            )
        )
        # Round d_model to nearest power of 2
        for c in configs:
            c["d_model"] = 2 ** round(__import__("math").log2(c["d_model"]))
            c["n_layers"] = max(1, round(c["n_layers"]))
    else:
        print("\nGrid sweep: lr x d_model")
        configs = list(
            grid_sweep(
                {
                    "lr": [3e-4, 1e-3, 3e-3],
                    "d_model": [64, 128],
                }
            )
        )

    print(f"Total trials: {len(configs)}")
    print(f"Steps per trial: {args.steps}\n")

    # --- Run trials ---
    results = []
    for i, trial_params in enumerate(configs):
        lr = trial_params.get("lr", 1e-3)
        d = int(trial_params.get("d_model", 64))
        nl = int(trial_params.get("n_layers", 2))
        print(
            f"  Trial {i + 1}/{len(configs)}: "
            f"lr={lr:.1e}, d_model={d}, n_layers={nl}...",
            end=" ",
            flush=True,
        )

        result = run_trial(trial_params, tokens, vocab, args.steps, i)
        results.append(result)
        print(f"loss={result['loss']:.4f}")

    # --- Results table ---
    print(
        f"\n{'Trial':>5} {'LR':>10} {'d_model':>8} "
        f"{'Layers':>7} {'Loss':>10} {'Params':>10} {'Time':>8}"
    )
    print("-" * 62)
    for r in sorted(results, key=lambda x: x["loss"]):
        print(
            f"{r['trial']:>5} {r['lr']:>10.1e} "
            f"{r['d_model']:>8} {r['n_layers']:>7} "
            f"{r['loss']:>10.4f} {r['params']:>10,} "
            f"{r['time']:>7.1f}s"
        )

    best = min(results, key=lambda x: x["loss"])
    print(
        f"\nBest: lr={best['lr']:.1e}, d_model={best['d_model']}, "
        f"loss={best['loss']:.4f}"
    )

    # Show experiment log
    log = ExperimentLog("experiments/results.jsonl")
    summary = log.summary()
    print(f"\nExperiment log: {summary['total']} total entries")


if __name__ == "__main__":
    main()
