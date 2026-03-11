"""Run a structured experiment with the experiment framework.

Demonstrates the autoresearch pattern:
- Fixed time budget per experiment
- Git commit tracking for reproducibility
- Results logged to results.jsonl
- Multi-seed runs for statistical rigor
- Automatic keep/discard based on improvement

Usage:
    uv run python recipes/run_experiment.py
    uv run python recipes/run_experiment.py --arch llama --seeds 3
    uv run python recipes/run_experiment.py --show-log
"""

import argparse
from dataclasses import replace

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner
from lmxlab.experiments.tracking import ExperimentLog
from lmxlab.models.base import LanguageModel
from lmxlab.models.deepseek import deepseek_tiny
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.llama import llama_tiny
from lmxlab.models.qwen import qwen_tiny
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

ARCH_FACTORIES = {
    "gpt": gpt_tiny,
    "llama": llama_tiny,
    "qwen": qwen_tiny,
    "deepseek": deepseek_tiny,
}

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
) * 5


def run_single(
    arch_name: str,
    seed: int,
    max_steps: int,
    time_budget: float,
) -> None:
    """Run a single experiment with the given config."""
    factory = ARCH_FACTORIES[arch_name]

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)

    model_config = factory()
    model_config = replace(model_config, vocab_size=tokenizer.vocab_size)

    exp_config = ExperimentConfig(
        name=f"{arch_name}-tiny",
        description=(
            f"{arch_name} tiny model, {max_steps} steps, seed={seed}"
        ),
        time_budget_s=time_budget,
        seed=seed,
        output_dir="experiments",
    )

    runner = ExperimentRunner(exp_config)
    runner.start()

    model = LanguageModel(model_config)
    mx.eval(model.parameters())
    param_count = model.count_parameters()

    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=max_steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=5,
    )

    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens,
            batch_size=4,
            seq_len=32,
            shuffle=True,
        )

    try:
        history = trainer.train(data_iter())
        final_loss = history[-1]["loss"] if history else float("inf")

        # Check against previous best
        log = ExperimentLog("experiments/results.jsonl")
        prev_best = log.best(metric="val_loss")
        if prev_best and prev_best.val_loss > 0:
            improved = final_loss < prev_best.val_loss
            status = "keep" if improved else "discard"
        else:
            status = "keep"

        entry = runner.finish(
            metrics={
                "val_loss": final_loss,
                "train_loss": final_loss,
                "steps": len(history),
            },
            param_count=param_count,
            config_dict={
                "arch": arch_name,
                "d_model": model_config.block.d_model,
                "n_heads": model_config.block.n_heads,
                "n_layers": model_config.n_layers,
                "attention": model_config.block.attention,
                "lr": train_config.learning_rate,
                "max_steps": max_steps,
            },
            status=status,
        )

        print(
            f"  [{entry.status:>7}] {arch_name} seed={seed}: "
            f"loss={final_loss:.4f}, "
            f"params={param_count:,}, "
            f"time={entry.wall_time_s:.1f}s"
        )

    except Exception as e:
        runner.finish(
            metrics={"error": str(e)},
            param_count=param_count,
            status="crash",
        )
        print(f"  [CRASH] {arch_name} seed={seed}: {e}")


def show_log() -> None:
    """Display the experiment log."""
    log = ExperimentLog("experiments/results.jsonl")
    entries = log.load()
    if not entries:
        print("No experiments logged yet.")
        return

    summary = log.summary()
    print(f"Total experiments: {summary['total']}")
    print(
        f"  Kept: {summary['kept']}, "
        f"Discarded: {summary['discarded']}, "
        f"Crashed: {summary['crashed']}"
    )
    print(f"  Best val_bpb: {summary.get('best_val_bpb', 'N/A')}")

    print(
        f"\n{'Experiment':<20} {'Status':<10} "
        f"{'Loss':<10} {'Params':<12} {'Time':<8}"
    )
    print("-" * 65)
    for e in entries[-20:]:  # Show last 20
        print(
            f"{e.experiment:<20} {e.status:<10} "
            f"{e.val_loss:<10.4f} {e.param_count:<12,} "
            f"{e.wall_time_s:<8.1f}"
        )


def main() -> None:
    """Run experiments or show log."""
    parser = argparse.ArgumentParser(description="Run structured experiments")
    parser.add_argument(
        "--arch",
        default="all",
        choices=["all"] + list(ARCH_FACTORIES),
        help="Architecture to test",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds to run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Max training steps per run",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=60.0,
        help="Time budget per run (seconds)",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show experiment log and exit",
    )
    args = parser.parse_args()

    if args.show_log:
        show_log()
        return

    archs = list(ARCH_FACTORIES) if args.arch == "all" else [args.arch]

    print(
        f"Running {len(archs)} architecture(s) "
        f"x {args.seeds} seed(s) "
        f"x {args.steps} steps"
    )
    print(f"Time budget: {args.time_budget}s per run\n")

    for arch in archs:
        for seed in range(args.seeds):
            run_single(
                arch,
                seed=seed + 42,
                max_steps=args.steps,
                time_budget=args.time_budget,
            )

    print("\nResults:")
    show_log()


if __name__ == "__main__":
    main()
