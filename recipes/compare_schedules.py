"""Compare learning rate schedules and optimizers.

Trains the same model with different LR schedule and optimizer
combinations to show their effect on convergence. Demonstrates:

- Schedules: cosine, linear, constant (all with warmup)
- Optimizers: AdamW, Lion, Adafactor
- How to use TrainConfig to switch between them

Usage:
    uv run python recipes/compare_schedules.py
    uv run python recipes/compare_schedules.py \
        --steps 300 --optimizers adamw lion
"""

import argparse
from dataclasses import replace

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.llama import llama_tiny
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
) * 5


def run_training(
    tokenizer: CharTokenizer,
    tokens: mx.array,
    optimizer: str,
    schedule: str,
    max_steps: int,
    seq_len: int,
) -> list[float]:
    """Train with a specific optimizer/schedule and return losses."""
    mx.random.seed(42)

    config = llama_tiny()
    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = LanguageModel(config)
    mx.eval(model.parameters())

    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=max_steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        optimizer=optimizer,
        lr_schedule=schedule,
    )

    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens,
            batch_size=4,
            seq_len=seq_len,
            shuffle=True,
        )

    history = trainer.train(data_iter())
    return [m["loss"] for m in history]


def main() -> None:
    """Compare optimizer and schedule combinations."""
    parser = argparse.ArgumentParser(
        description="Compare LR schedules and optimizers"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Training steps"
    )
    parser.add_argument(
        "--seq-len", type=int, default=32, help="Sequence length"
    )
    parser.add_argument(
        "--schedules",
        nargs="+",
        default=["cosine", "linear", "constant"],
        choices=["cosine", "linear", "constant"],
        help="Schedules to compare",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["adamw"],
        choices=["adamw", "lion", "adafactor"],
        help="Optimizers to compare",
    )
    args = parser.parse_args()

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)

    combos = [
        (opt, sched) for opt in args.optimizers for sched in args.schedules
    ]

    print(
        f"Comparing {len(combos)} combinations: "
        f"{len(args.optimizers)} optimizer(s) x "
        f"{len(args.schedules)} schedule(s)\n"
    )

    results = {}
    for opt, sched in combos:
        label = f"{opt}/{sched}"
        print(f"Training: {label}...", end=" ", flush=True)
        losses = run_training(
            tokenizer,
            tokens,
            opt,
            sched,
            args.steps,
            args.seq_len,
        )
        results[label] = losses
        print(f"final loss = {losses[-1]:.4f}")

    # --- Results table ---
    print(f"\n{'Combination':<22} {'Initial':<10} {'Final':<10} {'Best':<10}")
    print("-" * 52)

    ranked = sorted(results.items(), key=lambda kv: kv[1][-1])
    for label, losses in ranked:
        print(
            f"{label:<22} "
            f"{losses[0]:<10.4f} "
            f"{losses[-1]:<10.4f} "
            f"{min(losses):<10.4f}"
        )

    # --- Loss at checkpoints ---
    checkpoints = [
        0,
        len(ranked[0][1]) // 4,
        len(ranked[0][1]) // 2,
        3 * len(ranked[0][1]) // 4,
        len(ranked[0][1]) - 1,
    ]

    print(f"\n{'Step →':<22}", end="")
    for cp in checkpoints:
        print(f"{cp + 1:<10}", end="")
    print()
    print("-" * (22 + 10 * len(checkpoints)))

    for label, losses in ranked:
        print(f"{label:<22}", end="")
        for cp in checkpoints:
            if cp < len(losses):
                print(f"{losses[cp]:<10.4f}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()

    winner = ranked[0][0]
    print(f"\nBest final loss: {winner} ({ranked[0][1][-1]:.4f})")

    print(
        "\nNote: Results on tiny data are noisy. On real tasks, cosine "
        "decay\ntypically outperforms constant LR, and AdamW is a strong "
        "default."
    )


if __name__ == "__main__":
    main()
