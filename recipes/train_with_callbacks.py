"""Train with callbacks: logging, throughput monitoring, early stopping.

Demonstrates the callback system that hooks into the training loop.
Three callbacks run simultaneously:

- MetricsLogger: prints loss and learning rate at intervals
- ThroughputMonitor: reports steps/sec and tokens/sec
- EarlyStopping: halts training when eval loss plateaus

Usage:
    uv run python recipes/train_with_callbacks.py
    uv run python recipes/train_with_callbacks.py --patience 3 --max-steps 300
"""

import argparse
from dataclasses import replace

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.llama import llama_tiny
from lmxlab.training.callbacks import (
    EarlyStopping,
    MetricsLogger,
    ThroughputMonitor,
)
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


def main() -> None:
    """Train a tiny LLaMA with all three callbacks active."""
    parser = argparse.ArgumentParser(description="Train with callbacks demo")
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Max training steps"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (eval intervals without improvement)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Steps between log/throughput reports",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Steps between evaluations",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Data ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    seq_len = 32
    batch_size = 4

    print(f"Vocab: {tokenizer.vocab_size} chars, {len(tokens)} tokens")

    # --- Model ---
    config = llama_tiny()
    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = LanguageModel(config)
    mx.eval(model.parameters())
    print(f"Model: {model.count_parameters():,} parameters")

    # --- Callbacks ---
    # 1) MetricsLogger: prints loss and lr at intervals
    logger = MetricsLogger(log_interval=args.log_interval)

    # 2) ThroughputMonitor: tracks steps/sec and tokens/sec
    throughput = ThroughputMonitor(
        log_interval=args.log_interval,
        tokens_per_step=batch_size * seq_len,
    )

    # 3) EarlyStopping: stops when eval loss plateaus
    early_stop = EarlyStopping(
        patience=args.patience,
        min_delta=0.001,
    )

    print(
        f"\nCallbacks: MetricsLogger (every {args.log_interval} steps), "
        f"ThroughputMonitor, EarlyStopping (patience={args.patience})"
    )

    # --- Training ---
    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.max_steps,
        batch_size=batch_size,
        eval_interval=args.eval_interval,
        compile_step=False,
        warmup_steps=10,
    )

    trainer = Trainer(
        model,
        train_config,
        callbacks=[logger, throughput, early_stop],
    )

    def train_data():
        yield from batch_iterator(
            tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle=True,
        )

    def eval_data():
        yield from batch_iterator(
            tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle=False,
        )

    print(f"\nTraining for up to {args.max_steps} steps...\n")
    history = trainer.train(train_data(), eval_data())

    # --- Summary ---
    if early_stop.should_stop:
        print(f"\nEarly stopping triggered after {len(history)} steps")
    else:
        print(f"\nCompleted all {len(history)} steps")

    if history:
        print(f"Final loss: {history[-1]['loss']:.4f}")

    # --- Generate ---
    print("\nGenerating sample text:")
    prompt = mx.array([tokenizer.encode("To be")])
    output = generate(model, prompt, max_tokens=80, temperature=0.8)
    print(f"  {tokenizer.decode(output[0].tolist())}")


if __name__ == "__main__":
    main()
