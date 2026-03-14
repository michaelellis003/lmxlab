"""Train on TinyStories with BPE tokenization.

Uses HuggingFace TinyStories dataset with tiktoken GPT-2 BPE
tokenizer (50257 vocab). Designed for research configs (10M/30M)
that need more data than Shakespeare char-level provides.

Requires: ``uv sync --extra hf --extra tokenizers``

Usage:
    uv run python recipes/tinystories_bpe.py
    uv run python recipes/tinystories_bpe.py --arch llama_10m
    uv run python recipes/tinystories_bpe.py --arch gpt_30m --steps 2000
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from lmxlab.data.dataset import HFDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_10m
from lmxlab.models.llama import llama_10m
from lmxlab.training.callbacks import MetricsLogger
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

ARCH_FACTORIES = {
    "gpt_10m": gpt_10m,
    "llama_10m": llama_10m,
}


def evaluate(model, val_batches):
    """Compute average loss over validation batches."""
    total_loss = 0.0
    n = 0
    for x, y in val_batches:
        logits, _ = model(x)
        logits = logits.reshape(-1, logits.shape[-1])
        loss = nn.losses.cross_entropy(logits, y.reshape(-1), reduction="mean")
        mx.eval(loss)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def main():
    """Train a model on TinyStories BPE."""
    parser = argparse.ArgumentParser(
        description="Train on TinyStories with BPE"
    )
    parser.add_argument(
        "--arch",
        default="gpt_10m",
        choices=list(ARCH_FACTORIES),
        help="Architecture to train (default: gpt_10m)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Training steps (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=10,
        help="Validation batches for eval (default: 10)",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # Tokenizer
    tokenizer = TiktokenTokenizer("gpt2")
    print(f"Tokenizer: GPT-2 BPE (vocab={tokenizer.vocab_size})")

    # Data
    print("Loading TinyStories train split...")
    train_ds = HFDataset(
        name="roneneldan/TinyStories",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        split="train",
    )
    print("Loading TinyStories validation split...")
    val_ds = HFDataset(
        name="roneneldan/TinyStories",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        split="validation",
    )

    # Preload validation batches
    print(f"Caching {args.eval_batches} val batches...")
    val_batches = list(
        val_ds.batch_iterator(
            batch_size=args.batch_size,
            max_batches=args.eval_batches,
        )
    )
    print(f"  {len(val_batches)} val batches ready")

    # Model
    config = ARCH_FACTORIES[args.arch]()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    n_params = model.count_parameters()
    print(f"\nModel: {args.arch} ({n_params:,} params)")

    # Training
    train_config = TrainConfig(
        learning_rate=args.lr,
        max_steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=min(100, args.steps // 10),
        eval_interval=100,
        compile_step=True,
    )

    logger = MetricsLogger(log_interval=50)
    trainer = Trainer(model, train_config, callbacks=[logger])

    # Initial val loss
    val_loss = evaluate(model, val_batches)
    print(f"Initial val_loss: {val_loss:.4f}")

    # Train
    start = time.monotonic()
    print(f"\nTraining for {args.steps} steps...")

    def train_iter():
        yield from train_ds.batch_iterator(
            batch_size=args.batch_size,
            max_batches=args.steps,
        )

    history = trainer.train(train_iter())
    elapsed = time.monotonic() - start

    # Final eval
    final_val_loss = evaluate(model, val_batches)
    train_loss = history[-1]["loss"] if history else float("inf")

    print(f"\n{'=' * 40}")
    print(f"Architecture:   {args.arch}")
    print(f"Parameters:     {n_params:,}")
    print(f"Steps:          {len(history)}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Train loss:     {train_loss:.4f}")
    print(f"Val loss:       {final_val_loss:.4f}")
    print(f"Train-val gap:  {train_loss - final_val_loss:+.4f}")


if __name__ == "__main__":
    main()
