"""Quickstart: train a language model from scratch in ~50 lines.

A minimal, annotated recipe showing the full pipeline:
load data → build model → train → evaluate → plot.

Usage:
    uv run python recipes/quickstart.py
"""

from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_tiny
from lmxlab.training.callbacks import MetricsLogger
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

# ── Step 1: Prepare data ─────────────────────────────────
TEXT = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles. "
) * 20  # Repeat for more training data

tokenizer = CharTokenizer(TEXT)
tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Total tokens: {len(tokens)}")

# Split 90/10 for train/val
split = int(0.9 * len(tokens))
train_tokens = tokens[:split]
val_tokens = tokens[split:]

# ── Step 2: Build model ──────────────────────────────────
config = gpt_tiny()
# Match vocab size to our tokenizer
config = replace(config, vocab_size=tokenizer.vocab_size)
model = LanguageModel(config)
mx.eval(model.parameters())
print(f"Model: GPT-tiny ({model.count_parameters():,} params)")

# ── Step 3: Train ────────────────────────────────────────
train_config = TrainConfig(
    learning_rate=1e-3,
    max_steps=200,
    batch_size=4,
    warmup_steps=10,
    compile_step=True,
)

logger = MetricsLogger(log_interval=50)
trainer = Trainer(model, train_config, callbacks=[logger])


def make_train_data():
    """Yield training batches."""
    yield from batch_iterator(
        train_tokens,
        batch_size=4,
        seq_len=32,
        shuffle=True,
    )


history = trainer.train(make_train_data())

# ── Step 4: Evaluate ─────────────────────────────────────
model.eval()
total_loss = 0.0
n = 0
for x, y in batch_iterator(
    val_tokens, batch_size=4, seq_len=32, shuffle=False
):
    logits, _ = model(x)
    logits = logits.reshape(-1, logits.shape[-1])
    loss = nn.losses.cross_entropy(logits, y.reshape(-1), reduction="mean")
    mx.eval(loss)
    total_loss += loss.item()
    n += 1

val_loss = total_loss / max(n, 1)
train_loss = history[-1]["loss"] if history else float("inf")

print(f"\nTrain loss: {train_loss:.4f}")
print(f"Val loss:   {val_loss:.4f}")

# ── Step 5: Plot (optional) ──────────────────────────────
try:
    from lmxlab.analysis.plotting import plot_loss_curves

    train_losses = [h["loss"] for h in history]
    fig = plot_loss_curves(train_losses)
    fig.savefig("quickstart_loss.png", dpi=100)
    print("Loss curve saved to quickstart_loss.png")
except ImportError:
    print("(matplotlib not installed — skipping plot)")
