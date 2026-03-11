"""Train using TextDataset and TokenDataset classes.

Demonstrates the dataset abstractions in lmxlab.data:

- TextDataset: takes raw text + tokenizer, handles tokenization
- TokenDataset: wraps pre-tokenized arrays
- Both provide __len__ and __getitem__ for windowed training pairs

Also shows how to build a simple batched iterator from a dataset.

Usage:
    uv run python recipes/train_with_datasets.py
    uv run python recipes/train_with_datasets.py --seq-len 64 --steps 300
"""

import argparse
from dataclasses import replace

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.dataset import TextDataset, TokenDataset
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.gpt import gpt_tiny
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


def train_model(
    name: str,
    tokens: mx.array,
    tokenizer: CharTokenizer,
    seq_len: int,
    max_steps: int,
) -> list[float]:
    """Train a tiny GPT and return loss history."""
    config = gpt_tiny()
    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = LanguageModel(config)
    mx.eval(model.parameters())

    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=max_steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
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
    losses = [m["loss"] for m in history]

    # Generate sample
    prompt = mx.array([tokenizer.encode("To be")])
    output = generate(model, prompt, max_tokens=60, temperature=0.8)
    sample = tokenizer.decode(output[0].tolist())

    return losses, sample


def main() -> None:
    """Compare TextDataset and TokenDataset approaches."""
    parser = argparse.ArgumentParser(description="Dataset classes demo")
    parser.add_argument(
        "--seq-len", type=int, default=32, help="Sequence length"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Training steps"
    )
    args = parser.parse_args()

    mx.random.seed(42)
    tokenizer = CharTokenizer(TEXT)

    # --- Approach 1: TextDataset ---
    print("=== TextDataset ===")
    print("Handles tokenization internally from raw text.\n")

    text_ds = TextDataset(TEXT, tokenizer, seq_len=args.seq_len)
    print(f"  Text length: {len(TEXT)} chars")
    print(f"  Token count: {len(text_ds.tokens)}")
    print(f"  Windows available: {len(text_ds)}")

    # Show a sample window
    x, y = text_ds[0]
    print(f"  Sample input:  {tokenizer.decode(x.tolist()[:20])}...")
    print(f"  Sample target: {tokenizer.decode(y.tolist()[:20])}...")

    losses_text, sample_text = train_model(
        "TextDataset",
        text_ds.tokens,
        tokenizer,
        args.seq_len,
        args.steps,
    )

    # --- Approach 2: TokenDataset ---
    print("\n=== TokenDataset ===")
    print("Wraps pre-tokenized arrays (useful for cached tokens).\n")

    # Pre-tokenize (simulating cached/preprocessed data)
    raw_tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    token_ds = TokenDataset(raw_tokens, seq_len=args.seq_len)
    print(f"  Token count: {len(token_ds.tokens)}")
    print(f"  Windows available: {len(token_ds)}")

    # Verify same data
    x2, y2 = token_ds[0]
    assert mx.array_equal(x, x2), "Datasets should produce same windows"
    print("  Verified: same windows as TextDataset")

    losses_token, sample_token = train_model(
        "TokenDataset",
        token_ds.tokens,
        tokenizer,
        args.seq_len,
        args.steps,
    )

    # --- Comparison ---
    print("\n=== Results ===\n")
    print(f"{'Dataset':<15} {'Initial':<10} {'Final':<10}")
    print("-" * 35)
    print(
        f"{'TextDataset':<15} {losses_text[0]:<10.4f} {losses_text[-1]:<10.4f}"
    )
    print(
        f"{'TokenDataset':<15} "
        f"{losses_token[0]:<10.4f} "
        f"{losses_token[-1]:<10.4f}"
    )

    print(f"\nTextDataset sample:  {sample_text}")
    print(f"TokenDataset sample: {sample_token}")

    print(
        "\nBoth datasets produce identical training data. TextDataset is "
        "convenient\nfor raw text; TokenDataset is useful when tokens are "
        "pre-computed or cached."
    )


if __name__ == "__main__":
    main()
