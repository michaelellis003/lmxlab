"""Compare training dynamics across architectures.

Trains tiny versions of GPT, LLaMA, and DeepSeek on the same
data and compares their loss curves. Demonstrates how different
architectural choices affect learning behavior — all using the
same LanguageModel class with different configs.

Usage:
    uv run python recipes/compare_training.py
"""

from dataclasses import replace

import mlx.core as mx

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.deepseek import deepseek_tiny
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.models.llama import llama_tiny
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer


def train_one(
    name: str,
    config,
    tokens: mx.array,
    train_config: TrainConfig,
) -> list[float]:
    """Train a model and return loss history."""
    model = LanguageModel(config)
    mx.eval(model.parameters())
    params = model.count_parameters()
    print(f"  {name}: {params:,} parameters")

    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens,
            batch_size=train_config.batch_size,
            seq_len=32,
            shuffle=True,
        )

    history = trainer.train(data_iter())
    losses = [m["loss"] for m in history]
    return losses


def main() -> None:
    """Compare training across architectures."""
    mx.random.seed(42)

    # --- Shared data ---
    text = (
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
    ) * 3  # Repeat for more data

    tokenizer = CharTokenizer(text)
    tokens = mx.array(tokenizer.encode(text), dtype=mx.int32)
    print(f"Data: {len(tokens)} tokens, vocab={tokenizer.vocab_size}")

    # --- Configs ---
    architectures = {
        "GPT (MHA+LayerNorm)": gpt_tiny(),
        "LLaMA (GQA+RMSNorm)": llama_tiny(),
        "DeepSeek (MLA)": deepseek_tiny(),
    }

    # Match vocab size
    architectures = {
        name: replace(cfg, vocab_size=tokenizer.vocab_size)
        for name, cfg in architectures.items()
    }

    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=100,
        batch_size=4,
        compile_step=False,
        warmup_steps=5,
    )

    # --- Train each ---
    print("\nTraining...")
    results = {}
    for name, config in architectures.items():
        mx.random.seed(42)  # Same init seed
        losses = train_one(name, config, tokens, train_config)
        results[name] = losses

    # --- Compare ---
    print("\n" + "=" * 55)
    print("Loss Comparison (every 10 steps)")
    print("=" * 55)

    header = f"{'Step':>5}"
    for name in results:
        short = name.split("(")[0].strip()
        header += f"  {short:>12}"
    print(header)
    print("-" * len(header))

    n_steps = min(len(v) for v in results.values())
    for step in range(0, n_steps, 10):
        row = f"{step + 1:>5}"
        for losses in results.values():
            row += f"  {losses[step]:>12.4f}"
        print(row)

    # Final losses
    print("-" * len(header))
    row = f"{'Final':>5}"
    for losses in results.values():
        row += f"  {losses[-1]:>12.4f}"
    print(row)

    # Summary
    print("\n" + "=" * 55)
    print("Key Takeaways:")
    best = min(results, key=lambda k: results[k][-1])
    print(f"  Lowest final loss: {best}")
    print("  Same LanguageModel class, different configs.")
    print("  Architectural choices directly affect convergence.")
    print("=" * 55)


if __name__ == "__main__":
    main()
