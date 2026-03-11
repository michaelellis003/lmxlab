"""Train a Qwen 3.5-style hybrid DeltaNet model.

Demonstrates hybrid attention: Gated DeltaNet (linear attention)
interleaved with standard GQA (softmax attention). The 3:1 ratio
gives efficient long-context processing from DeltaNet plus global
context modeling from periodic full attention layers.

Compares:
1. Pure GQA (LLaMA-style, all softmax attention)
2. Hybrid DeltaNet (Qwen 3.5-style, 75% DeltaNet + 25% GQA)

Usage:
    uv run python recipes/train_deltanet.py
    uv run python recipes/train_deltanet.py --steps 300 --seq-len 64
"""

import argparse
from dataclasses import replace

import mlx.core as mx
import mlx.utils

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.generate import generate
from lmt_metal.models.llama import llama_tiny
from lmt_metal.models.qwen35 import qwen35_tiny
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer

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


def count_params(model: LanguageModel) -> int:
    """Count trainable parameters."""
    leaves = mlx.utils.tree_flatten(model.parameters())
    return sum(p.size for _, p in leaves)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid DeltaNet training")
    parser.add_argument(
        "--steps", type=int, default=300, help="Training steps"
    )
    parser.add_argument(
        "--seq-len", type=int, default=32, help="Sequence length"
    )
    args = parser.parse_args()

    mx.random.seed(42)

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    print(f"Vocab: {tokenizer.vocab_size}, Tokens: {len(tokens)}")

    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        log_interval=100,
    )

    def data_iter():
        yield from batch_iterator(
            tokens,
            batch_size=4,
            seq_len=args.seq_len,
            shuffle=True,
        )

    # ── Experiment 1: Pure GQA (LLaMA-style) ──
    print(f"\n{'=' * 55}")
    print("Experiment 1: Pure GQA (LLaMA-style)")
    print(f"{'=' * 55}")

    gqa_config = replace(llama_tiny(), vocab_size=tokenizer.vocab_size)
    mx.random.seed(42)
    gqa_model = LanguageModel(gqa_config)
    mx.eval(gqa_model.parameters())

    gqa_params = count_params(gqa_model)
    print(f"  Parameters: {gqa_params:,}")
    print("  All layers use softmax attention (GQA)")

    trainer = Trainer(gqa_model, train_config)
    gqa_history = trainer.train(data_iter())
    gqa_loss = gqa_history[-1]["loss"] if gqa_history else float("nan")
    print(f"  Final loss: {gqa_loss:.4f}")

    # ── Experiment 2: Hybrid DeltaNet (Qwen 3.5-style) ──
    print(f"\n{'=' * 55}")
    print("Experiment 2: Hybrid DeltaNet (Qwen 3.5-style)")
    print(f"{'=' * 55}")

    dn_config = replace(qwen35_tiny(), vocab_size=tokenizer.vocab_size)
    mx.random.seed(42)
    dn_model = LanguageModel(dn_config)
    mx.eval(dn_model.parameters())

    dn_params = count_params(dn_model)
    print(f"  Parameters: {dn_params:,}")

    # Show layer pattern
    if dn_config.block_configs:
        pattern = [
            "GQA" if bc.attention == "gqa" else "DeltaNet"
            for bc in dn_config.block_configs
        ]
        print(f"  Layer pattern: {' -> '.join(pattern)}")

    trainer = Trainer(dn_model, train_config)
    dn_history = trainer.train(data_iter())
    dn_loss = dn_history[-1]["loss"] if dn_history else float("nan")
    print(f"  Final loss: {dn_loss:.4f}")

    # ── Comparison ──
    print(f"\n{'=' * 55}")
    print("Results")
    print(f"{'=' * 55}")
    print(f"  {'Model':<25s} {'Params':>10s} {'Loss':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10}")
    print(f"  {'GQA (LLaMA)':<25s} {gqa_params:>10,} {gqa_loss:>10.4f}")
    print(f"  {'DeltaNet (Qwen 3.5)':<25s} {dn_params:>10,} {dn_loss:>10.4f}")

    diff = gqa_loss - dn_loss
    if diff > 0:
        print(f"\n  DeltaNet wins by {diff:.4f}")
    elif diff < 0:
        print(f"\n  GQA wins by {-diff:.4f}")
    else:
        print("\n  Tie!")

    print("\n  Note: DeltaNet's advantage grows with longer")
    print("  sequences due to O(d^2) vs O(n^2) complexity.")

    # ── Generation ──
    print("\nGeneration comparison:")
    prompt = mx.array([tokenizer.encode("To be")])

    output = generate(gqa_model, prompt, max_tokens=60, temperature=0.7)
    print(f'  GQA:      "{tokenizer.decode(output[0].tolist())}"')

    output = generate(dn_model, prompt, max_tokens=60, temperature=0.7)
    print(f'  DeltaNet: "{tokenizer.decode(output[0].tolist())}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
