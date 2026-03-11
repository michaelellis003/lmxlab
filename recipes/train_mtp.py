"""Train a model with Multi-Token Prediction (MTP).

Demonstrates MTP training: the model predicts not just the next
token, but multiple future tokens simultaneously. This provides
richer training signal and can enable speculative decoding.

Compares standard next-token prediction against MTP to show
the effect of auxiliary prediction heads.

Reference: DeepSeek-V3 (arxiv.org/abs/2501.12948), Meta (2024)

Usage:
    uv run python recipes/train_mtp.py
    uv run python recipes/train_mtp.py --steps 300 --n-predict 3
"""

import argparse
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.generate import generate
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.mtp import MultiTokenPrediction
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Token Prediction training"
    )
    parser.add_argument(
        "--steps", type=int, default=300, help="Training steps"
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=2,
        help="Number of future tokens to predict",
    )
    parser.add_argument(
        "--mtp-weight",
        type=float,
        default=0.3,
        help="Weight for MTP auxiliary losses",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)

    print(f"Vocab: {tokenizer.vocab_size}, Tokens: {len(tokens)}")

    # ── Experiment 1: Standard next-token prediction ──
    print(f"\n{'=' * 50}")
    print("Experiment 1: Standard (next-token only)")
    print(f"{'=' * 50}")

    mx.random.seed(42)
    baseline = LanguageModel(config)
    mx.eval(baseline.parameters())

    baseline_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        log_interval=100,
    )
    trainer = Trainer(baseline, baseline_config)

    def data_iter():
        yield from batch_iterator(
            tokens, batch_size=4, seq_len=32, shuffle=True
        )

    baseline_history = trainer.train(data_iter())
    baseline_loss = (
        baseline_history[-1]["loss"] if baseline_history else float("nan")
    )
    print(f"Baseline final loss: {baseline_loss:.4f}")

    # ── Experiment 2: Multi-Token Prediction ──
    print(f"\n{'=' * 50}")
    print(
        f"Experiment 2: MTP (n_predict={args.n_predict}, "
        f"weight={args.mtp_weight})"
    )
    print(f"{'=' * 50}")

    mx.random.seed(42)
    mtp_model = LanguageModel(config)
    mx.eval(mtp_model.parameters())

    mtp = MultiTokenPrediction(
        mtp_model,
        n_predict=args.n_predict,
        mtp_weight=args.mtp_weight,
    )
    mx.eval(mtp.parameters())

    # MTP needs a custom training loop since it computes its
    # own losses internally
    optimizer = mx.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.01)

    def mtp_loss_fn(mtp_module, x, y):
        _, losses = mtp_module(x, y)
        return losses["total_loss"]

    loss_and_grad = nn.value_and_grad(mtp, mtp_loss_fn)

    for step, (x, y) in enumerate(data_iter()):
        if step >= args.steps:
            break

        loss, grads = loss_and_grad(mtp, x, y)
        optimizer.update(mtp, grads)
        mx.eval(loss, mtp.parameters(), optimizer.state)

        if (step + 1) % 100 == 0 or step == 0:
            # Also compute individual loss components
            _, losses = mtp(x, y)
            mx.eval(losses["main_loss"], losses["mtp_loss"])
            print(
                f"  Step {step + 1}: "
                f"total={loss.item():.4f}, "
                f"main={losses['main_loss'].item():.4f}, "
                f"mtp={losses['mtp_loss'].item():.4f}"
            )

    # Final losses
    final_x, final_y = next(iter(data_iter()))
    _, final_losses = mtp(final_x, final_y)
    mx.eval(final_losses["main_loss"])
    mtp_main_loss = final_losses["main_loss"].item()
    print(f"MTP final main loss: {mtp_main_loss:.4f}")

    # ── Comparison ──
    print(f"\n{'=' * 50}")
    print("Results")
    print(f"{'=' * 50}")
    print(f"  Baseline loss:  {baseline_loss:.4f}")
    print(f"  MTP main loss:  {mtp_main_loss:.4f}")

    diff = baseline_loss - mtp_main_loss
    if diff > 0:
        print(f"  MTP is better by {diff:.4f}")
    else:
        print(f"  Baseline is better by {-diff:.4f}")

    # Parameter comparison
    import mlx.utils

    base_params = baseline.count_parameters()
    mtp_total = sum(
        p.size for _, p in mlx.utils.tree_flatten(mtp.parameters())
    )
    print(f"\n  Baseline parameters: {base_params:,}")
    print(f"  MTP total parameters: {mtp_total:,}")
    overhead = mtp_total - base_params
    print(
        f"  MTP overhead: {overhead:,} ({100 * overhead / base_params:.1f}%)"
    )

    # ── Generate from both ──
    print("\nGeneration comparison:")
    prompt = mx.array([tokenizer.encode("To be")])

    output = generate(baseline, prompt, max_tokens=60, temperature=0.7)
    print(f'  Baseline: "{tokenizer.decode(output[0].tolist())}"')

    output = generate(mtp_model, prompt, max_tokens=60, temperature=0.7)
    print(f'  MTP:      "{tokenizer.decode(output[0].tolist())}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
