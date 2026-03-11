"""Train a model with Direct Preference Optimization (DPO).

Demonstrates DPO training: first train a base model (SFT), then
fine-tune with preference pairs using DPO. The reference model is
a frozen copy of the SFT model.

This is a toy example with synthetic preferences for illustration.
In practice, you'd use human-labeled preference data.

Usage:
    uv run python recipes/train_dpo.py
    uv run python recipes/train_dpo.py --sft-steps 200 --dpo-steps 100
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
from lmt_metal.training.dpo import dpo_loss
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
)


def make_preference_pairs(
    tokens: mx.array,
    seq_len: int,
    n_pairs: int,
) -> list[tuple[mx.array, mx.array]]:
    """Create synthetic preference pairs for demonstration.

    'Chosen' sequences come from the actual text (more coherent).
    'Rejected' sequences are random permutations (less coherent).
    """
    pairs = []
    n_tokens = len(tokens)
    for i in range(n_pairs):
        start = (i * seq_len) % max(n_tokens - seq_len, 1)
        chosen = tokens[start : start + seq_len]

        # Rejected: shuffle the tokens (destroys coherence)
        perm = mx.random.permutation(seq_len)
        rejected = chosen[perm]

        pairs.append((chosen[None, :], rejected[None, :]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training")
    parser.add_argument(
        "--sft-steps", type=int, default=200, help="SFT training steps"
    )
    parser.add_argument(
        "--dpo-steps", type=int, default=50, help="DPO training steps"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1, help="DPO beta parameter"
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Setup ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)

    print(f"Vocab: {tokenizer.vocab_size}, Tokens: {len(tokens)}")

    # ── Phase 1: SFT (Supervised Fine-Tuning) ──
    print(f"\nPhase 1: SFT for {args.sft_steps} steps...")
    model = LanguageModel(config)
    mx.eval(model.parameters())

    sft_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.sft_steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        log_interval=50,
    )
    trainer = Trainer(model, sft_config)

    def data_iter():
        yield from batch_iterator(
            tokens, batch_size=4, seq_len=32, shuffle=True
        )

    history = trainer.train(data_iter())
    sft_loss = history[-1]["loss"] if history else float("nan")
    print(f"SFT final loss: {sft_loss:.4f}")

    # ── Phase 2: DPO ──
    print(f"\nPhase 2: DPO for {args.dpo_steps} steps (beta={args.beta})...")

    # Freeze a copy as the reference model
    ref_model = LanguageModel(config)
    import mlx.utils

    ref_weights = dict(mlx.utils.tree_flatten(model.parameters()))
    ref_model.load_weights(list(ref_weights.items()))
    mx.eval(ref_model.parameters())
    ref_model.freeze()

    # Create preference pairs
    pairs = make_preference_pairs(tokens, seq_len=32, n_pairs=20)

    # DPO training loop
    dpo_optimizer = mx.optimizers.Adam(learning_rate=1e-4)
    loss_and_grad = nn.value_and_grad(model, dpo_loss)

    for step in range(args.dpo_steps):
        pair = pairs[step % len(pairs)]
        chosen, rejected = pair

        loss, grads = loss_and_grad(
            model, ref_model, chosen, rejected, args.beta
        )
        dpo_optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), dpo_optimizer.state)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  DPO step {step + 1}: loss={loss.item():.4f}")

    # ── Generate comparison ──
    print("\nGeneration comparison:")
    prompt = mx.array([tokenizer.encode("To be")])

    # Generate from DPO-tuned model
    output = generate(model, prompt, max_tokens=60, temperature=0.7)
    dpo_text = tokenizer.decode(output[0].tolist())
    print(f'  DPO model:  "{dpo_text}"')

    # Generate from reference (SFT-only) model
    ref_model.unfreeze()
    output = generate(ref_model, prompt, max_tokens=60, temperature=0.7)
    ref_text = tokenizer.decode(output[0].tolist())
    print(f'  SFT model:  "{ref_text}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
