"""Train a model with Group Relative Policy Optimization (GRPO).

Demonstrates GRPO training: first train a base model (SFT), then
fine-tune with group-relative rewards. For each prompt, multiple
completions are scored and normalized within the group.

This is a toy example with synthetic rewards for illustration.
In practice, you'd use a reward model or human evaluations.

Usage:
    uv run python recipes/train_grpo.py
    uv run python recipes/train_grpo.py --sft-steps 200 --grpo-steps 100
"""

import argparse
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.gpt import gpt_tiny
from lmxlab.training.config import TrainConfig
from lmxlab.training.grpo import grpo_loss
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
)


def synthetic_reward(tokens: mx.array, reference: mx.array) -> mx.array:
    """Compute synthetic reward based on overlap with reference.

    Higher reward for completions that share more tokens
    with the reference text. This is a toy proxy for a real
    reward model.
    """
    batch_size = tokens.shape[0]
    rewards = []
    for i in range(batch_size):
        seq = tokens[i]
        # Count how many tokens appear in reference
        matches = 0
        for t in seq.tolist():
            if t in reference.tolist():
                matches += 1
        score = matches / max(seq.shape[0], 1)
        rewards.append(score)
    return mx.array(rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training")
    parser.add_argument(
        "--sft-steps",
        type=int,
        default=200,
        help="SFT training steps",
    )
    parser.add_argument(
        "--grpo-steps",
        type=int,
        default=50,
        help="GRPO training steps",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Completions per prompt for GRPO",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty coefficient",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Setup ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)

    print(f"Vocab: {tokenizer.vocab_size}, Tokens: {len(tokens)}")

    # ── Phase 1: SFT ──
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

    # ── Phase 2: GRPO ──
    print(
        f"\nPhase 2: GRPO for {args.grpo_steps} steps "
        f"(group_size={args.group_size}, beta={args.beta})..."
    )

    # Freeze a copy as the reference model
    ref_model = LanguageModel(config)
    ref_weights = dict(mlx.utils.tree_flatten(model.parameters()))
    ref_model.load_weights(list(ref_weights.items()))
    mx.eval(ref_model.parameters())
    ref_model.freeze()

    # GRPO training loop
    optimizer = mx.optimizers.Adam(learning_rate=1e-4)
    loss_and_grad = nn.value_and_grad(model, grpo_loss)
    seq_len = 32

    for step in range(args.grpo_steps):
        # Sample a prompt (first few tokens)
        prompt_len = 8
        start = (step * prompt_len) % max(len(tokens) - prompt_len, 1)
        prompt = tokens[start : start + prompt_len]

        # Generate group of completions
        prompt_batch = mx.broadcast_to(
            prompt[None, :], (args.group_size, prompt_len)
        )
        completions = generate(
            model,
            prompt_batch,
            max_tokens=seq_len - prompt_len,
            temperature=0.8,
        )

        # Score completions with synthetic reward
        rewards = synthetic_reward(completions, tokens)

        # GRPO update
        loss, grads = loss_and_grad(
            model,
            ref_model,
            prompt_batch,
            completions,
            rewards,
            args.beta,
        )
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        if (step + 1) % 10 == 0 or step == 0:
            avg_reward = mx.mean(rewards).item()
            print(
                f"  GRPO step {step + 1}: "
                f"loss={loss.item():.4f}, "
                f"avg_reward={avg_reward:.4f}"
            )

    # ── Generate comparison ──
    print("\nGeneration comparison:")
    prompt = mx.array([tokenizer.encode("To be")])

    output = generate(model, prompt, max_tokens=60, temperature=0.7)
    grpo_text = tokenizer.decode(output[0].tolist())
    print(f'  GRPO model: "{grpo_text}"')

    ref_model.unfreeze()
    output = generate(ref_model, prompt, max_tokens=60, temperature=0.7)
    ref_text = tokenizer.decode(output[0].tolist())
    print(f'  SFT model:  "{ref_text}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
