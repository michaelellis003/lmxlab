"""Train a model with curriculum learning.

Demonstrates two curriculum strategies:
1. Length curriculum: start with short sequences, gradually increase
2. Difficulty curriculum: mix easy and hard data with increasing ratio

Compares curriculum training against a baseline to show the effect.

Usage:
    uv run python recipes/train_curriculum.py
    uv run python recipes/train_curriculum.py --steps 400 --stages 4
"""

import argparse
from dataclasses import replace

import mlx.core as mx

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.generate import generate
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.curriculum import length_curriculum
from lmt_metal.training.trainer import Trainer

# Simple text for "easy" data
EASY_TEXT = (
    "The cat sat on the mat. "
    "The dog ran in the park. "
    "The bird flew over the tree. "
    "The fish swam in the pond. "
) * 20

# More complex text for "hard" data
HARD_TEXT = (
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
) * 10

ALL_TEXT = EASY_TEXT + HARD_TEXT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum learning training"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Total training steps per experiment",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=4,
        help="Number of curriculum stages",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=16,
        help="Starting sequence length",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64,
        help="Final sequence length",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    tokenizer = CharTokenizer(ALL_TEXT)
    all_tokens = mx.array(tokenizer.encode(ALL_TEXT), dtype=mx.int32)
    print(f"Vocab: {tokenizer.vocab_size}, Tokens: {len(all_tokens)}")

    config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)

    # ── Experiment 1: Baseline (fixed sequence length) ──
    print(f"\n{'=' * 50}")
    print("Experiment 1: Baseline (fixed seq_len)")
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

    def baseline_data():
        yield from batch_iterator(
            all_tokens,
            batch_size=4,
            seq_len=args.max_seq_len,
            shuffle=True,
        )

    baseline_history = trainer.train(baseline_data())
    baseline_loss = (
        baseline_history[-1]["loss"] if baseline_history else float("nan")
    )
    print(f"Baseline final loss: {baseline_loss:.4f}")

    # ── Experiment 2: Length Curriculum ──
    print(f"\n{'=' * 50}")
    print("Experiment 2: Length curriculum")
    print(f"{'=' * 50}")
    print(
        f"  Stages: {args.stages}, "
        f"seq_len: {args.min_seq_len} -> {args.max_seq_len}"
    )

    mx.random.seed(42)
    curriculum_model = LanguageModel(config)
    mx.eval(curriculum_model.parameters())

    batches_per_stage = args.steps // args.stages
    curriculum_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        log_interval=100,
    )
    trainer = Trainer(curriculum_model, curriculum_config)

    def curriculum_data():
        yield from length_curriculum(
            all_tokens,
            batch_size=4,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            n_stages=args.stages,
            batches_per_stage=batches_per_stage,
        )

    curriculum_history = trainer.train(curriculum_data())
    curriculum_loss = (
        curriculum_history[-1]["loss"] if curriculum_history else float("nan")
    )
    print(f"Curriculum final loss: {curriculum_loss:.4f}")

    # ── Comparison ──
    print(f"\n{'=' * 50}")
    print("Results")
    print(f"{'=' * 50}")
    print(f"  Baseline loss:   {baseline_loss:.4f}")
    print(f"  Curriculum loss:  {curriculum_loss:.4f}")

    diff = baseline_loss - curriculum_loss
    if diff > 0:
        print(f"  Curriculum is better by {diff:.4f}")
    else:
        print(f"  Baseline is better by {-diff:.4f}")

    # ── Generate from both ──
    print("\nGeneration comparison:")
    prompt = mx.array([tokenizer.encode("To be")])

    output = generate(baseline, prompt, max_tokens=60, temperature=0.7)
    print(f'  Baseline:   "{tokenizer.decode(output[0].tolist())}"')

    output = generate(curriculum_model, prompt, max_tokens=60, temperature=0.7)
    print(f'  Curriculum: "{tokenizer.decode(output[0].tolist())}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
