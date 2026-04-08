"""Advanced sampling strategies: best-of-N and majority vote.

Uses extra inference compute to improve output quality:
- Best-of-N: generate multiple completions, keep the highest-scoring
- Majority vote: generate multiple completions, return the most common

Usage:
    uv run python recipes/advanced_sampling.py
    uv run python recipes/advanced_sampling.py --n 8 --steps 300
"""

import argparse
from dataclasses import replace

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.inference.sampling import best_of_n, majority_vote
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced sampling")
    parser.add_argument(
        "--steps", type=int, default=300, help="Training steps"
    )
    parser.add_argument(
        "--n", type=int, default=4, help="Number of candidates"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Train a small model first ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)

    print(f"Vocab: {tokenizer.vocab_size}, Tokens: {len(tokens)}")
    print(f"Training for {args.steps} steps...")

    model = LanguageModel(config)
    mx.eval(model.parameters())

    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        log_interval=100,
    )
    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens, batch_size=4, seq_len=32, shuffle=True
        )

    history = trainer.train(data_iter())
    final_loss = history[-1]["loss"] if history else float("nan")
    print(f"Final loss: {final_loss:.4f}")

    prompt_text = "To be"
    prompt = mx.array([tokenizer.encode(prompt_text)])

    # ── 1. Greedy decoding (baseline) ──
    print(f"\n{'=' * 50}")
    print("1. Greedy decoding (temperature=0)")
    print(f"{'=' * 50}")

    output = generate(model, prompt, max_tokens=60, temperature=0.0)
    text = tokenizer.decode(output[0].tolist())
    print(f'  "{text}"')

    # ── 2. Standard sampling ──
    print(f"\n{'=' * 50}")
    print(f"2. Standard sampling (temperature={args.temperature})")
    print(f"{'=' * 50}")

    for i in range(3):
        output = generate(
            model, prompt, max_tokens=60, temperature=args.temperature
        )
        text = tokenizer.decode(output[0].tolist())
        print(f'  Sample {i + 1}: "{text}"')

    # ── 3. Best-of-N ──
    print(f"\n{'=' * 50}")
    print(f"3. Best-of-{args.n} (log probability scoring)")
    print(f"{'=' * 50}")

    output = best_of_n(
        model,
        prompt,
        n=args.n,
        max_tokens=60,
        temperature=args.temperature,
        score_fn="log_prob",
    )
    text = tokenizer.decode(output[0].tolist())
    print(f'  Best (log_prob): "{text}"')

    output = best_of_n(
        model,
        prompt,
        n=args.n,
        max_tokens=60,
        temperature=args.temperature,
        score_fn="length_normalized",
    )
    text = tokenizer.decode(output[0].tolist())
    print(f'  Best (normalized): "{text}"')

    # ── 4. Majority vote ──
    print(f"\n{'=' * 50}")
    print(f"4. Majority vote ({args.n} completions)")
    print(f"{'=' * 50}")

    results = majority_vote(
        model,
        prompt,
        n=args.n,
        max_tokens=30,
        temperature=args.temperature,
    )

    for rank, (token_list, count) in enumerate(results[:5], 1):
        full = tokenizer.encode(prompt_text) + token_list
        text = tokenizer.decode(full)
        print(f'  #{rank} (count={count}): "{text}"')

    n_unique = len(results)
    print(f"\n  Unique completions: {n_unique}/{args.n}")
    if n_unique < args.n:
        print("  (Some completions are identical — model is confident)")
    else:
        print("  (All completions differ — high diversity)")

    print("\nDone!")


if __name__ == "__main__":
    main()
