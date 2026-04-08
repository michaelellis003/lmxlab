"""Interactive streaming text generation.

Trains a tiny model, then generates text token-by-token using
stream_generate(). Shows stop tokens and repetition penalty.

Usage:
    uv run python recipes/interactive_generate.py
    uv run python recipes/interactive_generate.py --temperature 0.5
    uv run python recipes/interactive_generate.py --max-tokens 200
"""

import argparse
import time
from dataclasses import replace

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import stream_generate
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
    "Ay, there's the rub; for in that sleep of death "
    "what dreams may come when we have shuffled off "
    "this mortal coil, must give us pause. "
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive streaming generation"
    )
    parser.add_argument(
        "--steps", type=int, default=300, help="Training steps"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=150, help="Max generated tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (1.0 = off)",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Tokenize ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    print(f"Vocab: {tokenizer.vocab_size} chars, {len(tokens)} tokens")

    # --- Train ---
    config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)
    model = LanguageModel(config)
    mx.eval(model.parameters())
    print(f"Model: {model.count_parameters():,} parameters")

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

    print(f"\nTraining for {args.steps} steps...")
    history = trainer.train(data_iter())
    final_loss = history[-1]["loss"] if history else float("nan")
    print(f"Final loss: {final_loss:.4f}")

    # --- Streaming generation ---
    prompts = ["To be", "The ", "And by"]

    for prompt_text in prompts:
        prompt_ids = mx.array([tokenizer.encode(prompt_text)])
        print(f'\nPrompt: "{prompt_text}"')
        print('Output: "', end="", flush=True)
        print(prompt_text, end="", flush=True)

        n_tokens = 0
        t0 = time.perf_counter()

        for token_id in stream_generate(
            model,
            prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        ):
            char = tokenizer.decode([token_id])
            print(char, end="", flush=True)
            n_tokens += 1

        elapsed = time.perf_counter() - t0
        tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0
        print(f'"\n  ({n_tokens} tokens, {tok_per_sec:.0f} tok/s)')

    print("\nDone!")


if __name__ == "__main__":
    main()
