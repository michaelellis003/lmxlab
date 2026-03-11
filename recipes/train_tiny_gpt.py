"""Train a tiny GPT on Shakespeare-like text.

End-to-end recipe: create model, tokenize data, train, generate.
Designed to run on any Mac with MLX (or in CI for smoke testing).

Usage:
    uv run python recipes/train_tiny_gpt.py
"""

import mlx.core as mx

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.dataset import TextDataset, TokenDataset
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.generate import generate
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer


def main() -> None:
    """Train a tiny GPT and generate some text."""
    mx.random.seed(42)

    # --- Data ---
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
    )

    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    dataset = TextDataset(text)
    token_dataset = TokenDataset(dataset, tokenizer, seq_len=32)
    print(f"Dataset: {len(token_dataset)} sequences of length 32")

    # --- Model ---
    config = gpt_tiny()
    # Override vocab to match our tokenizer
    from dataclasses import replace

    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = LanguageModel(config)
    mx.eval(model.parameters())
    print(f"Model: {model.count_parameters():,} parameters")

    # --- Training ---
    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=200,
        batch_size=4,
        eval_interval=50,
        log_interval=25,
        compile_step=False,  # Simpler for tiny model
        warmup_steps=10,
    )

    trainer = Trainer(model, train_config)

    # Create batches
    def data_iter():
        yield from batch_iterator(token_dataset, batch_size=4, shuffle=True)

    print("\nTraining...")
    history = trainer.train(data_iter())

    # Print loss curve
    for i, m in enumerate(history):
        if i % 25 == 0 or i == len(history) - 1:
            print(f"  Step {i + 1}: loss={m['loss']:.4f}")

    # --- Generation ---
    print("\nGenerating text:")
    prompt = tokenizer.encode("To be")
    prompt_ids = mx.array([prompt])

    generated = generate(
        model,
        prompt_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=10,
    )
    output_text = tokenizer.decode(generated[0].tolist())
    print(f"  {output_text}")

    print("\nDone!")


if __name__ == "__main__":
    main()
