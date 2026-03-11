"""Evaluate a language model with perplexity and bits-per-byte.

Trains a tiny model, then evaluates on held-out data using
lmt-metal's evaluation metrics. Demonstrates the eval pipeline:
train/eval split, perplexity, BPB, and metric comparison across
different architectures.

Usage:
    uv run python recipes/evaluate_model.py
"""

from dataclasses import replace

import mlx.core as mx

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.eval.metrics import bits_per_byte, perplexity
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.models.llama import llama_tiny
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
    "Ay, there's the rub; for in that sleep of death "
    "what dreams may come when we have shuffled off "
    "this mortal coil, must give us pause. "
)


def train_and_evaluate(
    name: str,
    config,
    tokenizer: CharTokenizer,
    train_tokens: mx.array,
    eval_tokens: mx.array,
    seq_len: int = 32,
) -> dict[str, float]:
    """Train a model and compute evaluation metrics."""
    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = LanguageModel(config)
    mx.eval(model.parameters())

    # Before training: baseline metrics
    eval_batches = list(
        batch_iterator(
            eval_tokens,
            batch_size=2,
            seq_len=seq_len,
            shuffle=False,
        )
    )
    # Wrap batches as token arrays for eval (concat x and last target)
    eval_data = [
        mx.concatenate([x, y[:, -1:]], axis=1) for x, y in eval_batches
    ]

    ppl_before = perplexity(model, eval_data)
    bpb_before = bits_per_byte(model, eval_data)

    # Train
    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=200,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        log_interval=100,
    )
    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            train_tokens,
            batch_size=4,
            seq_len=seq_len,
            shuffle=True,
        )

    trainer.train(data_iter())

    # After training: final metrics
    ppl_after = perplexity(model, eval_data)
    bpb_after = bits_per_byte(model, eval_data)

    print(f"\n{name}:")
    print(f"  Parameters:     {model.count_parameters():,}")
    print(f"  PPL  (before):  {ppl_before:.1f}")
    print(f"  PPL  (after):   {ppl_after:.1f}")
    print(f"  BPB  (before):  {bpb_before:.3f}")
    print(f"  BPB  (after):   {bpb_after:.3f}")

    return {
        "name": name,
        "params": model.count_parameters(),
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "bpb_before": bpb_before,
        "bpb_after": bpb_after,
    }


def main() -> None:
    """Train and evaluate two architectures side-by-side."""
    mx.random.seed(42)

    # Tokenize
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(tokens)}")

    # 80/20 train/eval split
    split = int(len(tokens) * 0.8)
    train_tokens = tokens[:split]
    eval_tokens = tokens[split:]
    print(f"Train: {len(train_tokens)}, Eval: {len(eval_tokens)}")

    # Evaluate two architectures
    results = []
    for name, config_fn in [
        ("GPT-tiny", gpt_tiny),
        ("LLaMA-tiny", llama_tiny),
    ]:
        mx.random.seed(42)
        r = train_and_evaluate(
            name,
            config_fn(),
            tokenizer,
            train_tokens,
            eval_tokens,
        )
        results.append(r)

    # Summary table
    print("\n" + "=" * 55)
    print(f"{'Model':<12} {'Params':>8} {'PPL':>8} {'BPB':>8}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['name']:<12} "
            f"{r['params']:>8,} "
            f"{r['ppl_after']:>8.1f} "
            f"{r['bpb_after']:>8.3f}"
        )
    print("=" * 55)
    print("\nLower PPL and BPB = better.")
    print("Done!")


if __name__ == "__main__":
    main()
