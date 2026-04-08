"""Quantize a trained model and compare inference quality.

Trains a model, quantizes it to 4-bit and 8-bit, then compares
generation quality and memory usage across the three versions.

Usage:
    uv run python recipes/quantize_and_generate.py
    uv run python recipes/quantize_and_generate.py --bits 4 8
"""

import argparse
import copy
from dataclasses import replace

import mlx.core as mx

from lmxlab.core.quantize import dequantize_model, quantize_model
from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.experiments.profiling import memory_estimate
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.llama import llama_tiny
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
    """Train, quantize, and compare generation."""
    parser = argparse.ArgumentParser(
        description="Quantization and generation demo"
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=[4, 8],
        help="Quantization bit widths to compare",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Training steps",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Train base model ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)

    config = llama_tiny()
    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = LanguageModel(config)
    mx.eval(model.parameters())

    print(f"Training {model.count_parameters():,} param model...")
    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
    )
    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens,
            batch_size=4,
            seq_len=32,
            shuffle=True,
        )

    history = trainer.train(data_iter())
    print(f"Final loss: {history[-1]['loss']:.4f}\n")

    # --- Baseline memory and generation ---
    prompt = mx.array([tokenizer.encode("To be")])

    base_mem = memory_estimate(model)
    base_output = generate(
        model,
        prompt,
        max_tokens=80,
        temperature=0.7,
    )
    base_text = tokenizer.decode(base_output[0].tolist())

    print("=== Baseline (float16) ===")
    print(f"  Memory: {base_mem['total_mb']:.2f} MB")
    print(f"  Output: {base_text}\n")

    # --- Quantize and compare ---
    results = [("float16", base_mem["total_mb"], base_text)]

    for bits in args.bits:
        # Deep copy so we keep the original
        q_model = copy.deepcopy(model)
        quantize_model(q_model, bits=bits, group_size=64)
        mx.eval(q_model.parameters())

        q_mem = memory_estimate(q_model)
        q_output = generate(
            q_model,
            prompt,
            max_tokens=80,
            temperature=0.7,
        )
        q_text = tokenizer.decode(q_output[0].tolist())

        label = f"{bits}-bit"
        print(f"=== {label} quantized ===")
        print(f"  Memory: {q_mem['total_mb']:.2f} MB")
        ratio = base_mem["total_mb"] / max(q_mem["total_mb"], 0.01)
        print(f"  Compression: {ratio:.1f}x")
        print(f"  Output: {q_text}\n")

        results.append((label, q_mem["total_mb"], q_text))

    # --- Dequantization demo ---
    print("=== Dequantization ===")
    q_model = copy.deepcopy(model)
    quantize_model(q_model, bits=4, group_size=64)
    mx.eval(q_model.parameters())
    q_mem_before = memory_estimate(q_model)

    dequantize_model(q_model)
    mx.eval(q_model.parameters())
    q_mem_after = memory_estimate(q_model)

    print(
        f"  4-bit -> float: "
        f"{q_mem_before['total_mb']:.2f} MB -> "
        f"{q_mem_after['total_mb']:.2f} MB"
    )
    print("  (Useful for fine-tuning after loading quantized weights)")

    # --- Summary table ---
    print(f"\n{'Mode':<12} {'Memory MB':<12} {'Compression':<14}")
    print("-" * 38)
    for label, mb, _ in results:
        ratio = base_mem["total_mb"] / max(mb, 0.01)
        print(f"{label:<12} {mb:<12.2f} {ratio:<14.1f}x")


if __name__ == "__main__":
    main()
