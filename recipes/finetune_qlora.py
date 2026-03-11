"""Fine-tune a quantized model using QLoRA.

Demonstrates the QLoRA pipeline:
1. Build a model
2. Quantize to 4-bit (base weights ~8x smaller)
3. Apply LoRA adapters on top of quantized layers
4. Train only the LoRA parameters (~0.1% of total)
5. Generate text with the fine-tuned model

For real fine-tuning, load a pretrained model from HuggingFace
with quantization: load_from_hf(repo, quantize=4).

Usage:
    uv run python recipes/finetune_qlora.py
    uv run python recipes/finetune_qlora.py --rank 16 --steps 200
    uv run python recipes/finetune_qlora.py --bits 8
"""

import argparse

import mlx.core as mx
import mlx.utils

from lmxlab.core.qlora import apply_qlora
from lmxlab.core.quantize import quantize_model
from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.llama import llama_config
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

# Training data — a small corpus for demonstration
TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. "
    "The only thing we have to fear is fear itself. "
    "In the middle of difficulty lies opportunity. "
    "Life is what happens when you're busy making other plans. "
    "The best time to plant a tree was twenty years ago. "
    "The second best time is now. "
    "Not all who wander are lost. "
) * 10


def main() -> None:
    """Run QLoRA fine-tuning demo."""
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning demo")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument(
        "--steps", type=int, default=100, help="Training steps"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--bits", type=int, default=4, choices=[4, 8], help="Quantization bits"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["attention"],
        choices=["attention", "ffn"],
        help="Which layers to apply LoRA to",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Tokenize ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    print(f"Data: {len(tokens)} tokens, vocab={tokenizer.vocab_size}")

    # --- Build model ---
    config = llama_config(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        d_ff=256,
        max_seq_len=128,
        tie_embeddings=True,
    )
    model = LanguageModel(config)
    mx.eval(model.parameters())

    total_params = _count(model.parameters())
    print(f"Model: {total_params:,} total parameters")

    # --- Quantize ---
    print(f"\nQuantizing to {args.bits}-bit...")
    quantize_model(model, bits=args.bits)
    print("  Base weights quantized")

    # --- Apply QLoRA ---
    apply_qlora(model, rank=args.rank, alpha=args.alpha, targets=args.targets)

    trainable = _count(model.trainable_parameters())
    pct = 100 * trainable / total_params

    print(f"\nQLoRA applied (rank={args.rank}, alpha={args.alpha})")
    print(f"  Targets: {args.targets}")
    print(f"  Base: {args.bits}-bit quantized")
    print(f"  Trainable: {trainable:,} ({pct:.1f}% of total)")
    print(f"  Frozen: {total_params - trainable:,}")

    # --- Train ---
    train_config = TrainConfig(
        learning_rate=args.lr,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=5,
        log_interval=20,
    )

    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens, batch_size=4, seq_len=32, shuffle=True
        )

    print(f"\nTraining for {args.steps} steps...")
    history = trainer.train(data_iter())

    # Print loss curve
    print(f"\n{'Step':>5} {'Loss':>10}")
    print("-" * 16)
    for i, m in enumerate(history):
        if (i + 1) % 20 == 0 or i == 0 or i == len(history) - 1:
            print(f"{i + 1:>5} {m['loss']:>10.4f}")

    # --- Generate ---
    print("\nGenerating text:")
    prompt_text = "The "
    prompt_ids = tokenizer.encode(prompt_text)
    prompt = mx.array([prompt_ids])

    from lmxlab.models.generate import generate

    output = generate(model, prompt, max_tokens=80, temperature=0.8)
    mx.eval(output)
    generated_text = tokenizer.decode(output[0].tolist())
    print(f"  {generated_text}")

    # --- Summary ---
    initial_loss = history[0]["loss"]
    final_loss = history[-1]["loss"]
    print("\nSummary:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Improvement:  {initial_loss - final_loss:.4f}")
    print(f"  Trainable:    {pct:.1f}% of parameters")
    print(f"  Base weights: {args.bits}-bit quantized")


def _count(params) -> int:
    return sum(p.size for _, p in mlx.utils.tree_flatten(params))


if __name__ == "__main__":
    main()
