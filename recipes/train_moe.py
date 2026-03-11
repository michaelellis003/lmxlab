"""Train a Mixture of Experts (MoE) model.

Demonstrates MoE training where each token is routed to a subset
of expert FFNs. Compares a standard dense model vs MoE to show
how experts increase capacity without proportional compute cost.

Usage:
    uv run python recipes/train_moe.py
    uv run python recipes/train_moe.py --experts 8 --top-k 2 --steps 200
"""

import argparse

import mlx.core as mx

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.llama import llama_config
from lmt_metal.models.mixtral import mixtral_config
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer

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


def train_model(name, config, tokens, train_config):
    """Train a model and return loss history."""
    mx.random.seed(42)
    model = LanguageModel(config)
    mx.eval(model.parameters())
    params = model.count_parameters()
    print(f"  {name}: {params:,} parameters")

    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens,
            batch_size=train_config.batch_size,
            seq_len=32,
            shuffle=True,
        )

    history = trainer.train(data_iter())
    return [m["loss"] for m in history], params


def main() -> None:
    """Compare dense vs MoE training."""
    parser = argparse.ArgumentParser(description="MoE training demo")
    parser.add_argument(
        "--experts", type=int, default=4, help="Number of experts"
    )
    parser.add_argument(
        "--top-k", type=int, default=2, help="Experts per token"
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Training steps"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # --- Tokenize ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    vocab = tokenizer.vocab_size
    print(f"Data: {len(tokens)} tokens, vocab={vocab}")

    train_config = TrainConfig(
        learning_rate=args.lr,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=5,
        log_interval=25,
    )

    # --- Dense baseline (LLaMA-style) ---
    dense_config = llama_config(
        vocab_size=vocab,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
        tie_embeddings=True,
    )

    # --- MoE model (same base dims, but with experts) ---
    moe_config = mixtral_config(
        vocab_size=vocab,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        n_experts=args.experts,
        top_k_experts=args.top_k,
        max_seq_len=128,
        tie_embeddings=True,
    )

    print(f"\nMoE config: {args.experts} experts, top-{args.top_k}")
    print(f"Training for {args.steps} steps\n")

    # --- Train ---
    print("Training dense (LLaMA):")
    dense_losses, dense_params = train_model(
        "Dense", dense_config, tokens, train_config
    )

    print("\nTraining MoE (Mixtral):")
    moe_losses, moe_params = train_model(
        "MoE", moe_config, tokens, train_config
    )

    # --- Compare ---
    print(f"\n{'Step':>5} {'Dense':>12} {'MoE':>12}")
    print("-" * 30)
    n = min(len(dense_losses), len(moe_losses))
    for i in range(0, n, max(1, n // 10)):
        print(f"{i + 1:>5} {dense_losses[i]:>12.4f} {moe_losses[i]:>12.4f}")
    print("-" * 30)
    print(f"{'Final':>5} {dense_losses[-1]:>12.4f} {moe_losses[-1]:>12.4f}")

    # --- Summary ---
    print("\nSummary:")
    print(
        f"  Dense:  {dense_params:,} params, final loss={dense_losses[-1]:.4f}"
    )
    print(f"  MoE:    {moe_params:,} params, final loss={moe_losses[-1]:.4f}")
    print(f"  Param ratio: {moe_params / dense_params:.1f}x")
    print("\n  MoE has more total parameters but each token only")
    print(f"  uses {args.top_k}/{args.experts} experts, so compute")
    print("  per token is similar to dense.")


if __name__ == "__main__":
    main()
