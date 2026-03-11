"""Speculative decoding: draft-then-verify generation.

Demonstrates speculative decoding on Apple Silicon, where both the
draft and target model share unified memory (zero-copy). A small
draft model proposes tokens; the large target model verifies them
in a single forward pass.

The speedup comes from batch-verifying multiple draft tokens at once
instead of running the target model autoregressively for each token.

Usage:
    uv run python recipes/speculative_decoding.py
    uv run python recipes/speculative_decoding.py --draft-tokens 8
"""

import argparse
import time

import mlx.core as mx

from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.inference.speculative import speculative_decode
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.generate import generate
from lmt_metal.models.llama import llama_config

# Training data — both models trained on same corpus
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


def train_model(
    model: LanguageModel,
    tokens: mx.array,
    steps: int = 200,
) -> list[dict]:
    """Quick training loop."""
    from lmt_metal.data.batching import batch_iterator
    from lmt_metal.training.config import TrainConfig
    from lmt_metal.training.trainer import Trainer

    config = TrainConfig(
        learning_rate=1e-3,
        max_steps=steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=5,
        log_interval=50,
    )
    trainer = Trainer(model, config)

    def data_iter():
        yield from batch_iterator(
            tokens, batch_size=4, seq_len=32, shuffle=True
        )

    return trainer.train(data_iter())


def main() -> None:
    """Run speculative decoding demo."""
    parser = argparse.ArgumentParser(description="Speculative decoding demo")
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=4,
        help="Tokens to draft per step",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Training steps for each model",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Tokenize ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    vocab = tokenizer.vocab_size
    print(f"Data: {len(tokens)} tokens, vocab={vocab}")

    # --- Build models ---
    # Target: larger model
    target_config = llama_config(
        vocab_size=vocab,
        d_model=256,
        n_heads=8,
        n_kv_heads=4,
        n_layers=6,
        d_ff=512,
        max_seq_len=128,
        tie_embeddings=True,
    )
    target = LanguageModel(target_config)
    mx.eval(target.parameters())

    # Draft: smaller model (same architecture family)
    draft_config = llama_config(
        vocab_size=vocab,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
        tie_embeddings=True,
    )
    draft = LanguageModel(draft_config)
    mx.eval(draft.parameters())

    t_params = target.count_parameters()
    d_params = draft.count_parameters()
    print(f"Target: {t_params:,} params")
    print(f"Draft:  {d_params:,} params ({d_params / t_params:.1%} of target)")

    # --- Train both ---
    print(f"\nTraining target ({args.train_steps} steps)...")
    mx.random.seed(42)
    train_model(target, tokens, args.train_steps)

    print(f"Training draft ({args.train_steps} steps)...")
    mx.random.seed(42)
    train_model(draft, tokens, args.train_steps)

    # --- Generate: standard autoregressive ---
    prompt_text = "The "
    prompt_ids = tokenizer.encode(prompt_text)
    prompt = mx.array([prompt_ids])

    print(f"\nPrompt: '{prompt_text}'")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Draft tokens per step: {args.draft_tokens}")

    print("\n--- Standard (target only) ---")
    t0 = time.perf_counter()
    standard_out = generate(
        target, prompt, max_tokens=args.max_tokens, temperature=0.0
    )
    mx.eval(standard_out)
    standard_time = time.perf_counter() - t0
    standard_text = tokenizer.decode(standard_out[0].tolist())
    print(f"  {standard_text}")
    print(f"  Time: {standard_time:.3f}s")

    # --- Generate: speculative decoding ---
    print("\n--- Speculative (draft + target) ---")
    t0 = time.perf_counter()
    spec_out, stats = speculative_decode(
        target,
        draft,
        prompt,
        max_tokens=args.max_tokens,
        draft_tokens=args.draft_tokens,
    )
    mx.eval(spec_out)
    spec_time = time.perf_counter() - t0
    spec_text = tokenizer.decode(spec_out[0].tolist())
    print(f"  {spec_text}")
    print(f"  Time: {spec_time:.3f}s")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(
        f"  Drafted: {stats['total_drafted']}, "
        f"Accepted: {stats['total_accepted']}"
    )

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  Standard: {standard_time:.3f}s")
    print(f"  Speculative: {spec_time:.3f}s")
    if standard_time > 0:
        ratio = standard_time / spec_time
        print(f"  Speedup: {ratio:.2f}x")
    match = standard_text == spec_text
    print(f"  Output match: {match}")
    if not match:
        print("  (Outputs may differ due to KV cache handling)")

    print("\n  Note: On tiny models, speculative decoding overhead")
    print("  may exceed savings. The benefit grows with model size")
    print("  — especially on Apple Silicon where both models share")
    print("  unified memory (no copy between CPU/GPU).")


if __name__ == "__main__":
    main()
