"""Compare KV cache: MLA vs MHA (Pre-registered Experiment 4).

Tests whether MLA's KV cache compression provides meaningful
benefits on unified memory.

Competing hypotheses:
  H1 (Memory benefit): MLA enables longer generation before OOM.
  H2 (No practical benefit): Unified memory is large enough that
     KV cache isn't the binding constraint at typical lengths.
  H3 (Speed benefit): MLA is faster per-token because reading a
     smaller KV cache is faster (bandwidth-bound).

Protocol: Compare DeepSeek-style MLA vs standard GQA at matched
parameter counts. Generate sequences of increasing length and
measure tokens/second and memory.

Usage:
    uv run python recipes/compare_kv_cache.py
    uv run python recipes/compare_kv_cache.py --max-gen 512
"""

import argparse

import mlx.core as mx

from lmxlab.experiments.profiling import (
    memory_estimate,
    profile_forward,
    profile_generation,
)
from lmxlab.models.base import LanguageModel
from lmxlab.models.deepseek import deepseek_config
from lmxlab.models.llama import llama_config


def build_mha_model(
    vocab_size: int,
    d_model: int,
    n_layers: int,
) -> LanguageModel:
    """Build a standard GQA model (LLaMA-style)."""
    config = llama_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=max(2, d_model // 32),
        n_kv_heads=max(1, d_model // 64),
        n_layers=n_layers,
        d_ff=d_model * 2,
        max_seq_len=2048,
        tie_embeddings=True,
    )
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


def build_mla_model(
    vocab_size: int,
    d_model: int,
    n_layers: int,
) -> LanguageModel:
    """Build a DeepSeek-style MLA model."""
    config = deepseek_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=max(2, d_model // 32),
        n_layers=n_layers,
        d_ff=d_model * 2,
        kv_lora_rank=max(8, d_model // 4),
        q_lora_rank=max(16, d_model // 2),
        rope_dim=max(4, d_model // (max(2, d_model // 32) * 2)),
        max_seq_len=2048,
        tie_embeddings=True,
    )
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


def main() -> None:
    """Run KV cache comparison experiment."""
    parser = argparse.ArgumentParser(
        description="KV cache comparison (Experiment 4)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model dimension",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--max-gen",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        help="Vocabulary size",
    )
    args = parser.parse_args()

    print("=== KV Cache Comparison (Experiment 4) ===")
    print(f"Model: d={args.d_model}, L={args.n_layers}")
    print(f"Vocab: {args.vocab_size}")
    print(f"Max generation: {args.max_gen} tokens\n")

    # --- Build models ---
    print("Building models...")
    mx.random.seed(42)
    mha_model = build_mha_model(
        args.vocab_size,
        args.d_model,
        args.n_layers,
    )
    mx.random.seed(42)
    mla_model = build_mla_model(
        args.vocab_size,
        args.d_model,
        args.n_layers,
    )

    mha_params = mha_model.count_parameters()
    mla_params = mla_model.count_parameters()

    mha_mem = memory_estimate(mha_model)
    mla_mem = memory_estimate(mla_model)

    print(f"  GQA: {mha_params:,} params, {mha_mem['total_mb']:.2f} MB")
    print(f"  MLA: {mla_params:,} params, {mla_mem['total_mb']:.2f} MB")
    param_ratio = mla_params / mha_params if mha_params > 0 else 0
    print(f"  Param ratio (MLA/GQA): {param_ratio:.2f}x\n")

    # --- Forward pass profiling ---
    print("--- Forward Pass Profiling ---")
    seq_lens = [32, 64, 128, 256]
    print(
        f"{'Seq Len':>8} {'GQA ms':>10} {'MLA ms':>10} "
        f"{'GQA tok/s':>12} {'MLA tok/s':>12} {'Speedup':>8}"
    )
    print("-" * 65)

    for seq_len in seq_lens:
        tokens = mx.random.randint(
            0,
            args.vocab_size,
            shape=(1, seq_len),
        )

        mha_fwd = profile_forward(
            mha_model,
            tokens,
            n_warmup=2,
            n_iter=3,
        )
        mla_fwd = profile_forward(
            mla_model,
            tokens,
            n_warmup=2,
            n_iter=3,
        )

        speedup = (
            mha_fwd["mean_ms"] / mla_fwd["mean_ms"]
            if mla_fwd["mean_ms"] > 0
            else 0
        )

        print(
            f"{seq_len:>8} {mha_fwd['mean_ms']:>10.2f} "
            f"{mla_fwd['mean_ms']:>10.2f} "
            f"{mha_fwd['tokens_per_sec']:>12.0f} "
            f"{mla_fwd['tokens_per_sec']:>12.0f} "
            f"{speedup:>7.2f}x"
        )

    # --- Generation profiling ---
    print(f"\n--- Generation Profiling ({args.max_gen} tokens) ---")
    prompt = mx.random.randint(
        0,
        args.vocab_size,
        shape=(1, 8),
    )

    gen_lengths = [32, 64, 128]
    if args.max_gen >= 256:
        gen_lengths.append(256)
    if args.max_gen >= 512:
        gen_lengths.append(512)

    print(
        f"{'Gen Len':>8} {'GQA ms/tok':>12} {'MLA ms/tok':>12} "
        f"{'GQA tok/s':>10} {'MLA tok/s':>10} {'Speedup':>8}"
    )
    print("-" * 65)

    gen_results = []
    for gen_len in gen_lengths:
        mha_gen = profile_generation(
            mha_model,
            prompt,
            max_tokens=gen_len,
        )
        mla_gen = profile_generation(
            mla_model,
            prompt,
            max_tokens=gen_len,
        )

        speedup = (
            mha_gen["decode_ms_per_token"] / mla_gen["decode_ms_per_token"]
            if mla_gen["decode_ms_per_token"] > 0
            else 0
        )

        gen_results.append(
            {
                "gen_len": gen_len,
                "mha_ms_per_tok": mha_gen["decode_ms_per_token"],
                "mla_ms_per_tok": mla_gen["decode_ms_per_token"],
                "speedup": speedup,
            }
        )

        print(
            f"{gen_len:>8} "
            f"{mha_gen['decode_ms_per_token']:>12.3f} "
            f"{mla_gen['decode_ms_per_token']:>12.3f} "
            f"{mha_gen['decode_tokens_per_sec']:>10.0f} "
            f"{mla_gen['decode_tokens_per_sec']:>10.0f} "
            f"{speedup:>7.2f}x"
        )

    # --- Hypothesis evaluation ---
    print(f"\n{'=' * 60}")
    print("Hypothesis evaluation")
    print(f"{'=' * 60}")

    # H3: Speed benefit — check if MLA is consistently faster
    if gen_results:
        avg_speedup = sum(r["speedup"] for r in gen_results) / len(gen_results)
        print(f"  Average MLA decode speedup: {avg_speedup:.2f}x")

        if avg_speedup > 1.1:
            print(
                "  -> Supports H3: MLA is faster per-token "
                "(smaller KV cache reads)"
            )
        elif avg_speedup < 0.9:
            print(
                "  -> Against H3: MLA is slower "
                "(compression overhead > cache savings)"
            )
        else:
            print(
                "  -> Inconclusive: similar decode speed "
                "(compression overhead ~= cache savings)"
            )

        # Check if speedup increases with sequence length
        if len(gen_results) >= 2:
            first = gen_results[0]["speedup"]
            last = gen_results[-1]["speedup"]
            if last > first * 1.1:
                print(
                    "  -> Speedup increases with length: "
                    "supports H1 (memory benefit at scale)"
                )
            else:
                print(
                    "  -> Speedup stable across lengths: "
                    "suggests H2 (not memory-bound yet)"
                )

    print(
        "\nNote: At educational scale, KV cache is small. "
        "Benefits grow with model size and sequence length."
    )
    print(
        f"  GQA model: {mha_mem['total_mb']:.1f} MB, "
        f"MLA model: {mla_mem['total_mb']:.1f} MB"
    )


if __name__ == "__main__":
    main()
