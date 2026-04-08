"""Profile model architectures: memory, throughput, and generation speed.

Benchmarks tiny versions of all architectures side-by-side to compare
forward pass throughput, memory usage, and autoregressive generation
speed on Apple Silicon.

Usage:
    uv run python recipes/profile_models.py
    uv run python recipes/profile_models.py --seq-len 64 --iter 20
"""

import argparse

import mlx.core as mx

from lmxlab.experiments.profiling import (
    count_parameters_by_module,
    memory_estimate,
    profile_forward,
    profile_generation,
)
from lmxlab.models.base import LanguageModel
from lmxlab.models.deepseek import deepseek_tiny
from lmxlab.models.falcon import falcon_h1_tiny
from lmxlab.models.gemma import gemma_tiny
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.jamba import jamba_tiny
from lmxlab.models.llama import llama_tiny
from lmxlab.models.mixtral import mixtral_tiny
from lmxlab.models.qwen35 import qwen35_tiny

ARCHITECTURES = {
    "GPT": gpt_tiny,
    "LLaMA": llama_tiny,
    "Gemma": gemma_tiny,
    "Mixtral": mixtral_tiny,
    "DeepSeek": deepseek_tiny,
    "Falcon-H1": falcon_h1_tiny,
    "Jamba": jamba_tiny,
    "Qwen3.5": qwen35_tiny,
}


def main() -> None:
    """Profile all architectures."""
    parser = argparse.ArgumentParser(description="Profile models")
    parser.add_argument(
        "--seq-len", type=int, default=32, help="Sequence length"
    )
    parser.add_argument(
        "--iter", type=int, default=10, help="Benchmark iterations"
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=20,
        help="Tokens for generation benchmark",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    print("=" * 70)
    print("Architecture Profiling (tiny configs)")
    print("=" * 70)

    # --- Memory and parameters ---
    print(
        f"\n{'Arch':<12} {'Params':>10} {'Memory MB':>10} "
        f"{'Embed %':>8} {'Blocks %':>9}"
    )
    print("-" * 55)

    models = {}
    for name, factory in ARCHITECTURES.items():
        config = factory()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        models[name] = (model, config)

        mem = memory_estimate(model)
        breakdown = count_parameters_by_module(model)
        total = mem["param_count"]

        embed_pct = 100 * breakdown.get("embed", 0) / total
        blocks_pct = 100 * breakdown.get("blocks", 0) / total

        print(
            f"{name:<12} {total:>10,} {mem['total_mb']:>10.2f} "
            f"{embed_pct:>7.1f}% {blocks_pct:>8.1f}%"
        )

    # --- Forward pass throughput ---
    print(f"\nForward Pass (seq_len={args.seq_len}, {args.iter} iterations)")
    print(f"{'Arch':<12} {'Mean ms':>10} {'Std ms':>10} {'Tok/s':>12}")
    print("-" * 48)

    for name, (model, config) in models.items():
        tokens = mx.random.randint(
            0, config.vocab_size, shape=(1, args.seq_len)
        )
        result = profile_forward(model, tokens, n_warmup=3, n_iter=args.iter)
        print(
            f"{name:<12} {result['mean_ms']:>10.2f} "
            f"{result['std_ms']:>10.2f} "
            f"{result['tokens_per_sec']:>12,.0f}"
        )

    # --- Generation throughput ---
    print(f"\nGeneration ({args.gen_tokens} tokens)")
    print(f"{'Arch':<12} {'Prefill ms':>11} {'ms/token':>10} {'Tok/s':>10}")
    print("-" * 47)

    for name, (model, _config) in models.items():
        prompt = mx.array([[1, 2, 3, 4]])
        result = profile_generation(model, prompt, max_tokens=args.gen_tokens)
        print(
            f"{name:<12} {result['prefill_ms']:>11.2f} "
            f"{result['decode_ms_per_token']:>10.2f} "
            f"{result['decode_tokens_per_sec']:>10,.0f}"
        )

    print("\n" + "=" * 70)
    print("Note: These are tiny configs for comparison. Real model")
    print("performance depends on model size, batch size, and hardware.")
    print("=" * 70)


if __name__ == "__main__":
    main()
