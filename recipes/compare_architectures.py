"""Compare architectures: GPT vs LLaMA vs DeepSeek (MLA).

Shows how different config factories produce models with different
parameter counts, cache sizes, and forward pass behavior — all
using the same LanguageModel class and ConfigurableBlock system.

Usage:
    uv run python recipes/compare_architectures.py
"""

import mlx.core as mx
import mlx.utils

from lmt_metal.models.base import LanguageModel
from lmt_metal.models.deepseek import deepseek_tiny
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.models.llama import llama_tiny


def count_params(model: LanguageModel) -> int:
    """Count trainable parameters."""
    leaves = mlx.utils.tree_flatten(model.parameters())
    return sum(p.size for _, p in leaves)


def measure_cache_size(model: LanguageModel, seq_len: int = 32) -> int:
    """Measure KV cache size after processing a sequence."""
    x = mx.array([[1] * seq_len])
    _, caches = model(x)
    mx.eval(*[c for pair in caches for c in pair])
    total = 0
    for pair in caches:
        for arr in pair:
            total += arr.size
    return total


def main() -> None:
    """Compare tiny versions of GPT, LLaMA, and DeepSeek."""
    configs = {
        "GPT (MHA + LayerNorm)": gpt_tiny(),
        "LLaMA (GQA + RMSNorm)": llama_tiny(),
        "DeepSeek (MLA + RMSNorm)": deepseek_tiny(),
    }

    print("=" * 60)
    print("Architecture Comparison (tiny configs)")
    print("=" * 60)

    seq_len = 32
    batch = mx.array([[1] * seq_len])

    for name, config in configs.items():
        model = LanguageModel(config)
        mx.eval(model.parameters())

        block = config.block
        n_params = count_params(model)
        cache_size = measure_cache_size(model, seq_len)

        print(f"\n{name}")
        print(f"  d_model={block.d_model}, heads={block.n_heads}")
        print(f"  attention={block.attention}, norm={block.norm}")
        print(f"  parameters: {n_params:,}")
        print(f"  KV cache ({seq_len} tokens): {cache_size:,} floats")

        if block.kv_lora_rank:
            print(f"  kv_lora_rank={block.kv_lora_rank}")
            print(f"  rope_dim={block.rope_dim}")

        # Forward pass
        logits, _ = model(batch)
        mx.eval(logits)
        print(f"  output shape: {logits.shape}")

    # Show the key insight
    print("\n" + "=" * 60)
    print("Key takeaway: Same LanguageModel class, same")
    print("ConfigurableBlock — different architectures emerge")
    print("purely from different config factory functions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
