"""Compare all 8 architectures side by side.

Shows how different config factories produce models with different
parameter counts, cache sizes, and forward pass behavior — all
using the same LanguageModel class and ConfigurableBlock system.

Architectures:
  - GPT: MHA + LayerNorm + Standard FFN (baseline)
  - LLaMA: GQA + RMSNorm + SwiGLU
  - Gemma: GQA + RMSNorm + GELU gated FFN
  - Gemma 3: Sliding window + global attention alternating
  - Qwen: GQA + RMSNorm + SwiGLU (with bias)
  - Qwen 3.5: Hybrid DeltaNet (linear attention + softmax)
  - Mixtral: MoE (routed sparse FFN)
  - DeepSeek: MLA (low-rank KV compression)

Usage:
    uv run python recipes/compare_architectures.py
"""

import mlx.core as mx
import mlx.utils

from lmt_metal.models.base import LanguageModel
from lmt_metal.models.deepseek import deepseek_tiny
from lmt_metal.models.gemma import gemma_tiny
from lmt_metal.models.gemma3 import gemma3_tiny
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.models.llama import llama_tiny
from lmt_metal.models.mixtral import mixtral_tiny
from lmt_metal.models.qwen import qwen_tiny
from lmt_metal.models.qwen35 import qwen35_tiny


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
    """Compare tiny versions of all 8 architectures."""
    configs = {
        "GPT (MHA + LayerNorm)": gpt_tiny(),
        "LLaMA (GQA + RMSNorm + SwiGLU)": llama_tiny(),
        "Gemma (GQA + RMSNorm + GELU)": gemma_tiny(),
        "Gemma 3 (Sliding Window)": gemma3_tiny(),
        "Qwen (GQA + bias)": qwen_tiny(),
        "Qwen 3.5 (DeltaNet hybrid)": qwen35_tiny(),
        "Mixtral (MoE)": mixtral_tiny(),
        "DeepSeek (MLA)": deepseek_tiny(),
    }

    print("=" * 65)
    print("Architecture Comparison — All 8 Architectures (tiny configs)")
    print("=" * 65)

    seq_len = 32
    batch = mx.array([[1] * seq_len])
    results = []

    for name, config in configs.items():
        model = LanguageModel(config)
        mx.eval(model.parameters())

        block = config.block
        n_params = count_params(model)
        cache_size = measure_cache_size(model, seq_len)

        print(f"\n{name}")
        print(f"  d_model={block.d_model}, heads={block.n_heads}")
        print(f"  attention={block.attention}, ffn={block.ffn}")
        print(f"  norm={block.norm}, position={block.position}")
        print(f"  parameters: {n_params:,}")
        print(f"  KV cache ({seq_len} tokens): {cache_size:,} floats")

        # Architecture-specific details
        if block.kv_lora_rank:
            print(f"  kv_lora_rank={block.kv_lora_rank}")
        if block.window_size is not None:
            print(f"  window_size={block.window_size}")
        if block.n_experts:
            print(f"  experts={block.n_experts}, top_k={block.top_k}")

        # Forward pass
        logits, _ = model(batch)
        mx.eval(logits)
        print(f"  output shape: {logits.shape}")

        results.append((name, n_params, cache_size))

    # Summary table
    print(f"\n{'=' * 65}")
    print("Summary")
    print(f"{'=' * 65}")
    print(f"  {'Architecture':<35s} {'Params':>10s} {'Cache':>10s}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10}")
    for name, params, cache in results:
        short_name = name.split("(")[0].strip()
        print(f"  {short_name:<35s} {params:>10,} {cache:>10,}")

    # Key insight
    print(f"\n{'=' * 65}")
    print("Key takeaway: Same LanguageModel class, same")
    print("ConfigurableBlock — 8 architectures emerge purely")
    print("from different config factory functions.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
