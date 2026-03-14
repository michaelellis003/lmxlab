"""Grok 2.5 configuration factory.

Grok 2.5 (xAI) uses GQA attention with SharedExpertMoE FFN
in every layer. All layers are homogeneous (no hybrid pattern).

References:
- Grok-2.5 model card (xAI, 2025)
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def grok_config(
    vocab_size: int = 131072,
    d_model: int = 6144,
    n_heads: int = 48,
    n_kv_heads: int = 8,
    n_layers: int = 64,
    d_ff: int = 8192,
    n_experts: int = 8,
    top_k_experts: int = 2,
    n_shared_experts: int = 1,
    max_seq_len: int = 8192,
    rope_theta: float = 500000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Grok 2.5 model configuration.

    Grok 2.5 uses: RMSNorm, GQA, SharedExpertMoE FFN, RoPE,
    pre-norm, no bias.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        n_layers: Number of transformer layers.
        d_ff: Per-expert FFN intermediate dimension.
        n_experts: Number of routed experts.
        top_k_experts: Experts activated per token.
        n_shared_experts: Number of shared experts.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Grok 2.5 model.
    """
    block = BlockConfig(
        attention="gqa",
        ffn="shared_moe",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        n_shared_experts=n_shared_experts,
        bias=False,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def grok_tiny() -> ModelConfig:
    """Tiny Grok for testing (d=64, 2 layers, 4 experts)."""
    return grok_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        n_experts=4,
        top_k_experts=2,
        n_shared_experts=1,
        max_seq_len=128,
    )
