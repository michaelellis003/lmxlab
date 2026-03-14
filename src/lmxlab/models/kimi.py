"""Kimi K2.5 configuration factory.

Kimi K2.5 (Moonshot AI) uses a hybrid architecture with GQA
attention in most layers and Gated DeltaNet (linear attention)
in periodic layers, combined with SharedExpertMoE FFN.

References:
- Kimi K2.5 (Moonshot AI, 2026)
- moonshotai/Kimi-K2.5-21B config
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def kimi_config(
    vocab_size: int = 131072,
    d_model: int = 3584,
    n_heads: int = 28,
    n_kv_heads: int = 4,
    n_layers: int = 28,
    d_ff: int = 4864,
    n_experts: int = 128,
    top_k_experts: int = 8,
    n_shared_experts: int = 1,
    linear_every: int = 4,
    max_seq_len: int = 32768,
    rope_theta: float = 1000000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Kimi K2.5 model configuration.

    Kimi K2.5 uses a hybrid: GQA (3/4 layers) + Gated DeltaNet
    (1/4 layers), all with SharedExpertMoE FFN. Linear attention
    layers replace GQA every ``linear_every`` layers.

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
        linear_every: Place a DeltaNet layer every N.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Kimi K2.5 model.
    """
    moe_kwargs = dict(
        ffn="shared_moe",
        norm="rms_norm",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        n_shared_experts=n_shared_experts,
        bias=False,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    # GQA layers (majority)
    gqa_block = BlockConfig(
        attention="gqa",
        position="rope",
        n_kv_heads=n_kv_heads,
        rope_theta=rope_theta,
        **moe_kwargs,
    )

    # Gated DeltaNet layers (linear attention)
    deltanet_block = BlockConfig(
        attention="gated_deltanet",
        position="none",
        use_short_conv=True,
        conv_kernel_size=4,
        **moe_kwargs,
    )

    block_configs = tuple(
        deltanet_block if (i + 1) % linear_every == 0 else gqa_block
        for i in range(n_layers)
    )

    return ModelConfig(
        block=gqa_block,
        block_configs=block_configs,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def kimi_tiny() -> ModelConfig:
    """Tiny Kimi K2.5 for testing.

    4 layers (3 GQA + 1 DeltaNet), d=64, 4 experts.
    """
    return kimi_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        d_ff=128,
        n_experts=4,
        top_k_experts=2,
        n_shared_experts=1,
        linear_every=4,
        max_seq_len=128,
    )
