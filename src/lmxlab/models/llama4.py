"""Llama 4 Scout configuration factory.

Llama 4 Scout uses a hybrid iRoPE pattern: most layers use
chunked local attention (fixed-size chunks with local RoPE),
while every 4th layer uses full GQA without positional
encoding (NoPE) for cross-chunk information flow.

All layers use SharedExpertMoE FFN.

References:
- Llama 4 Scout (Meta, 2025)
- iRoPE: Interleaved RoPE and NoPE pattern
- meta-llama/Llama-4-Scout config
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def llama4_scout_config(
    vocab_size: int = 202048,
    d_model: int = 5120,
    n_heads: int = 40,
    n_kv_heads: int = 8,
    n_layers: int = 48,
    d_ff: int = 8192,
    n_experts: int = 16,
    top_k_experts: int = 1,
    n_shared_experts: int = 1,
    attention_chunk_size: int = 8192,
    nope_every: int = 4,
    max_seq_len: int = 65536,
    rope_theta: float = 500000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Llama 4 Scout model configuration.

    Uses iRoPE pattern: chunked local attention (3/4 layers)
    interleaved with full NoPE attention (1/4 layers). All
    layers use SharedExpertMoE FFN.

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
        attention_chunk_size: Chunk size for local attention.
        nope_every: Place a NoPE (full GQA) layer every N.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Llama 4 Scout model.
    """
    moe_kwargs = dict(
        ffn="shared_moe",
        norm="rms_norm",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        n_shared_experts=n_shared_experts,
        bias=False,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    # Chunked local attention with RoPE (majority)
    chunked_block = BlockConfig(
        attention="chunked_gqa",
        position="rope",
        rope_theta=rope_theta,
        attention_chunk_size=attention_chunk_size,
        **moe_kwargs,
    )

    # Full GQA without positional encoding (NoPE)
    nope_block = BlockConfig(
        attention="gqa",
        position="none",
        **moe_kwargs,
    )

    block_configs = tuple(
        nope_block if (i + 1) % nope_every == 0 else chunked_block
        for i in range(n_layers)
    )

    return ModelConfig(
        block=chunked_block,
        block_configs=block_configs,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def llama4_scout_tiny() -> ModelConfig:
    """Tiny Llama 4 Scout for testing.

    4 layers (3 chunked + 1 NoPE), d=64, 4 experts.
    """
    return llama4_scout_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        d_ff=128,
        n_experts=4,
        top_k_experts=2,
        n_shared_experts=1,
        attention_chunk_size=16,
        nope_every=4,
        max_seq_len=128,
    )


def llama4_maverick_config(
    vocab_size: int = 202048,
    d_model: int = 5120,
    n_heads: int = 40,
    n_kv_heads: int = 8,
    n_layers: int = 48,
    d_ff: int = 8192,
    n_experts: int = 128,
    top_k_experts: int = 1,
    n_shared_experts: int = 1,
    attention_chunk_size: int = 8192,
    nope_every: int = 4,
    max_seq_len: int = 65536,
    rope_theta: float = 500000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Llama 4 Maverick model configuration.

    Same iRoPE pattern as Scout (chunked + NoPE) but with
    128 experts and top_k=1 routing. Maverick is the larger
    MoE variant in the Llama 4 family.

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
        attention_chunk_size: Chunk size for local attention.
        nope_every: Place a NoPE (full GQA) layer every N.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Llama 4 Maverick model.
    """
    return llama4_scout_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        n_shared_experts=n_shared_experts,
        attention_chunk_size=attention_chunk_size,
        nope_every=nope_every,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        tie_embeddings=tie_embeddings,
    )


def llama4_maverick_tiny() -> ModelConfig:
    """Tiny Llama 4 Maverick for testing.

    4 layers (3 chunked + 1 NoPE), d=64, 8 experts, top_k=1.
    """
    return llama4_maverick_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        d_ff=128,
        n_experts=8,
        top_k_experts=1,
        n_shared_experts=1,
        attention_chunk_size=16,
        nope_every=4,
        max_seq_len=128,
    )
