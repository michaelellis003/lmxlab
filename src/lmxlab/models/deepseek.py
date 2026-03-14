"""DeepSeek V2/V3 configuration factory.

DeepSeek V2 introduced Multi-Head Latent Attention (MLA) which
compresses KV representations into a low-rank latent space,
dramatically reducing KV cache memory (up to 28x vs standard MHA).

DeepSeek V3 adds MoE FFN layers alongside MLA attention.

Key features:
- MLA: Low-rank KV compression with decoupled RoPE
- RMSNorm, GatedFFN (SwiGLU), pre-norm
- High RoPE theta for long context
- V3: SharedExpertMoE with sigmoid routing

References:
- DeepSeek-V2 (Bi et al., 2024, arXiv:2405.04434)
- DeepSeek-V3 Technical Report (DeepSeek-AI, 2025, arXiv:2412.19437)
"""

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.core.mla import MLA  # noqa: F401  # register MLA


def deepseek_config(
    vocab_size: int = 102400,
    d_model: int = 5120,
    n_heads: int = 128,
    n_layers: int = 60,
    d_ff: int = 12288,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 1536,
    rope_dim: int = 64,
    max_seq_len: int = 4096,
    rope_theta: float = 10000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a DeepSeek V2-style model configuration.

    DeepSeek V2 uses: RMSNorm, MLA, GatedFFN (SwiGLU),
    decoupled RoPE, pre-norm, no bias.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        kv_lora_rank: Latent dimension for KV compression.
        q_lora_rank: Latent dimension for Q compression.
        rope_dim: Number of head dims for RoPE.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a DeepSeek V2-style model.
    """
    block = BlockConfig(
        attention="mla",
        ffn="gated",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=False,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        pre_norm=True,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        rope_dim=rope_dim,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def deepseek_tiny() -> ModelConfig:
    """Tiny DeepSeek for testing (d=64, 2 layers, 4 heads)."""
    return deepseek_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        kv_lora_rank=16,
        q_lora_rank=32,
        rope_dim=8,
        max_seq_len=128,
    )


def deepseek_v3_config(
    vocab_size: int = 129280,
    d_model: int = 7168,
    n_heads: int = 128,
    n_layers: int = 61,
    d_ff: int = 18432,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 1536,
    rope_dim: int = 64,
    n_experts: int = 256,
    top_k_experts: int = 8,
    n_shared_experts: int = 1,
    n_dense_layers: int = 1,
    max_seq_len: int = 4096,
    rope_theta: float = 10000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a DeepSeek V3 model configuration.

    DeepSeek V3 uses MLA attention with SharedExpertMoE FFN
    for most layers. The first ``n_dense_layers`` use dense
    GatedFFN instead of MoE.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Per-expert FFN intermediate dimension.
        kv_lora_rank: Latent dimension for KV compression.
        q_lora_rank: Latent dimension for Q compression.
        rope_dim: Number of head dims for RoPE.
        n_experts: Number of routed experts.
        top_k_experts: Experts activated per token.
        n_shared_experts: Number of shared (always-on) experts.
        n_dense_layers: First N layers use dense FFN.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a DeepSeek V3 model.

    References:
        DeepSeek-V3 (DeepSeek-AI, 2025, arXiv:2412.19437).
    """
    common_attn = dict(
        attention="mla",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        bias=False,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        pre_norm=True,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        rope_dim=rope_dim,
    )

    dense_block = BlockConfig(
        **common_attn,
        ffn="gated",
        d_ff=d_ff,
    )

    moe_block = BlockConfig(
        **common_attn,
        ffn="shared_moe",
        d_ff=d_ff,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        n_shared_experts=n_shared_experts,
    )

    block_configs = tuple(
        dense_block if i < n_dense_layers else moe_block
        for i in range(n_layers)
    )

    return ModelConfig(
        block=dense_block,
        block_configs=block_configs,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def deepseek_v3_tiny() -> ModelConfig:
    """Tiny DeepSeek V3 for testing.

    4 layers (1 dense + 3 MoE), d=64, 4 experts.
    """
    return deepseek_v3_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=128,
        kv_lora_rank=16,
        q_lora_rank=32,
        rope_dim=8,
        n_experts=4,
        top_k_experts=2,
        n_shared_experts=1,
        n_dense_layers=1,
        max_seq_len=128,
    )
