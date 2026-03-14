"""Bamba configuration factory.

IBM's Bamba uses a hybrid Mamba-2 + GQA architecture, similar
to Falcon H1. Most layers are Mamba-2 with periodic GQA
attention layers for global information flow. All layers use
dense GatedFFN (SwiGLU).

References:
- Bamba Technical Report (IBM, 2025)
- ibm-fms/Bamba-9B (HuggingFace)
"""

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.nemotron import _parse_hybrid_pattern


def bamba_config(
    hybrid_pattern: str = "MMMM*MMM*MMM*MMM*",
    vocab_size: int = 128000,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    d_ff: int = 14336,
    mamba_n_heads: int = 128,
    mamba_head_dim: int = 64,
    ssm_state_size: int = 128,
    mamba_expand: int = 2,
    mamba_n_groups: int = 8,
    mamba_chunk_size: int = 256,
    conv_kernel_size: int = 4,
    max_seq_len: int = 4096,
    rope_theta: float = 500000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Bamba hybrid model configuration.

    Bamba is a hybrid Mamba-2 + GQA model similar to Falcon H1.
    Mamba-2 layers handle most sequence mixing, with periodic
    GQA layers for global attention. Both use GatedFFN.

    Args:
        hybrid_pattern: Layer type pattern (M=Mamba-2, *=GQA).
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads (for GQA layers).
        n_kv_heads: Number of KV heads (for GQA layers).
        d_ff: Feed-forward intermediate dimension.
        mamba_n_heads: Number of Mamba SSM heads.
        mamba_head_dim: Dimension per Mamba head.
        ssm_state_size: SSM state dimension N.
        mamba_expand: Mamba inner dimension multiplier.
        mamba_n_groups: Number of B/C sharing groups.
        mamba_chunk_size: Chunk size for SSD parallel form.
        conv_kernel_size: Mamba conv kernel size.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Bamba hybrid model.
    """
    # GQA attention layers (*)
    attn_cfg = BlockConfig(
        attention="gqa",
        ffn="gated",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        bias=False,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    # Mamba-2 layers (M)
    mamba_cfg = BlockConfig(
        attention="mamba2",
        ffn="gated",
        norm="rms_norm",
        position="none",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=False,
        pre_norm=True,
        mamba_n_heads=mamba_n_heads,
        mamba_head_dim=mamba_head_dim,
        ssm_state_size=ssm_state_size,
        mamba_expand=mamba_expand,
        mamba_n_groups=mamba_n_groups,
        mamba_chunk_size=mamba_chunk_size,
        conv_kernel_size=conv_kernel_size,
    )

    # Bamba uses only M and * (no MoE)
    dummy_moe = attn_cfg  # unused placeholder
    block_configs = _parse_hybrid_pattern(
        hybrid_pattern,
        attn_cfg,
        dummy_moe,
        mamba_cfg,
    )

    return ModelConfig(
        block=attn_cfg,
        block_configs=block_configs,
        vocab_size=vocab_size,
        n_layers=len(hybrid_pattern),
        tie_embeddings=tie_embeddings,
    )


def bamba_tiny() -> ModelConfig:
    """Tiny Bamba for testing.

    4 layers (MMM*), d=64.
    """
    return bamba_config(
        hybrid_pattern="MMM*",
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        mamba_n_heads=4,
        mamba_head_dim=32,
        ssm_state_size=16,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_chunk_size=32,
        max_seq_len=128,
    )
