"""Falcon H1 configuration factory.

TII's Falcon H1 uses a hybrid Mamba-2 + GQA architecture,
similar to Nemotron-H. Most layers are Mamba-2 (SSM), with
periodic GQA attention layers for global context.

All layers use GatedFFN (SwiGLU) with RMSNorm.

References:
- Falcon-H1 Technical Report (TII, 2025)
- tiiuae/Falcon-H1-34B-Base (HuggingFace)
"""

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.nemotron import _parse_hybrid_pattern


def falcon_h1_config(
    hybrid_pattern: str = "MMMM*MMM*MMM*MMM*",
    vocab_size: int = 65024,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    d_ff: int = 14336,
    mamba_n_heads: int = 64,
    mamba_head_dim: int = 64,
    ssm_state_size: int = 128,
    mamba_expand: int = 2,
    mamba_n_groups: int = 8,
    mamba_chunk_size: int = 256,
    conv_kernel_size: int = 4,
    max_seq_len: int = 8192,
    rope_theta: float = 500000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Falcon H1 hybrid model configuration.

    Falcon H1 is a hybrid Mamba-2 + GQA model. Mamba-2 layers
    handle most of the sequence mixing, with periodic GQA
    layers for global attention. Both layer types have their
    own GatedFFN (SwiGLU).

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
        ModelConfig for a Falcon H1 hybrid model.
    """
    # GQA attention layers (*): GQA + GatedFFN
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

    # Mamba-2 layers (M): Mamba-2 + GatedFFN
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

    # Falcon H1 uses only M and * (no MoE or dense-only)
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


def falcon_h1_10m() -> ModelConfig:
    """Falcon H1 ~10M params for research experiments.

    12 layers (MMM*MMM*MMM*), d=128, BPE vocab (50257),
    tied embeddings. ~9.3M params.
    """
    return falcon_h1_config(
        hybrid_pattern="MMM*MMM*MMM*",
        vocab_size=50257,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=384,
        mamba_n_heads=8,
        mamba_head_dim=32,
        ssm_state_size=16,
        mamba_expand=2,
        mamba_n_groups=1,
        max_seq_len=512,
        tie_embeddings=True,
    )


def falcon_h1_tiny() -> ModelConfig:
    """Tiny Falcon H1 for testing.

    4 layers (MMM*), d=64.
    """
    return falcon_h1_config(
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
