"""Jamba configuration factory.

AI21's Jamba uses a hybrid Mamba-2 + GQA architecture with
MoE FFN. The pattern repeats ``MMMA`` (3 Mamba + 1 attention),
where some attention layers use MoE and others dense FFN.

References:
- Jamba: A Hybrid Transformer-Mamba Language Model (AI21, 2024)
- ai21labs/Jamba-v0.1 (HuggingFace)
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def jamba_config(
    vocab_size: int = 65536,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    n_layers: int = 32,
    d_ff: int = 14336,
    mamba_n_heads: int = 64,
    mamba_head_dim: int = 64,
    ssm_state_size: int = 16,
    mamba_expand: int = 2,
    mamba_n_groups: int = 1,
    mamba_chunk_size: int = 256,
    conv_kernel_size: int = 4,
    n_experts: int = 16,
    top_k_experts: int = 2,
    moe_every: int = 2,
    attn_every: int = 4,
    max_seq_len: int = 4096,
    rope_theta: float = 10000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Jamba model configuration.

    Jamba alternates Mamba-2 and GQA layers in an MMMA pattern
    (3 Mamba + 1 GQA per cycle). Even-indexed attention layers
    use MoE FFN, others use dense GatedFFN.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads (for GQA layers).
        n_kv_heads: Number of KV heads (for GQA layers).
        n_layers: Number of layers.
        d_ff: Feed-forward intermediate dimension.
        mamba_n_heads: Number of Mamba SSM heads.
        mamba_head_dim: Dimension per Mamba head.
        ssm_state_size: SSM state dimension N.
        mamba_expand: Mamba inner dimension multiplier.
        mamba_n_groups: Number of B/C sharing groups.
        mamba_chunk_size: Chunk size for SSD parallel form.
        conv_kernel_size: Mamba conv kernel size.
        n_experts: Number of routed experts (for MoE layers).
        top_k_experts: Experts activated per token.
        moe_every: Place MoE FFN every N attention layers.
        attn_every: Place an attention layer every N layers.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Jamba model.
    """
    common = dict(
        norm="rms_norm",
        d_model=d_model,
        n_heads=n_heads,
        bias=False,
        pre_norm=True,
    )

    # Mamba-2 layers: SSM + GatedFFN (always dense)
    mamba_cfg = BlockConfig(
        attention="mamba2",
        ffn="gated",
        position="none",
        d_ff=d_ff,
        mamba_n_heads=mamba_n_heads,
        mamba_head_dim=mamba_head_dim,
        ssm_state_size=ssm_state_size,
        mamba_expand=mamba_expand,
        mamba_n_groups=mamba_n_groups,
        mamba_chunk_size=mamba_chunk_size,
        conv_kernel_size=conv_kernel_size,
        **common,
    )

    # Attention + dense FFN
    attn_dense_cfg = BlockConfig(
        attention="gqa",
        ffn="gated",
        position="rope",
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        **common,
    )

    # Attention + MoE FFN
    attn_moe_cfg = BlockConfig(
        attention="gqa",
        ffn="moe",
        position="rope",
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        **common,
    )

    # Build per-layer configs
    configs = []
    attn_count = 0
    for i in range(n_layers):
        if (i + 1) % attn_every == 0:
            # Attention layer: alternate MoE and dense
            if attn_count % moe_every == 0:
                configs.append(attn_moe_cfg)
            else:
                configs.append(attn_dense_cfg)
            attn_count += 1
        else:
            configs.append(mamba_cfg)

    return ModelConfig(
        block=attn_dense_cfg,
        block_configs=tuple(configs),
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def jamba_10m() -> ModelConfig:
    """Jamba ~10M params for research experiments.

    12 layers (MMMA pattern x3), d=128, 4 MoE experts,
    BPE vocab (50257), tied embeddings. ~10.2M params.
    """
    return jamba_config(
        vocab_size=50257,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=12,
        d_ff=384,
        mamba_n_heads=8,
        mamba_head_dim=32,
        ssm_state_size=16,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_chunk_size=64,
        n_experts=4,
        top_k_experts=2,
        moe_every=2,
        attn_every=4,
        max_seq_len=512,
        tie_embeddings=True,
    )


def jamba_tiny() -> ModelConfig:
    """Tiny Jamba for testing.

    8 layers (MMMA pattern x2), d=64, 4 MoE experts.
    """
    return jamba_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=8,
        d_ff=128,
        mamba_n_heads=4,
        mamba_head_dim=32,
        ssm_state_size=16,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_chunk_size=32,
        n_experts=4,
        top_k_experts=2,
        moe_every=2,
        attn_every=4,
        max_seq_len=128,
    )
