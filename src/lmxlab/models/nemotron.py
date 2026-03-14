"""Nemotron 3 hybrid Mamba-Transformer MoE configuration factory.

NVIDIA's Nemotron 3 family (Super, Nano, Ultra) uses a hybrid
architecture with three layer types encoded in a
``hybrid_override_pattern`` string:

- **M** = Mamba-2/SSD (structured state space sequence mixer)
- **E** = LatentMoE (down-project before routing, many experts)
- **\\*** = Standard attention + dense FFN (with squared ReLU)

References:
- NVIDIA Nemotron 3 Technical Report (2025)
- nvidia/Nemotron-H-8B-Base-8K (HuggingFace)
- LatentMoE paper arXiv:2601.18089
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def _parse_hybrid_pattern(
    pattern: str,
    attn_cfg: BlockConfig,
    moe_cfg: BlockConfig,
    mamba_cfg: BlockConfig,
    dense_cfg: BlockConfig | None = None,
) -> tuple[BlockConfig, ...]:
    """Convert hybrid pattern string to per-layer BlockConfigs.

    Args:
        pattern: String of 'M', 'E', '*', '-' characters.
        attn_cfg: Config for '*' (attention + dense FFN) layers.
        moe_cfg: Config for 'E' (LatentMoE) layers.
        mamba_cfg: Config for 'M' (Mamba-2) layers.
        dense_cfg: Config for '-' (dense MLP, no attention).
            If None, '-' is not allowed.

    Returns:
        Tuple of BlockConfig, one per layer.

    Raises:
        ValueError: If pattern contains unknown characters.
    """
    mapping: dict[str, BlockConfig] = {
        'M': mamba_cfg, 'E': moe_cfg, '*': attn_cfg,
    }
    if dense_cfg is not None:
        mapping['-'] = dense_cfg
    configs = []
    for i, c in enumerate(pattern):
        if c not in mapping:
            valid = ', '.join(sorted(mapping.keys()))
            raise ValueError(
                f'Unknown pattern character {c!r} at '
                f'position {i}. Expected {valid}.'
            )
        configs.append(mapping[c])
    return tuple(configs)


def nemotron3_config(
    hybrid_override_pattern: str = 'MEME*ME*',
    vocab_size: int = 256000,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 2,
    d_ff: int = 16384,
    mamba_n_heads: int = 128,
    mamba_head_dim: int = 64,
    ssm_state_size: int = 128,
    mamba_expand: int = 2,
    mamba_n_groups: int = 8,
    mamba_chunk_size: int = 128,
    n_experts: int = 512,
    top_k_experts: int = 22,
    moe_latent_size: int = 1024,
    moe_d_ff: int = 1024,
    shared_expert_d_ff: int = 16384,
    moe_routed_scaling_factor: float = 5.0,
    moe_n_groups: int = 8,
    moe_topk_groups: int = 4,
    max_seq_len: int = 4096,
    rope_theta: float = 10000.0,
    tie_embeddings: bool = False,
    conv_kernel_size: int = 4,
) -> ModelConfig:
    """Create a Nemotron 3 hybrid model configuration.

    Builds three base BlockConfigs (attention, MoE, Mamba) and
    maps the pattern string to per-layer configs.

    Args:
        hybrid_override_pattern: Layer type pattern (M/E/-/*).
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads (for * layers).
        n_kv_heads: Number of KV heads (for * layers).
        d_ff: Dense FFN intermediate dimension.
        mamba_n_heads: Number of Mamba SSM heads.
        mamba_head_dim: Dimension per Mamba head.
        ssm_state_size: SSM state dimension N.
        mamba_expand: Mamba inner dimension multiplier.
        mamba_n_groups: Number of B/C sharing groups.
        mamba_chunk_size: Chunk size for SSD parallel form.
        n_experts: Total number of routed experts.
        top_k_experts: Number of experts per token.
        moe_latent_size: Latent dimension for MoE routing.
        moe_d_ff: Per-expert FFN intermediate dimension.
        shared_expert_d_ff: Shared expert FFN dimension.
        moe_routed_scaling_factor: Routed expert output scale.
        moe_n_groups: Number of expert groups for selection.
        moe_topk_groups: Number of top groups to select from.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.
        conv_kernel_size: Mamba conv kernel size.

    Returns:
        ModelConfig for a Nemotron 3 hybrid model.
    """
    # Attention layers (*): GQA only, no FFN.
    # FFN is in separate dense (-) or MoE (E) layers.
    # Reference: nvidia/Nemotron-H-8B layer 7 has only
    # q/k/v/o projections, no MLP weights.
    attn_cfg = BlockConfig(
        attention='gqa',
        ffn='none',
        norm='rms_norm',
        position='rope',
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        bias=False,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    # MoE layers (E): no attention, LatentMoE FFN
    moe_cfg = BlockConfig(
        attention='none',
        ffn='latent_moe',
        norm='rms_norm',
        position='none',
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=False,
        pre_norm=True,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        moe_latent_size=moe_latent_size,
        moe_d_ff=moe_d_ff,
        shared_expert_d_ff=shared_expert_d_ff,
        moe_routed_scaling_factor=moe_routed_scaling_factor,
        moe_n_groups=moe_n_groups,
        moe_topk_groups=moe_topk_groups,
    )

    # Mamba layers (M): Mamba-2 attention, no FFN
    mamba_cfg = BlockConfig(
        attention='mamba2',
        ffn='none',
        norm='rms_norm',
        position='none',
        d_model=d_model,
        n_heads=n_heads,
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

    # Dense MLP layers (-): no attention, relu2 FFN.
    # Used in 8B model pattern where some layers have
    # dense MLP instead of MoE.
    dense_cfg = BlockConfig(
        attention='none',
        ffn='relu2',
        norm='rms_norm',
        position='none',
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=False,
        pre_norm=True,
    )

    block_configs = _parse_hybrid_pattern(
        hybrid_override_pattern, attn_cfg, moe_cfg, mamba_cfg,
        dense_cfg,
    )

    return ModelConfig(
        block=attn_cfg,
        block_configs=block_configs,
        n_layers=len(hybrid_override_pattern),
        vocab_size=vocab_size,
        tie_embeddings=tie_embeddings,
    )


def nemotron3_tiny() -> ModelConfig:
    """Tiny Nemotron 3 for testing.

    4 layers (MEM*), d=64, small experts.
    """
    return nemotron3_config(
        hybrid_override_pattern='MEM*',
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
        n_experts=4,
        top_k_experts=2,
        moe_latent_size=32,
        moe_d_ff=64,
        shared_expert_d_ff=128,
        moe_routed_scaling_factor=1.0,
        moe_n_groups=1,
        moe_topk_groups=1,
        max_seq_len=128,
        conv_kernel_size=4,
    )


def nemotron3_super() -> ModelConfig:
    """Nemotron 3 Super 120B configuration.

    88 layers, d=4096, 512 experts (22 active), 120B total
    params / 12B active. Does NOT fit in 36GB.
    """
    pattern = (
        'MEMEMEM*MEMEMEM*MEMEMEM*MEMEMEM*MEMEMEM*'
        'MEMEMEM*MEMEMEM*MEMEMEM*MEMEMEM*MEMEMEM*'
        'MEMEMEM*'
    )
    return nemotron3_config(
        hybrid_override_pattern=pattern,
        vocab_size=256000,
        d_model=4096,
        n_heads=32,
        n_kv_heads=2,
        d_ff=16384,
        mamba_n_heads=128,
        mamba_head_dim=64,
        ssm_state_size=128,
        mamba_expand=2,
        n_experts=512,
        top_k_experts=22,
        moe_latent_size=1024,
        moe_d_ff=1024,
        shared_expert_d_ff=16384,
        max_seq_len=4096,
    )


def nemotron3_nano() -> ModelConfig:
    """Nemotron 3 Nano 30B configuration.

    52 layers, d=2688, 128 experts (6 active), 31.6B total
    params / 3.2B active. Does NOT fit in 36GB.
    """
    pattern = (
        'MEMEMEM*MEMEMEM*MEMEMEM*MEMEMEM*'
        'MEMEMEM*MEMEMEM*MEME'
    )
    return nemotron3_config(
        hybrid_override_pattern=pattern,
        vocab_size=256000,
        d_model=2688,
        n_heads=24,
        n_kv_heads=2,
        d_ff=10752,
        mamba_n_heads=64,
        mamba_head_dim=84,
        ssm_state_size=128,
        mamba_expand=2,
        n_experts=128,
        top_k_experts=6,
        moe_latent_size=768,
        moe_d_ff=768,
        shared_expert_d_ff=10752,
        max_seq_len=4096,
    )


def nemotron3_8b() -> ModelConfig:
    """Nemotron-H 8B configuration.

    52 layers (24 Mamba, 24 dense MLP, 4 attention), d=4096.
    No MoE — uses dense MLPs instead.
    Reference: nvidia/Nemotron-H-8B-Base-8K config.json.
    """
    pattern = (
        'M-M-M-M*-'
        'M-M-M-M-M*-'
        'M-M-M-M-M*-'
        'M-M-M-M-M*-'
        'M-M-M-M-M-'
    )
    return nemotron3_config(
        hybrid_override_pattern=pattern,
        vocab_size=131072,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        d_ff=21504,
        mamba_n_heads=128,
        mamba_head_dim=64,
        ssm_state_size=128,
        mamba_expand=2,
        mamba_n_groups=8,
        max_seq_len=8192,
        tie_embeddings=False,
    )
