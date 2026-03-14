"""Configuration dataclasses for model and block definitions."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockConfig:
    """Configuration for a single transformer block.

    Defines which components (attention, FFN, norm, position encoding)
    to use and their parameters. Components are looked up by name
    from the registry.

    Args:
        attention: Registry name for attention module.
        ffn: Registry name for feed-forward module.
        norm: Registry name for normalization function.
        position: Registry name for positional encoding.
        d_model: Hidden dimension size.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA).
            Defaults to n_heads (standard MHA).
        d_ff: Feed-forward intermediate dimension.
        bias: Whether to use bias in linear layers.
        dropout: Dropout rate (0.0 = no dropout).
        norm_eps: Epsilon for normalization layers.
        rope_theta: Base frequency for RoPE.
        max_seq_len: Maximum sequence length.
        pre_norm: If True, apply norm before attention/FFN (pre-norm).
            If False, apply after (post-norm).
        window_size: Sliding window size for local attention.
            None means full (global) attention.
        qk_norm: If True, apply per-head RMSNorm to Q and K
            after reshape, before RoPE (OLMo 2 style).
        attention_chunk_size: Chunk size for chunked local
            attention. None means full (global) attention.
        mup: If True, use μP attention scaling (1/d_head
            instead of 1/√d_head).
    """

    attention: str = "mha"
    ffn: str = "standard"
    norm: str = "layer_norm"
    position: str = "sinusoidal"
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int | None = None
    d_ff: int = 2048
    bias: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 2048
    pre_norm: bool = True
    # Sliding window attention parameters
    window_size: int | None = None
    # MLA (Multi-Head Latent Attention) parameters
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    rope_dim: int | None = None
    # Mixture of Experts parameters
    n_experts: int | None = None
    top_k_experts: int = 2
    n_shared_experts: int | None = None
    # Gated DeltaNet (linear attention) parameters
    conv_kernel_size: int = 4
    use_short_conv: bool = False
    # Mamba-2 (SSM) parameters
    mamba_n_heads: int | None = None
    mamba_head_dim: int | None = None
    ssm_state_size: int = 128
    mamba_expand: int = 2
    mamba_n_groups: int = 1
    mamba_chunk_size: int = 128
    # LatentMoE parameters
    moe_latent_size: int | None = None
    moe_d_ff: int | None = None
    shared_expert_d_ff: int | None = None
    moe_routed_scaling_factor: float = 1.0
    moe_n_groups: int = 1
    moe_topk_groups: int = 1
    # QK-norm (per-head RMSNorm on Q and K)
    qk_norm: bool = False
    # Chunked local attention chunk size
    attention_chunk_size: int | None = None
    # μP (Maximal Update Parameterization) flag
    mup: bool = False

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def effective_n_kv_heads(self) -> int:
        """Number of KV heads (defaults to n_heads if not set)."""
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a full language model.

    Args:
        block: Block configuration (shared across all layers).
        vocab_size: Vocabulary size.
        n_layers: Number of transformer blocks.
        tie_embeddings: Whether to tie input/output embeddings.
        block_configs: Per-layer block overrides (optional).
            If provided, must have length n_layers.
        mup_base_width: Base model width for μP. When set,
            enables μP scaling. None means standard
            parameterization (SP).
        mtp_n_predict: Number of multi-token prediction heads.
            0 disables MTP.
        mtp_lambda: Weight for MTP auxiliary loss.
    """

    block: BlockConfig = field(default_factory=BlockConfig)
    vocab_size: int = 32000
    n_layers: int = 6
    tie_embeddings: bool = True
    block_configs: tuple[BlockConfig, ...] | None = None
    mup_base_width: int | None = None
    mtp_n_predict: int = 0
    mtp_lambda: float = 0.1

    @property
    def width_mult(self) -> float:
        """Width multiplier for μP (d_model / base_width).

        Returns 1.0 when μP is disabled.
        """
        if self.mup_base_width is None:
            return 1.0
        return self.block.d_model / self.mup_base_width

    def get_block_config(self, layer_idx: int) -> BlockConfig:
        """Get block config for a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            BlockConfig for the given layer.
        """
        if self.block_configs is not None:
            return self.block_configs[layer_idx]
        return self.block
