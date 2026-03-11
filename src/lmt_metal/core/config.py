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
    """

    block: BlockConfig = field(default_factory=BlockConfig)
    vocab_size: int = 32000
    n_layers: int = 6
    tie_embeddings: bool = True
    block_configs: tuple[BlockConfig, ...] | None = None

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
