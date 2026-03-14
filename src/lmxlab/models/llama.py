"""LLaMA configuration factory."""

from lmxlab.core.config import BlockConfig, ModelConfig


def llama_config(
    vocab_size: int = 32000,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    n_layers: int = 32,
    d_ff: int = 11008,
    max_seq_len: int = 4096,
    rope_theta: float = 10000.0,
    tie_embeddings: bool = False,
    dropout: float = 0.0,
    mup_base_width: int | None = None,
) -> ModelConfig:
    """Create a LLaMA-style model configuration.

    LLaMA uses: RMSNorm, GQA, GatedFFN (SwiGLU), RoPE,
    pre-norm, no bias.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads (for GQA).
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie input/output embeddings.
        dropout: Dropout rate.
        mup_base_width: Base width for μP. When set, enables
            μP attention scaling and logit scaling.

    Returns:
        ModelConfig for a LLaMA-style model.
    """
    block = BlockConfig(
        attention="gqa",
        ffn="gated",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        bias=False,
        dropout=dropout,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        pre_norm=True,
        mup=mup_base_width is not None,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
        mup_base_width=mup_base_width,
    )


def llama_tiny() -> ModelConfig:
    """Tiny LLaMA for testing (d=64, 2 layers, 4 heads, 2 kv)."""
    return llama_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
    )


def llama_7b() -> ModelConfig:
    """LLaMA-7B configuration."""
    return llama_config()


def llama_13b() -> ModelConfig:
    """LLaMA-13B configuration."""
    return llama_config(
        d_model=5120,
        n_heads=40,
        n_kv_heads=10,
        n_layers=40,
        d_ff=13824,
    )
