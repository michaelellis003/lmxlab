"""Gemma configuration factory."""

from lmt_metal.core.config import BlockConfig, ModelConfig


def gemma_config(
    vocab_size: int = 256000,
    d_model: int = 2048,
    n_heads: int = 8,
    n_kv_heads: int = 1,
    n_layers: int = 18,
    d_ff: int = 16384,
    max_seq_len: int = 8192,
    rope_theta: float = 10000.0,
    tie_embeddings: bool = True,
) -> ModelConfig:
    """Create a Gemma-style model configuration.

    Gemma uses: RMSNorm, GQA (multi-query), GatedFFN (GeGLU),
    RoPE, pre-norm, no bias, tied embeddings.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Gemma-style model.
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
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def gemma_tiny() -> ModelConfig:
    """Tiny Gemma for testing."""
    return gemma_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
    )
