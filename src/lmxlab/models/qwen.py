"""Qwen configuration factory."""

from lmxlab.core.config import BlockConfig, ModelConfig


def qwen_config(
    vocab_size: int = 151936,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 32,
    n_layers: int = 32,
    d_ff: int = 11008,
    max_seq_len: int = 32768,
    rope_theta: float = 1000000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Qwen-style model configuration.

    Qwen uses: RMSNorm, GQA, GatedFFN (SwiGLU), RoPE
    (high theta for long context), pre-norm, bias in QKV.

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
        ModelConfig for a Qwen-style model.
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
        bias=True,
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


def qwen_tiny() -> ModelConfig:
    """Tiny Qwen for testing."""
    return qwen_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
    )
