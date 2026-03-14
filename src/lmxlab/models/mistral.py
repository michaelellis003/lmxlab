"""Mistral Small 3.1 configuration factory.

Mistral Small 3.1 uses sliding window attention for efficient
long-context processing. Every layer uses a fixed-size local
attention window instead of full global attention.

References:
- Mistral Small 3.1 (Mistral AI, 2025)
- mistralai/Mistral-Small-3.1-24B config
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def mistral_small_config(
    vocab_size: int = 131072,
    d_model: int = 5120,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    n_layers: int = 48,
    d_ff: int = 14336,
    window_size: int = 4096,
    max_seq_len: int = 32768,
    rope_theta: float = 1000000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Mistral Small 3.1 model configuration.

    Mistral Small uses: RMSNorm, sliding-window GQA,
    GatedFFN (SwiGLU), RoPE, pre-norm, no bias.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        window_size: Sliding window size for attention.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Mistral Small 3.1 model.
    """
    block = BlockConfig(
        attention="sliding_window_gqa",
        ffn="gated",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        bias=False,
        window_size=window_size,
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


def mistral_small_tiny() -> ModelConfig:
    """Tiny Mistral Small for testing (d=64, 2 layers)."""
    return mistral_small_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        window_size=32,
        max_seq_len=128,
    )
