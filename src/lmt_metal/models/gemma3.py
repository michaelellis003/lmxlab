"""Gemma 3 configuration factory with interleaved attention."""

from lmt_metal.core.config import BlockConfig, ModelConfig


def gemma3_config(
    vocab_size: int = 256000,
    d_model: int = 2048,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    n_layers: int = 26,
    d_ff: int = 16384,
    max_seq_len: int = 8192,
    rope_theta: float = 10000.0,
    window_size: int = 4096,
    global_every: int = 6,
    tie_embeddings: bool = True,
) -> ModelConfig:
    """Create a Gemma 3-style model configuration.

    Gemma 3 interleaves local (sliding window) and global
    attention layers. Every ``global_every``-th layer (0-indexed,
    i.e. layers 5, 11, 17, ...) uses full global GQA; all other
    layers use sliding window GQA with the given window size.

    Uses: RMSNorm, GatedFFN (GeGLU), RoPE, pre-norm, no bias,
    tied embeddings.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        window_size: Sliding window size for local layers.
        global_every: Place a global attention layer every N
            layers (1-indexed: layer global_every-1, 2*global_every-1, ...).
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Gemma 3-style model.
    """
    shared = dict(
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

    # Default block uses sliding window (most common)
    default_block = BlockConfig(
        attention="sliding_window_gqa",
        window_size=window_size,
        **shared,
    )

    # Global block uses standard GQA
    global_block = BlockConfig(
        attention="gqa",
        **shared,
    )

    # Build per-layer configs: every global_every-th layer
    # is global (0-indexed: layers global_every-1, 2*global_every-1, ...)
    block_configs = tuple(
        global_block if (i + 1) % global_every == 0 else default_block
        for i in range(n_layers)
    )

    return ModelConfig(
        block=default_block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
        block_configs=block_configs,
    )


def gemma3_tiny() -> ModelConfig:
    """Tiny Gemma 3 for testing."""
    return gemma3_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=6,
        d_ff=128,
        max_seq_len=128,
        window_size=16,
        global_every=6,
    )
