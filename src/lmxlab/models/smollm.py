"""SmolLM3 configuration factory.

SmolLM3 (HuggingFace) uses an iRoPE pattern similar to
Llama 4: most layers use standard GQA with RoPE, while
periodic NoPE layers use GQA without positional encoding
for long-range information flow.

References:
- SmolLM3 (HuggingFace, 2025)
- HuggingFaceTB/SmolLM3-3B config
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def smollm3_config(
    vocab_size: int = 128256,
    d_model: int = 2560,
    n_heads: int = 20,
    n_kv_heads: int = 5,
    n_layers: int = 36,
    d_ff: int = 6912,
    nope_every: int = 4,
    max_seq_len: int = 8192,
    rope_theta: float = 500000.0,
    tie_embeddings: bool = True,
) -> ModelConfig:
    """Create a SmolLM3 model configuration.

    SmolLM3 uses: RMSNorm, GQA with iRoPE pattern (RoPE +
    periodic NoPE layers), GatedFFN (SwiGLU), pre-norm,
    no bias, tied embeddings.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        nope_every: Place a NoPE layer every N layers.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a SmolLM3 model.
    """
    rope_block = BlockConfig(
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

    nope_block = BlockConfig(
        attention="gqa",
        ffn="gated",
        norm="rms_norm",
        position="none",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        bias=False,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    block_configs = tuple(
        nope_block if (i + 1) % nope_every == 0 else rope_block
        for i in range(n_layers)
    )

    return ModelConfig(
        block=rope_block,
        block_configs=block_configs,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def smollm3_tiny() -> ModelConfig:
    """Tiny SmolLM3 for testing.

    4 layers (3 RoPE + 1 NoPE), d=64.
    """
    return smollm3_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        d_ff=128,
        nope_every=4,
        max_seq_len=128,
    )
