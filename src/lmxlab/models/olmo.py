"""OLMo 2 configuration factory.

OLMo 2 (AllenAI) uses standard LLaMA-like architecture with
the key addition of QK-norm: per-head RMSNorm applied to Q
and K after projection but before RoPE.

References:
- OLMo 2 (Groeneveld et al., 2025, arXiv:2501.00656)
- allenai/OLMo-2-7B config
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def olmo2_config(
    vocab_size: int = 100352,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    n_layers: int = 32,
    d_ff: int = 11008,
    max_seq_len: int = 4096,
    rope_theta: float = 500000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create an OLMo 2-style model configuration.

    OLMo 2 uses: RMSNorm, GQA with QK-norm, GatedFFN (SwiGLU),
    RoPE, pre-norm, no bias.

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
        ModelConfig for an OLMo 2 model.
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
        qk_norm=True,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def olmo2_tiny() -> ModelConfig:
    """Tiny OLMo 2 for testing (d=64, 2 layers)."""
    return olmo2_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
    )
