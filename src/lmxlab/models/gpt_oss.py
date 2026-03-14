"""GPT-OSS configuration factory.

GPT-OSS (OpenAI, 2025) is the open-source GPT model family
with LLaMA-like architecture plus QK-norm and tied embeddings.

References:
- GPT-OSS: Open-Sourcing GPT (OpenAI, 2025)
- openai/GPT-OSS-1.5B config
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def gpt_oss_config(
    vocab_size: int = 200064,
    d_model: int = 2048,
    n_heads: int = 16,
    n_kv_heads: int = 4,
    n_layers: int = 24,
    d_ff: int = 5632,
    max_seq_len: int = 8192,
    rope_theta: float = 500000.0,
    tie_embeddings: bool = True,
) -> ModelConfig:
    """Create a GPT-OSS model configuration.

    GPT-OSS uses: RMSNorm, GQA with QK-norm, GatedFFN (SwiGLU),
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
        ModelConfig for a GPT-OSS model.
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


def gpt_oss_tiny() -> ModelConfig:
    """Tiny GPT-OSS for testing (d=64, 2 layers)."""
    return gpt_oss_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
    )
