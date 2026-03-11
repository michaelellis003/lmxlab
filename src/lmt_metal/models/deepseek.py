"""DeepSeek V2/V3 configuration factory.

DeepSeek V2 introduced Multi-Head Latent Attention (MLA) which
compresses KV representations into a low-rank latent space,
dramatically reducing KV cache memory (up to 28x vs standard MHA).

Key features:
- MLA: Low-rank KV compression with decoupled RoPE
- RMSNorm, GatedFFN (SwiGLU), pre-norm
- High RoPE theta for long context
"""

from lmt_metal.core.config import BlockConfig, ModelConfig
from lmt_metal.core.mla import MLA  # noqa: F401  # register MLA


def deepseek_config(
    vocab_size: int = 102400,
    d_model: int = 5120,
    n_heads: int = 128,
    n_layers: int = 60,
    d_ff: int = 12288,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 1536,
    rope_dim: int = 64,
    max_seq_len: int = 4096,
    rope_theta: float = 10000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a DeepSeek V2-style model configuration.

    DeepSeek V2 uses: RMSNorm, MLA, GatedFFN (SwiGLU), decoupled RoPE,
    pre-norm, no bias.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        kv_lora_rank: Latent dimension for KV compression.
        q_lora_rank: Latent dimension for Q compression.
        rope_dim: Number of head dimensions for RoPE (rest are nope).
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie input/output embeddings.

    Returns:
        ModelConfig for a DeepSeek V2-style model.
    """
    block = BlockConfig(
        attention="mla",
        ffn="gated",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=False,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
        pre_norm=True,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        rope_dim=rope_dim,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def deepseek_tiny() -> ModelConfig:
    """Tiny DeepSeek for testing (d=64, 2 layers, 4 heads)."""
    return deepseek_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        kv_lora_rank=16,
        q_lora_rank=32,
        rope_dim=8,
        max_seq_len=128,
    )
