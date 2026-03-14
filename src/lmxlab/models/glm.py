"""GLM-4.5 configuration factory.

Zhipu AI's GLM-4.5 uses Multi-Head Latent Attention (MLA)
without positional encoding (NoPE). It shares the MLA
architecture with DeepSeek V2 but sets ``rope_dim=0`` to
disable RoPE entirely, relying on learned attention patterns.

References:
- GLM-4.5 (Zhipu AI / THUDM, 2025)
- THUDM/GLM-4.5-9B-Chat (HuggingFace)
"""

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.core.mla import MLA  # noqa: F401  # register MLA


def glm45_config(
    vocab_size: int = 151552,
    d_model: int = 3584,
    n_heads: int = 28,
    n_layers: int = 40,
    d_ff: int = 18944,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 1536,
    max_seq_len: int = 8192,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a GLM-4.5 model configuration.

    GLM-4.5 uses MLA attention with ``rope_dim=0`` (no RoPE),
    GatedFFN (SwiGLU), RMSNorm, pre-norm. The MLA module
    handles ``rope_dim=0`` by skipping RoPE application.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        kv_lora_rank: Latent dimension for KV compression.
        q_lora_rank: Latent dimension for Q compression.
        max_seq_len: Maximum sequence length.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a GLM-4.5 model.
    """
    block = BlockConfig(
        attention="mla",
        ffn="gated",
        norm="rms_norm",
        position="none",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=False,
        max_seq_len=max_seq_len,
        pre_norm=True,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        rope_dim=0,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
    )


def glm45_tiny() -> ModelConfig:
    """Tiny GLM-4.5 for testing (d=64, 2 layers)."""
    return glm45_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        kv_lora_rank=16,
        q_lora_rank=32,
        max_seq_len=128,
    )
