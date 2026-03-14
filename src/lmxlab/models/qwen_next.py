"""Qwen3-Next configuration factory.

Qwen3-Next uses gated attention (GatedGQA) with sigmoid output
gating, combined with GatedFFN (SwiGLU). The gated attention
variant from "Gated Attention: G1 elementwise" improves
gradient flow and representational capacity.

References:
- Gated Attention (arXiv:2505.06708, NeurIPS 2025 best paper)
- Qwen3-Next blog post (Alibaba, 2026)
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def qwen_next_config(
    vocab_size: int = 152064,
    d_model: int = 3584,
    n_heads: int = 28,
    n_kv_heads: int = 4,
    n_layers: int = 28,
    d_ff: int = 18944,
    max_seq_len: int = 32768,
    rope_theta: float = 1000000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Qwen3-Next model configuration.

    Qwen3-Next uses: RMSNorm, GatedGQA (sigmoid output gate),
    GatedFFN (SwiGLU), RoPE, pre-norm, no bias.

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
        ModelConfig for a Qwen3-Next model.
    """
    block = BlockConfig(
        attention="gated_gqa",
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


def qwen_next_tiny() -> ModelConfig:
    """Tiny Qwen3-Next for testing (d=64, 2 layers)."""
    return qwen_next_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
    )
