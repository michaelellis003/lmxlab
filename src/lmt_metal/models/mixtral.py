"""Mixtral configuration factory.

Mixtral uses the same base as LLaMA but with Mixture of Experts
(MoE) FFN layers. Each token is routed to the top-2 of 8 experts.
"""

from lmt_metal.core.config import BlockConfig, ModelConfig


def mixtral_config(
    vocab_size: int = 32000,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    n_layers: int = 32,
    d_ff: int = 14336,
    n_experts: int = 8,
    top_k_experts: int = 2,
    max_seq_len: int = 32768,
    rope_theta: float = 1000000.0,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Mixtral-style model configuration.

    Mixtral uses GQA attention with MoE FFN: each token is
    routed to top-k of n_experts GatedFFN (SwiGLU) experts.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        n_layers: Number of transformer layers.
        d_ff: Per-expert feed-forward dimension.
        n_experts: Number of expert FFNs.
        top_k_experts: Experts per token.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Mixtral-style model.
    """
    block = BlockConfig(
        attention="gqa",
        ffn="moe",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
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


def mixtral_tiny() -> ModelConfig:
    """Tiny Mixtral for testing (with MoE)."""
    return mixtral_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        n_experts=4,
        top_k_experts=2,
        max_seq_len=128,
    )
