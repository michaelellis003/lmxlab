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


def qwen3_moe_config(
    vocab_size: int = 151936,
    d_model: int = 2048,
    n_heads: int = 16,
    n_kv_heads: int = 4,
    n_layers: int = 28,
    d_ff: int = 1024,
    n_experts: int = 64,
    top_k_experts: int = 8,
    n_shared_experts: int = 4,
    max_seq_len: int = 32768,
    rope_theta: float = 1000000.0,
    tie_embeddings: bool = True,
) -> ModelConfig:
    """Create a Qwen3 MoE model configuration.

    Qwen3 MoE extends the Qwen architecture with
    SharedExpertMoE FFN. Uses GQA attention, RoPE, RMSNorm,
    pre-norm, no bias (unlike base Qwen which has bias).

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        n_layers: Number of transformer layers.
        d_ff: Per-expert FFN intermediate dimension.
        n_experts: Number of routed experts.
        top_k_experts: Experts activated per token.
        n_shared_experts: Number of shared experts.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Qwen3 MoE model.
    """
    block = BlockConfig(
        attention="gqa",
        ffn="shared_moe",
        norm="rms_norm",
        position="rope",
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        n_shared_experts=n_shared_experts,
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


def qwen3_moe_tiny() -> ModelConfig:
    """Tiny Qwen3 MoE for testing (d=64, 2 layers, 4 experts)."""
    return qwen3_moe_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        n_experts=4,
        top_k_experts=2,
        n_shared_experts=1,
        max_seq_len=128,
    )
