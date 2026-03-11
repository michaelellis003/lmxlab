"""Qwen 3.5 configuration factory with hybrid attention.

Qwen 3.5 uses a hybrid architecture interleaving Gated DeltaNet
(linear attention) with standard GQA (full attention) in a 3:1
ratio. This gives efficient long-context processing from DeltaNet
plus global context modeling from periodic full attention layers.
"""

from lmxlab.core.config import BlockConfig, ModelConfig


def qwen35_config(
    vocab_size: int = 151936,
    d_model: int = 2048,
    n_heads: int = 16,
    n_kv_heads: int = 4,
    n_layers: int = 28,
    d_ff: int = 5504,
    max_seq_len: int = 32768,
    rope_theta: float = 1000000.0,
    global_every: int = 4,
    tie_embeddings: bool = False,
) -> ModelConfig:
    """Create a Qwen 3.5-style model configuration.

    Qwen 3.5 interleaves Gated DeltaNet (linear attention) and
    standard GQA layers. Every ``global_every``-th layer uses
    full GQA; all other layers use Gated DeltaNet.

    Uses: RMSNorm, GatedFFN (SwiGLU), RoPE (for GQA layers),
    short causal convolutions (for DeltaNet layers), no bias.

    The 3:1 hybrid ratio (75% DeltaNet, 25% GQA) balances
    efficiency and expressiveness:
    - DeltaNet: O(d^2) per token, fixed-size state, no KV cache
    - GQA: O(n^2) per token, growing KV cache, global context

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of KV heads (for GQA layers).
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency (for GQA layers).
        global_every: Place a GQA layer every N layers.
        tie_embeddings: Whether to tie embeddings.

    Returns:
        ModelConfig for a Qwen 3.5-style model.
    """
    # DeltaNet block (majority of layers)
    deltanet_block = BlockConfig(
        attention="gated_deltanet",
        ffn="gated",
        norm="rms_norm",
        position="none",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=False,
        max_seq_len=max_seq_len,
        pre_norm=True,
        use_short_conv=True,
        conv_kernel_size=4,
    )

    # GQA block (every global_every-th layer)
    gqa_block = BlockConfig(
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

    # Build per-layer configs: 3:1 DeltaNet:GQA pattern
    block_configs = tuple(
        gqa_block if (i + 1) % global_every == 0 else deltanet_block
        for i in range(n_layers)
    )

    return ModelConfig(
        block=deltanet_block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
        block_configs=block_configs,
    )


def qwen35_tiny() -> ModelConfig:
    """Tiny Qwen 3.5 for testing (4 layers, global every 4th)."""
    return qwen35_config(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        d_ff=128,
        max_seq_len=128,
        global_every=4,
    )
