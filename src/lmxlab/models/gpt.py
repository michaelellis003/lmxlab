"""GPT configuration factory."""

from lmxlab.core.config import BlockConfig, ModelConfig


def gpt_config(
    vocab_size: int = 50257,
    d_model: int = 768,
    n_heads: int = 12,
    n_layers: int = 12,
    d_ff: int = 3072,
    max_seq_len: int = 1024,
    tie_embeddings: bool = True,
    dropout: float = 0.0,
    mup_base_width: int | None = None,
) -> ModelConfig:
    """Create a GPT-style model configuration.

    GPT uses: LayerNorm, standard MHA, standard FFN (GELU),
    sinusoidal positional encoding, pre-norm, bias everywhere.

    Args:
        vocab_size: Vocabulary size (default: GPT-2 BPE vocab).
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward intermediate dimension.
        max_seq_len: Maximum sequence length.
        tie_embeddings: Whether to tie input/output embeddings.
        dropout: Dropout rate.
        mup_base_width: Base width for μP. When set, enables
            μP attention scaling and logit scaling.

    Returns:
        ModelConfig for a GPT-style model.
    """
    block = BlockConfig(
        attention="mha",
        ffn="standard",
        norm="layer_norm",
        position="sinusoidal",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=True,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pre_norm=True,
        mup=mup_base_width is not None,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=n_layers,
        tie_embeddings=tie_embeddings,
        mup_base_width=mup_base_width,
    )


# Common GPT sizes


def gpt_tiny() -> ModelConfig:
    """Tiny GPT for testing (d=64, 2 layers, 2 heads)."""
    return gpt_config(
        vocab_size=256,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
    )


def gpt_small() -> ModelConfig:
    """GPT-small (~125M params)."""
    return gpt_config()


def gpt_medium() -> ModelConfig:
    """GPT-medium (~350M params)."""
    return gpt_config(
        d_model=1024,
        n_heads=16,
        n_layers=24,
        d_ff=4096,
    )
