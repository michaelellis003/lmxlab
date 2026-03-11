"""Base language model built from ModelConfig."""

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.core.block import ConfigurableBlock
from lmt_metal.core.config import ModelConfig
from lmt_metal.core.norm import norm_registry


def _create_causal_mask(
    seq_len: int,
    cache_len: int = 0,
) -> mx.array:
    """Create additive causal mask with optional cache offset.

    Args:
        seq_len: Length of the query sequence.
        cache_len: Length of cached key/value sequence.

    Returns:
        Additive mask of shape (seq_len, cache_len + seq_len)
        where allowed positions are 0 and masked positions
        are -inf.
    """
    total_len = cache_len + seq_len
    # Row i can attend to columns 0..cache_len+i (inclusive)
    indices = mx.arange(total_len)
    row_indices = mx.arange(seq_len)[:, None] + cache_len
    mask = mx.where(indices[None, :] <= row_indices, 0.0, -1e9)
    return mask


class LanguageModel(nn.Module):
    """Transformer language model assembled from config.

    Uses ConfigurableBlock for each layer. Supports tied
    input/output embeddings and KV caching for generation.

    Args:
        config: Full model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        block_cfg = config.block

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, block_cfg.d_model)

        # Transformer blocks
        self.blocks = [
            ConfigurableBlock(config.get_block_config(i))
            for i in range(config.n_layers)
        ]

        # Final norm
        final_norm_cls = norm_registry.get(block_cfg.norm)
        self.final_norm = final_norm_cls(block_cfg)

        # Output head (possibly tied with embedding)
        if not config.tie_embeddings:
            self.head = nn.Linear(
                block_cfg.d_model, config.vocab_size, bias=False
            )

    def __call__(
        self,
        x: mx.array,
        cache: list[tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Forward pass.

        Args:
            x: Token IDs of shape (batch, seq_len).
            cache: Optional list of KV caches per layer.

        Returns:
            Tuple of (logits, updated_caches).
                logits shape: (batch, seq_len, vocab_size)
        """
        h = self.embed(x)

        # Create causal mask
        T = h.shape[1]
        if cache is not None and cache[0] is not None:
            cache_len = cache[0][0].shape[2]
            mask = _create_causal_mask(T, cache_len)
        else:
            mask = _create_causal_mask(T)

        new_caches: list[tuple[mx.array, mx.array]] = []
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            h, new_cache = block(h, mask=mask, cache=layer_cache)
            new_caches.append(new_cache)

        h = self.final_norm(h)

        # Output projection
        if self.config.tie_embeddings:
            logits = h @ self.embed.weight.T
        else:
            logits = self.head(h)

        return logits, new_caches

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        leaves = mx.utils.tree_flatten(self.parameters())
        return sum(p.size for _, p in leaves)
