"""ConfigurableBlock: assembles transformer blocks from registry components."""

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.core.attention import attention_registry
from lmt_metal.core.config import BlockConfig
from lmt_metal.core.ffn import ffn_registry
from lmt_metal.core.norm import norm_registry
from lmt_metal.core.position import position_registry


class ConfigurableBlock(nn.Module):
    """A transformer block assembled from registry components.

    The block uses pre-norm or post-norm residual connections
    depending on the config. Components (attention, FFN, norm,
    position encoding) are looked up from registries by name.

    Args:
        config: Block configuration specifying components.

    Example:
        >>> config = BlockConfig(
        ...     attention='gqa', ffn='gated',
        ...     norm='rms_norm', position='rope',
        ...     d_model=256, n_heads=4, n_kv_heads=2,
        ... )
        >>> block = ConfigurableBlock(config)
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self.config = config

        # Build components from registries
        attn_cls = attention_registry.get(config.attention)
        ffn_cls = ffn_registry.get(config.ffn)
        norm_cls = norm_registry.get(config.norm)

        self.attention = attn_cls(config)
        self.ffn = ffn_cls(config)
        self.attn_norm = norm_cls(config)
        self.ffn_norm = norm_cls(config)

        # Position encoding (created but applied externally
        # for RoPE, or internally for additive encodings)
        self.position = position_registry.get(config.position)(config)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass through the block.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            mask: Optional attention mask.
            cache: Optional KV cache for generation.

        Returns:
            Tuple of (output, updated_cache).
        """
        if self.config.pre_norm:
            return self._pre_norm_forward(x, mask, cache)
        return self._post_norm_forward(x, mask, cache)

    def _pre_norm_forward(
        self,
        x: mx.array,
        mask: mx.array | None,
        cache: tuple[mx.array, mx.array] | None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Pre-norm: norm -> sublayer -> residual."""
        # Attention sublayer
        residual = x
        h = self.attn_norm(x)
        h, new_cache = self.attention(h, mask=mask, cache=cache)
        x = residual + h

        # FFN sublayer
        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = residual + h

        return x, new_cache

    def _post_norm_forward(
        self,
        x: mx.array,
        mask: mx.array | None,
        cache: tuple[mx.array, mx.array] | None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Post-norm: sublayer -> residual -> norm."""
        # Attention sublayer
        h, new_cache = self.attention(x, mask=mask, cache=cache)
        x = self.attn_norm(x + h)

        # FFN sublayer
        h = self.ffn(x)
        x = self.ffn_norm(x + h)

        return x, new_cache
