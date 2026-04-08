"""Base language model built from ModelConfig."""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from lmxlab.core.block import ConfigurableBlock
from lmxlab.core.config import ModelConfig
from lmxlab.core.norm import norm_registry


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
        are -1e9.
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

        # Embedding dropout
        self.embed_dropout = nn.Dropout(p=block_cfg.dropout)

        # Transformer blocks
        self.blocks = [
            ConfigurableBlock(config.get_block_config(i))
            for i in range(config.n_layers)
        ]

        # Sinusoidal PE applied once at model level
        self._sinusoidal = block_cfg.position == "sinusoidal"

        # Final norm
        final_norm_cls = norm_registry.get(block_cfg.norm)
        self.final_norm = final_norm_cls(block_cfg)  # type: ignore[call-arg]

        # Output head (possibly tied with embedding)
        if not config.tie_embeddings:
            self.head = nn.Linear(
                block_cfg.d_model, config.vocab_size, bias=False
            )

        # Apply μP weight initialization scaling
        if config.mup_base_width is not None:
            self._apply_mup_init(config.width_mult)

    def __call__(
        self,
        x: mx.array,
        cache: list | None = None,
        return_hidden: bool = False,
    ) -> tuple[mx.array, list] | tuple[mx.array, list, mx.array]:
        """Forward pass.

        Args:
            x: Token IDs of shape (batch, seq_len).
            cache: Optional list of caches per layer. Cache
                types may be heterogeneous in hybrid models
                (KV tuples for attention, SSM state tuples
                for Mamba, None for identity layers).
            return_hidden: If True, also return hidden states
                from final_norm (before lm_head projection).
                Used by Multi-Token Prediction.

        Returns:
            Tuple of (logits, updated_caches) by default.
            If return_hidden is True, returns
            (logits, updated_caches, hidden_states).
        """
        h = self.embed_dropout(self.embed(x))

        # Sinusoidal position encoding (at model level)
        if self._sinusoidal:
            h = self.blocks[0].position(h)

        # Create causal mask
        T = h.shape[1]
        cache_len = 0
        if cache is not None:
            # Find cache_len from first attention-style KV cache.
            # KV caches are (K, V) where both are 4D arrays
            # (B, heads, seq, head_dim) with matching seq dim.
            # SSM caches differ: (ssm_state_4D, conv_state_3D).
            for layer_cache in cache:
                if (
                    layer_cache is not None
                    and isinstance(layer_cache, tuple)
                    and len(layer_cache) == 2
                    and isinstance(layer_cache[0], mx.array)
                    and isinstance(layer_cache[1], mx.array)
                    and layer_cache[0].ndim == 4
                    and layer_cache[1].ndim == 4
                ):
                    cache_len = layer_cache[0].shape[2]
                    break
        mask = _create_causal_mask(T, cache_len)

        new_caches: list = []
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

        # μP: scale logits by 1/width_mult
        if self.config.mup_base_width is not None:
            logits = logits / self.config.width_mult

        if return_hidden:
            return logits, new_caches, h

        return logits, new_caches

    def _apply_mup_init(self, width_mult: float) -> None:
        """Rescale hidden layer weights for μP.

        Scales hidden layer weight init by 1/√width_mult.
        Embedding weights are left unchanged (μP prescribes
        constant embedding init across widths).

        Args:
            width_mult: d_model / base_d_model ratio.
        """
        if width_mult == 1.0:
            return

        scale = 1.0 / math.sqrt(width_mult)

        # Rescale all block (hidden layer) weights
        for block in self.blocks:
            flat = mlx.utils.tree_flatten(block.parameters())
            updates = [
                (k, v * scale)
                for k, v in flat
                if v.ndim >= 2  # only weight matrices, not biases
            ]
            if updates:
                block.load_weights(updates, strict=False)

        # Rescale output head if untied
        if not self.config.tie_embeddings and hasattr(self, "head"):
            self.head.weight = self.head.weight * scale

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        leaves = mlx.utils.tree_flatten(self.parameters())
        return sum(p.size for _, p in leaves)
