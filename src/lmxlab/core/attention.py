"""Attention modules: MHA, GQA, and Sliding Window GQA."""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.config import BlockConfig
from lmxlab.core.registry import Registry

# Registry for attention variants
attention_registry: Registry[type["AttentionBase"]] = Registry("attention")


class AttentionBase(nn.Module):
    """Base class for attention modules.

    Subclasses implement different head configurations but share
    the same forward interface.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        rope: nn.Module | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask.
            cache: Optional KV cache tuple (keys, values).
            rope: Optional RoPE module for Q/K rotation.

        Returns:
            Tuple of (output, updated_cache).
        """
        raise NotImplementedError


@attention_registry.register("mha")
class MHA(AttentionBase):
    """Multi-Head Attention using mx.fast.scaled_dot_product_attention.

    Standard MHA where n_kv_heads == n_heads.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        # μP uses 1/d_head; SP uses 1/√d_head
        exp = -1.0 if config.mup else -0.5
        self.scale = self.head_dim ** exp

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        rope: nn.Module | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if rope is not None:
            offset = (
                cache[0].shape[2]
                if cache is not None else 0
            )
            q, k = rope(q, k, offset=offset)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out), new_cache


@attention_registry.register("gqa")
class GQA(AttentionBase):
    """Grouped-Query Attention.

    Uses fewer KV heads than query heads for memory efficiency.
    When n_kv_heads == 1, this is Multi-Query Attention (MQA).
    When n_kv_heads == n_heads, this is standard MHA.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        self.n_kv_heads = config.effective_n_kv_heads
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        # μP uses 1/d_head; SP uses 1/√d_head
        exp = -1.0 if config.mup else -0.5
        self.scale = self.head_dim ** exp

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        rope: nn.Module | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if rope is not None:
            offset = (
                cache[0].shape[2]
                if cache is not None else 0
            )
            q, k = rope(q, k, offset=offset)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out), new_cache


@attention_registry.register("none")
class NoneAttention(AttentionBase):
    """Identity attention — returns input unchanged.

    Used in hybrid architectures where some layers don't need
    attention (e.g. Mamba layers in Nemotron 3).
    """

    def __init__(self, config: BlockConfig) -> None:
        nn.Module.__init__(self)
        self.config = config

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        rope: nn.Module | None = None,
    ) -> tuple[mx.array, None]:
        """Return input unchanged with no cache."""
        return x, None


def _apply_sliding_window(
    mask: mx.array | None,
    window_size: int,
    seq_len: int,
    cache_len: int = 0,
) -> mx.array:
    """Apply sliding window constraint to a causal mask.

    Args:
        mask: Existing additive causal mask
            (seq_len, cache_len + seq_len) or None.
        window_size: Number of recent tokens each query can
            attend to (including itself).
        seq_len: Query sequence length.
        cache_len: Length of cached KV sequence.

    Returns:
        Additive mask with sliding window applied.
    """
    total_len = cache_len + seq_len
    # Column indices (key positions)
    col_idx = mx.arange(total_len)[None, :]
    # Row indices mapped to absolute positions
    row_idx = mx.arange(seq_len)[:, None] + cache_len
    # A token at position i can attend to [i - window_size + 1, i]
    window_mask = mx.where(col_idx >= (row_idx - window_size + 1), 0.0, -1e9)
    if mask is not None:
        # Combine: both causal and window constraints must pass
        return mx.maximum(mask, window_mask)
    return window_mask


@attention_registry.register("sliding_window_gqa")
class SlidingWindowGQA(AttentionBase):
    """Grouped-Query Attention with sliding window masking.

    Each token can only attend to the most recent
    ``window_size`` tokens (including itself). Uses GQA head
    configuration for memory efficiency.

    The window size is read from ``config.window_size``.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        if config.window_size is None:
            raise ValueError("SlidingWindowGQA requires config.window_size")
        self.window_size = config.window_size
        self.n_kv_heads = config.effective_n_kv_heads
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        # μP uses 1/d_head; SP uses 1/√d_head
        exp = -1.0 if config.mup else -0.5
        self.scale = self.head_dim ** exp

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        rope: nn.Module | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        cache_len = (
            cache[0].shape[2]
            if cache is not None else 0
        )

        if rope is not None:
            q, k = rope(q, k, offset=cache_len)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # Apply sliding window to the mask
        mask = _apply_sliding_window(mask, self.window_size, L, cache_len)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out), new_cache
