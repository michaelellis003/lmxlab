"""Attention modules: MHA and Grouped-Query Attention."""

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.core.config import BlockConfig
from lmt_metal.core.registry import Registry

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
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask.
            cache: Optional KV cache tuple (keys, values).

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
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

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
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
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

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out), new_cache
