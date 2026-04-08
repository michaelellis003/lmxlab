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

    def _init_qk_norm(self) -> None:
        """Initialize per-head QK-norm layers if enabled.

        Creates RMSNorm(head_dim) for Q and K projections.
        OLMo 2 style: learnable gamma, applied per head.

        References:
            - OLMo 2 (allenai/OLMo-2)
            - HF transformers modeling_olmo2.py
        """
        if self.config.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def _apply_qk_norm(
        self, q: mx.array, k: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Apply per-head RMSNorm to Q and K if enabled.

        Args:
            q: Query (B, n_heads, L, head_dim).
            k: Key (B, n_kv_heads, L, head_dim).

        Returns:
            Normalized (q, k) tuple.
        """
        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k

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
        self._init_qk_norm()
        # μP uses 1/d_head; SP uses 1/√d_head
        exp = -1.0 if config.mup else -0.5
        self.scale = self.head_dim**exp

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

        q, k = self._apply_qk_norm(q, k)

        if rope is not None:
            offset = cache[0].shape[2] if cache is not None else 0
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
        self._init_qk_norm()
        # μP uses 1/d_head; SP uses 1/√d_head
        exp = -1.0 if config.mup else -0.5
        self.scale = self.head_dim**exp

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

        q, k = self._apply_qk_norm(q, k)

        if rope is not None:
            offset = cache[0].shape[2] if cache is not None else 0
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
        self._init_qk_norm()
        # μP uses 1/d_head; SP uses 1/√d_head
        exp = -1.0 if config.mup else -0.5
        self.scale = self.head_dim**exp

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

        q, k = self._apply_qk_norm(q, k)

        cache_len = cache[0].shape[2] if cache is not None else 0

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


def _apply_chunk_mask(
    mask: mx.array | None,
    seq_len: int,
    chunk_size: int,
) -> mx.array:
    """Apply block-diagonal chunked causal mask.

    Each chunk of ``chunk_size`` tokens can only attend
    within itself. Cross-chunk attention is blocked.

    Args:
        mask: Existing additive causal mask (seq_len,
            seq_len) or None.
        seq_len: Sequence length.
        chunk_size: Chunk size for local attention.

    Returns:
        Additive mask with block-diagonal structure.
    """
    pos = mx.arange(seq_len)
    chunks_q = pos[:, None] // chunk_size
    chunks_k = pos[None, :] // chunk_size
    chunk_mask = mx.where(chunks_q == chunks_k, 0.0, -1e9)
    if mask is not None:
        # Both constraints must allow: use minimum
        # (more negative = more restrictive)
        return mx.minimum(mask, chunk_mask)
    return chunk_mask


@attention_registry.register("chunked_gqa")
class ChunkedGQA(GQA):
    """Grouped-Query Attention with chunked local masking.

    Splits the sequence into fixed-size chunks and restricts
    attention to within each chunk using a block-diagonal
    causal mask. RoPE positions reset to 0 at each chunk
    boundary.

    Used by Llama 4 iRoPE: chunked layers provide local
    context, while interleaved NoPE layers (standard GQA
    without position encoding) provide cross-chunk info flow.

    The chunk size is read from ``config.attention_chunk_size``.

    References:
        - Llama 4 (Meta, 2025)
        - iRoPE (interleaved RoPE) pattern
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        if config.attention_chunk_size is None:
            raise ValueError("ChunkedGQA requires config.attention_chunk_size")
        self.chunk_size = config.attention_chunk_size

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

        q, k = self._apply_qk_norm(q, k)

        if rope is not None:
            # Apply RoPE with chunk-local positions.
            # Position within chunk = pos % chunk_size.
            q, k = self._apply_chunked_rope(q, k, rope, L)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # Apply chunked block-diagonal causal mask
        mask = _apply_chunk_mask(mask, L, self.chunk_size)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out), new_cache

    def _apply_chunked_rope(
        self,
        q: mx.array,
        k: mx.array,
        rope_module: nn.Module,
        seq_len: int,
    ) -> tuple[mx.array, mx.array]:
        """Apply RoPE with positions that reset per chunk.

        Instead of positions 0..L-1, uses positions that
        reset at each chunk boundary: 0..C-1, 0..C-1, ...

        Accesses the inner ``nn.RoPE`` via ``rope_module._rope``
        to get base frequency and dims, then applies rotation
        with chunk-local position IDs.

        Args:
            q: Query (B, H, L, head_dim).
            k: Key (B, H, L, head_dim).
            rope_module: Wrapper RoPE module with ``_rope``.
            seq_len: Sequence length.

        Returns:
            Rotated (q, k) tuple.
        """
        inner = rope_module._rope
        dims = inner.dims
        base = inner.base

        # Compute frequencies: 1 / (base^(2i/dims))
        freqs = 1.0 / (
            base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )

        # Chunk-local positions: 0, 1, ..., C-1, 0, 1, ...
        positions = mx.arange(seq_len) % self.chunk_size
        # Angles: (L, dims//2)
        angles = positions[:, None].astype(mx.float32) * freqs[None, :]
        cos_a = mx.cos(angles)[None, None, :, :]
        sin_a = mx.sin(angles)[None, None, :, :]

        d2 = dims // 2
        q1, q2 = q[..., :d2], q[..., d2:dims]
        k1, k2 = k[..., :d2], k[..., d2:dims]

        q_rot = mx.concatenate(
            [q1 * cos_a - q2 * sin_a, q2 * cos_a + q1 * sin_a],
            axis=-1,
        )
        k_rot = mx.concatenate(
            [k1 * cos_a - k2 * sin_a, k2 * cos_a + k1 * sin_a],
            axis=-1,
        )

        # If head_dim > dims, pass through extra dims
        if dims < q.shape[-1]:
            q_rot = mx.concatenate([q_rot, q[..., dims:]], axis=-1)
            k_rot = mx.concatenate([k_rot, k[..., dims:]], axis=-1)

        return q_rot, k_rot
