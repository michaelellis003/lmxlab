"""DeepSeek Sparse Attention (DSA).

Implements the three-branch sparse attention from DeepSeek-V3.2
(arXiv:2512.02556). Each query attends to three complementary
views of the key-value history:

1. **Compressed tokens**: average-pool KV into compressed tokens
   (stride = compress_ratio), providing a coarse global view.
2. **Selected tokens**: a learned linear scorer picks top-k
   important tokens from the full history for fine-grained
   retrieval.
3. **Sliding window**: local window attention for recent context
   (reuses window_size from config).

The three branch outputs are summed and passed through a shared
output projection.

References:
- DeepSeek-V3.2 (arXiv:2512.02556), Section 3.2
"""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.attention import GQA, attention_registry
from lmxlab.core.config import BlockConfig


@attention_registry.register("sparse_gqa")
class SparseGQA(GQA):
    """GQA with DeepSeek Sparse Attention (three branches).

    Combines compressed-token, selected-token, and sliding-window
    attention branches. Requires ``sparse_compress_ratio``,
    ``sparse_select_k``, and ``window_size`` in config.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        if config.sparse_compress_ratio is None:
            raise ValueError("SparseGQA requires config.sparse_compress_ratio")
        if config.sparse_select_k is None:
            raise ValueError("SparseGQA requires config.sparse_select_k")
        if config.window_size is None:
            raise ValueError("SparseGQA requires config.window_size")
        self.compress_ratio = config.sparse_compress_ratio
        self.select_k = config.sparse_select_k
        self.window_size = config.window_size

        kv_dim = self.n_kv_heads * self.head_dim

        # Compress branch: project pooled KV
        self.compress_k_proj = nn.Linear(
            kv_dim,
            kv_dim,
            bias=False,
        )
        self.compress_v_proj = nn.Linear(
            kv_dim,
            kv_dim,
            bias=False,
        )

        # Select branch: per-head importance scorer
        self.token_scorer = nn.Linear(
            self.d_model,
            self.n_kv_heads,
            bias=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        rope: nn.Module | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass with three-branch sparse attention.

        Args:
            x: Input (B, L, d_model).
            mask: Optional causal mask.
            cache: Optional KV cache (not used for DSA).
            rope: Optional RoPE module.

        Returns:
            Output (B, L, d_model) and KV cache.
        """
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k_4d = k.reshape(B, L, self.n_kv_heads, self.head_dim)
        k_4d = k_4d.transpose(0, 2, 1, 3)
        v_4d = v.reshape(B, L, self.n_kv_heads, self.head_dim)
        v_4d = v_4d.transpose(0, 2, 1, 3)

        q, k_4d = self._apply_qk_norm(q, k_4d)

        if rope is not None:
            offset = cache[0].shape[2] if cache is not None else 0
            q, k_4d = rope(q, k_4d, offset=offset)

        if cache is not None:
            k_4d = mx.concatenate([cache[0], k_4d], axis=2)
            v_4d = mx.concatenate([cache[1], v_4d], axis=2)
        new_cache = (k_4d, v_4d)

        # Branch 1: Compressed tokens
        out_compress = self._compress_branch(q, k, v, B, L)

        # Branch 2: Selected tokens
        out_select = self._select_branch(q, k_4d, v_4d, x, B, L)

        # Branch 3: Sliding window
        out_window = self._window_branch(q, k_4d, v_4d, L)

        # Sum all branches
        out = out_compress + out_select + out_window
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)

        return self.o_proj(out), new_cache

    def _compress_branch(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        B: int,
        L: int,
    ) -> mx.array:
        """Attend to average-pooled compressed KV tokens.

        Args:
            q: Query (B, n_heads, L, head_dim).
            k: Key before reshape (B, L, kv_dim).
            v: Value before reshape (B, L, kv_dim).
            B: Batch size.
            L: Sequence length.

        Returns:
            Branch output (B, n_heads, L, head_dim).
        """
        r = self.compress_ratio
        # Pad to multiple of compress_ratio
        pad_len = (r - L % r) % r
        if pad_len > 0:
            k_pad = mx.pad(k, [(0, 0), (0, pad_len), (0, 0)])
            v_pad = mx.pad(v, [(0, 0), (0, pad_len), (0, 0)])
        else:
            k_pad = k
            v_pad = v
        padded_len = k_pad.shape[1]

        # Average pool: (B, L_pad, kv_dim) -> (B, L_pad//r, kv_dim)
        k_pool = k_pad.reshape(B, padded_len // r, r, -1).mean(axis=2)
        v_pool = v_pad.reshape(B, padded_len // r, r, -1).mean(axis=2)

        # Project compressed KV
        k_c = self.compress_k_proj(k_pool)
        v_c = self.compress_v_proj(v_pool)

        Lc = k_c.shape[1]
        k_c = k_c.reshape(
            B,
            Lc,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)
        v_c = v_c.reshape(
            B,
            Lc,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        # Causal mask for compressed tokens: each query at pos i
        # can attend to compressed tokens covering pos <= i
        q_pos = mx.arange(L)[:, None]
        c_pos = mx.arange(Lc)[None, :] * r + (r - 1)
        cmask = mx.where(q_pos >= c_pos, 0.0, -1e9)

        return mx.fast.scaled_dot_product_attention(
            q,
            k_c,
            v_c,
            scale=self.scale,
            mask=cmask,
        )

    def _select_branch(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        x: mx.array,
        B: int,
        L: int,
    ) -> mx.array:
        """Attend to top-k important tokens per head.

        Args:
            q: Query (B, n_heads, L, head_dim).
            k: Key (B, n_kv_heads, L_total, head_dim).
            v: Value (B, n_kv_heads, L_total, head_dim).
            x: Input (B, L, d_model) for scoring.
            B: Batch size.
            L: Sequence length.

        Returns:
            Branch output (B, n_heads, L, head_dim).
        """
        L_total = k.shape[2]
        select_k = min(self.select_k, L_total)

        # Score tokens: (B, L, n_kv_heads) -> (B, n_kv_heads, L)
        scores = self.token_scorer(x).transpose(0, 2, 1)

        # For causal: only score tokens up to current position.
        # Apply causal mask to scores.
        causal = mx.arange(L_total)[None, None, :]
        # Scores should be -inf for future positions.
        # Use the full L_total for the score mask.
        scores = mx.where(
            causal < L_total,
            scores,
            mx.array(-1e9),
        )

        # Select top-k token indices per head
        # (B, n_kv_heads, select_k)
        top_idx = mx.argpartition(
            -scores,
            kth=select_k - 1,
            axis=-1,
        )[..., :select_k]

        # Gather selected KV
        # k: (B, n_kv_heads, L_total, head_dim)
        # top_idx: (B, n_kv_heads, select_k)
        idx_exp = top_idx[..., :, None]  # (B, H, K, 1)
        idx_exp = mx.broadcast_to(
            idx_exp,
            (B, self.n_kv_heads, select_k, self.head_dim),
        )
        k_sel = mx.take_along_axis(k, idx_exp, axis=2)
        v_sel = mx.take_along_axis(v, idx_exp, axis=2)

        # No causal mask needed — selected tokens are already
        # from valid (past) positions due to score masking.
        return mx.fast.scaled_dot_product_attention(
            q,
            k_sel,
            v_sel,
            scale=self.scale,
        )

    def _window_branch(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        L: int,
    ) -> mx.array:
        """Attend to recent tokens within sliding window.

        Args:
            q: Query (B, n_heads, L, head_dim).
            k: Key (B, n_kv_heads, L_total, head_dim).
            v: Value (B, n_kv_heads, L_total, head_dim).
            L: Query sequence length.

        Returns:
            Branch output (B, n_heads, L, head_dim).
        """
        from lmxlab.core.attention import _apply_sliding_window

        L_total = k.shape[2]
        cache_len = L_total - L

        # Build causal mask
        col = mx.arange(L_total)[None, :]
        row = mx.arange(L)[:, None] + cache_len
        causal = mx.where(col <= row, 0.0, -1e9)

        # Apply sliding window
        wmask = _apply_sliding_window(
            causal,
            self.window_size,
            L,
            cache_len,
        )

        return mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=wmask,
        )
