"""Multi-Head Latent Attention (MLA) from DeepSeek V2/V3.

MLA compresses KV representations into a low-rank latent space
before caching, reducing KV cache size by up to 28x compared to
standard MHA. The key insight: instead of caching n_kv_heads * head_dim
per token, cache a single kv_lora_rank-dimensional latent vector.

Architecture:
    1. Down-project KV: x -> c_kv (d_model -> kv_lora_rank)
    2. Cache c_kv (compressed latent, much smaller)
    3. Up-project: c_kv -> K, V (kv_lora_rank -> n_heads * head_dim)
    4. Optionally compress Q too: x -> c_q -> Q

RoPE handling: A portion of Q and K dimensions use RoPE (rope_dim),
while the rest are "nope" (no position encoding). This is the
"decoupled RoPE" approach from DeepSeek V2.

Reference: DeepSeek-V2 (arxiv.org/abs/2405.04434)
"""

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.core.attention import AttentionBase, attention_registry
from lmt_metal.core.config import BlockConfig


@attention_registry.register("mla")
class MLA(AttentionBase):
    """Multi-Head Latent Attention.

    Compresses KV into a low-rank latent for efficient caching.

    Config requirements:
        kv_lora_rank: Latent dimension for KV compression.
        q_lora_rank: Latent dimension for Q compression (optional).
        rope_dim: Dimensions allocated for RoPE (rest are nope).
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)

        kv_lora_rank = config.kv_lora_rank
        if kv_lora_rank is None:
            raise ValueError("MLA requires kv_lora_rank in BlockConfig")

        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_dim = config.rope_dim or 0
        self.nope_dim = self.head_dim - self.rope_dim

        # KV compression: down-project to latent, up-project to K and V
        self.kv_down = nn.Linear(self.d_model, kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        # Up-project latent to K (nope part) and V
        self.k_up = nn.Linear(
            kv_lora_rank, self.n_heads * self.nope_dim, bias=False
        )
        self.v_up = nn.Linear(
            kv_lora_rank, self.n_heads * self.head_dim, bias=False
        )

        # Decoupled RoPE: separate projection for rope-applied K dims
        if self.rope_dim > 0:
            self.k_rope_proj = nn.Linear(
                self.d_model, self.n_heads * self.rope_dim, bias=False
            )

        # Q projection (optionally compressed)
        if self.q_lora_rank is not None:
            self.q_down = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank)
            self.q_up = nn.Linear(
                self.q_lora_rank,
                self.n_heads * self.head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.d_model, self.d_model, bias=config.bias
            )

        # Output projection
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.scale = self.head_dim**-0.5

        # RoPE for decoupled dimensions
        if self.rope_dim > 0:
            self._rope = nn.RoPE(
                self.rope_dim, traditional=False, base=config.rope_theta
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, L, _ = x.shape

        # --- Q projection ---
        if self.q_lora_rank is not None:
            c_q = self.q_norm(self.q_down(x))
            q = self.q_up(c_q)
        else:
            q = self.q_proj(x)

        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)  # (B, n_heads, L, head_dim)

        # --- KV compression ---
        c_kv = self.kv_norm(self.kv_down(x))  # (B, L, kv_lora_rank)

        # Cache the compressed latent (much smaller than full KV)
        # Also cache rope K if using decoupled RoPE
        if self.rope_dim > 0:
            k_rope = self.k_rope_proj(x)  # (B, L, n_heads * rope_dim)
            k_rope = k_rope.reshape(
                B, L, self.n_heads, self.rope_dim
            ).transpose(0, 2, 1, 3)

            # Apply RoPE to the rope dimensions
            offset = 0
            if cache is not None:
                offset = cache[0].shape[2]
            q_rope = q[:, :, :, : self.rope_dim]
            q_nope = q[:, :, :, self.rope_dim :]
            q_rope = self._rope(q_rope, offset=offset)
            k_rope = self._rope(k_rope, offset=offset)
            q = mx.concatenate([q_rope, q_nope], axis=-1)

        if cache is not None:
            prev_c_kv = cache[0]  # (B, 1, prev_L, kv_lora_rank)
            c_kv_cached = mx.concatenate(
                [prev_c_kv.squeeze(1), c_kv], axis=1
            )  # (B, total_L, kv_lora_rank)
            if self.rope_dim > 0:
                prev_k_rope = cache[1]  # (B, n_heads, prev_L, rope_dim)
                k_rope = mx.concatenate([prev_k_rope, k_rope], axis=2)
        else:
            c_kv_cached = c_kv

        # Store compressed latent and rope K for cache
        # c_kv gets unsqueezed to (B, 1, total_L, kv_lora_rank)
        if self.rope_dim > 0:
            new_cache = (c_kv_cached[:, None, :, :], k_rope)
        else:
            c_kv_exp = c_kv_cached[:, None, :, :]
            new_cache = (c_kv_exp, c_kv_exp)

        # --- Up-project from latent ---
        k_nope = self.k_up(c_kv_cached)  # (B, total_L, n_heads * nope_dim)
        v = self.v_up(c_kv_cached)  # (B, total_L, n_heads * head_dim)

        k_nope = k_nope.reshape(B, -1, self.n_heads, self.nope_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, -1, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Combine rope and nope K dimensions
        if self.rope_dim > 0:
            k = mx.concatenate([k_rope, k_nope], axis=-1)
        else:
            k = k_nope

        # --- Attention ---
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out), new_cache
