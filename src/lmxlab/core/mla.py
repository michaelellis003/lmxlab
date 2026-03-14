"""Multi-Head Latent Attention (MLA) from DeepSeek V2/V3.

MLA compresses KV representations into a low-rank latent space
before caching, reducing KV cache size dramatically compared to
standard MHA.

Architecture:
    1. Down-project: x -> (c_kv, k_pe) where c_kv is the latent
       and k_pe is a shared single-head RoPE key
    2. Cache only c_kv and k_pe (much smaller than full K, V)
    3. Up-project: c_kv -> K_nope, V (multi-head, from latent)
    4. Decoupled RoPE: apply RoPE only to k_pe and q_pe
    5. Full key: K = concat(K_nope, broadcast(k_pe))

Cache per token = kv_lora_rank + rope_dim vs 2*n_heads*head_dim
for MHA. With typical values (512 + 64 = 576 vs 2*128*128 = 32768),
this is a ~57x reduction.

Reference: DeepSeek-V2 (Bi et al., 2024, arXiv:2405.04434)

Cross-references:
- deepseek-ai/DeepSeek-V2 modeling_deepseek.py
- HuggingFace transformers modeling_deepseek_v3.py
"""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.attention import AttentionBase, attention_registry
from lmxlab.core.config import BlockConfig


@attention_registry.register("mla")
class MLA(AttentionBase):
    """Multi-Head Latent Attention.

    Compresses KV into a low-rank latent for efficient caching.
    Uses a shared single-head RoPE key (MQA-style) for the
    position-dependent portion of K.

    Config requirements:
        kv_lora_rank: Latent dimension for KV compression.
        q_lora_rank: Latent dimension for Q compression (optional).
        rope_dim: Dimensions allocated for decoupled RoPE.
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

        # KV down-projection: produces latent + shared rope key
        # in a single projection for efficiency
        kv_down_dim = kv_lora_rank
        if self.rope_dim > 0:
            kv_down_dim += self.rope_dim  # shared single-head
        self.kv_down = nn.Linear(self.d_model, kv_down_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)

        # KV up-projection: latent -> multi-head K_nope and V
        self.kv_up = nn.Linear(
            kv_lora_rank,
            self.n_heads * (self.nope_dim + self.head_dim),
            bias=False,
        )

        # Q projection (optionally compressed via LoRA)
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
                self.rope_dim,
                traditional=False,
                base=config.rope_theta,
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        rope: nn.Module | None = None,
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

        # --- KV down-projection ---
        compressed = self.kv_down(x)  # (B, L, kv_lora_rank [+ rope])

        if self.rope_dim > 0:
            # Split into latent and shared rope key
            c_kv = compressed[:, :, : self.kv_lora_rank]
            k_pe = compressed[:, :, self.kv_lora_rank :]

            # k_pe is shared single-head: (B, L, rope_dim)
            # -> (B, 1, L, rope_dim) for MQA-style broadcast
            k_pe = k_pe[:, None, :, :]

            # Apply RoPE
            offset = 0
            if cache is not None:
                offset = cache[0].shape[1]
            q_pe = q[:, :, :, : self.rope_dim]
            q_nope = q[:, :, :, self.rope_dim :]
            q_pe = self._rope(q_pe, offset=offset)
            k_pe = self._rope(k_pe, offset=offset)
            q = mx.concatenate([q_pe, q_nope], axis=-1)
        else:
            c_kv = compressed

        # Normalize latent
        c_kv = self.kv_norm(c_kv)  # (B, L, kv_lora_rank)

        # --- Caching ---
        # Cache: (c_kv, k_pe) — compressed representations
        if cache is not None:
            prev_c_kv, prev_k_pe = cache
            c_kv = mx.concatenate([prev_c_kv, c_kv], axis=1)
            if self.rope_dim > 0:
                k_pe = mx.concatenate([prev_k_pe, k_pe], axis=2)
        new_cache = (c_kv, k_pe) if self.rope_dim > 0 else (c_kv, c_kv)

        # --- KV up-projection from latent ---
        kv = self.kv_up(c_kv)
        # (B, total_L, n_heads * (nope_dim + head_dim))
        kv = kv.reshape(
            B, -1, self.n_heads, self.nope_dim + self.head_dim
        ).transpose(0, 2, 1, 3)

        k_nope = kv[:, :, :, : self.nope_dim]
        v = kv[:, :, :, self.nope_dim :]

        # Combine rope and nope K dimensions
        if self.rope_dim > 0:
            # Broadcast shared k_pe (B,1,L,rope) -> (B,n_heads,L,rope)
            k_pe_broad = mx.broadcast_to(
                k_pe,
                (B, self.n_heads, k_pe.shape[2], self.rope_dim),
            )
            k = mx.concatenate([k_pe_broad, k_nope], axis=-1)
        else:
            k = k_nope

        # --- Attention ---
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out), new_cache
