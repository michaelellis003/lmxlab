"""Mamba-2 SSD layer: structured state-space sequence mixer.

Implements the Mamba-2 architecture from "Transformers are SSMs:
Generalized Models and Efficient Algorithms Through Structured
State Space Duality" (Dao & Gu, 2024, arXiv:2405.21060).

Key ideas:
- Selective SSM with input-dependent B, C, and dt parameters
- Multi-head structure with diagonal A (scalar per head)
- Short causal convolution for local context
- Output gating with SiLU activation

Two execution paths:
- Recurrent: O(d * N) per token, used for inference (L=1)
  and short sequences (L < chunk_size).
- Chunked SSD: O(C^2 * H * d) intra-chunk matmuls +
  O(L/C * H * d * N) inter-chunk, parallel within chunks.
  Used for training when L >= chunk_size.

Architecture diagram::

                                  +--- z (gate) ------+
  x -> in_proj -> split -> x_BC -> conv1d -> SiLU      |
                           dt -> softplus(dt+bias)      |
                                   |                    |
                             SSM(A,B,C,D,dt)            |
                                   |                    |
                           RMSNorm(SiLU(z) * ssm_out)   |
                                            |
                                         out_proj -> y

Cross-references:
- state-spaces/mamba mamba2.py (canonical)
- HuggingFace transformers modeling_mamba2.py
- Nemotron 3 (nvidia/Nemotron-H-8B) modeling_nemotron_h.py
"""

import math

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.attention import AttentionBase, attention_registry
from lmxlab.core.config import BlockConfig


def _mamba_conv1d(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    conv_state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Causal 1D convolution for Mamba.

    Args:
        x: Input (B, L, D).
        weight: Conv weights (D, K) -- depthwise.
        bias: Conv bias (D,).
        conv_state: Previous state (B, K-1, D).

    Returns:
        Tuple of (output, new_conv_state).
    """
    B, L, D = x.shape
    K = weight.shape[1]

    if conv_state is not None:
        # Inference: single token with cached state
        full = mx.concatenate([conv_state, x], axis=1)
        # full is (B, K, D)
        # Depthwise conv: weight is (D, K), transpose to (K, D)
        out = (
            mx.sum(
                full * weight.T[None, :, :],
                axis=1,
                keepdims=True,
            )
            + bias[None, None, :]
        )
        new_state = full[:, 1:, :]
        return out, new_state

    # Training: full sequence with left-padding
    pad = mx.zeros((B, K - 1, D))
    padded = mx.concatenate([pad, x], axis=1)

    # Sliding window conv via shifted views
    stacked = mx.stack(
        [padded[:, i : i + L, :] for i in range(K)],
        axis=2,
    )
    # stacked: (B, L, K, D)
    # weight: (D, K) -> (1, 1, K, D)
    out = (
        mx.sum(
            stacked * weight.T[None, None, :, :],
            axis=2,
        )
        + bias[None, None, :]
    )
    # out: (B, L, D)

    # New conv state = last K-1 positions
    if L >= K - 1:
        new_state = x[:, -(K - 1) :, :]
    else:
        pad_len = K - 1 - L
        new_state = mx.concatenate(
            [mx.zeros((B, pad_len, D)), x],
            axis=1,
        )

    return out, new_state


def _segsum(x: mx.array) -> mx.array:
    """Stable segment sum for causal decay matrix.

    Computes L[i,j] = sum(x[k] for k in range(j, i)) with
    causal masking (L[i,j] = -inf for j > i). Used to build
    the intra-chunk attention matrix from dt*A values.

    Args:
        x: Input (..., T) where T is chunk_size.

    Returns:
        Causal decay matrix (..., T, T).

    Reference: Dao & Gu (2024) Algorithm 1.
    """
    T = x.shape[-1]
    # Cumulative sum along last axis
    x_cumsum = mx.cumsum(x, axis=-1)  # (..., T)
    # L[i,j] = cumsum[i] - cumsum[j]
    # Broadcast: (..., T, 1) - (..., 1, T)
    L = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    # Causal mask: only j <= i are valid
    mask = mx.triu(
        mx.full((T, T), -1e9),
        k=1,
    )
    return L + mask


def _ssd_chunk_scan(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    chunk_size: int,
    initial_state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Chunked SSD scan for parallel training.

    Splits the sequence into chunks and uses matrix
    multiplications within chunks (GPU-parallel). Only
    the inter-chunk state propagation is sequential
    (O(L/chunk_size) steps).

    Args:
        x: Input (B, L, H, d) — multi-head SSM input.
        dt: Discretization (B, L, H).
        A: Diagonal SSM param (H,) — negative.
        B: SSM input matrix (B, L, H, N).
        C: SSM output matrix (B, L, H, N).
        D: Skip connection (H,).
        chunk_size: Chunk size C.
        initial_state: (B, H, d, N) initial SSM state.

    Returns:
        Tuple of (output, final_state) where:
            output: (B, L, H, d)
            final_state: (B, H, d, N)

    Reference: Dao & Gu (2024) Algorithm 1, "Mamba-2".
    """
    Bs, seq_len, H, d = x.shape
    N = B.shape[-1]
    C_sz = chunk_size

    # Pad to multiple of chunk_size
    pad_len = (C_sz - seq_len % C_sz) % C_sz
    if pad_len > 0:
        x = mx.concatenate(
            [x, mx.zeros((Bs, pad_len, H, d))],
            axis=1,
        )
        dt = mx.concatenate(
            [dt, mx.zeros((Bs, pad_len, H))],
            axis=1,
        )
        B = mx.concatenate(
            [B, mx.zeros((Bs, pad_len, H, N))],
            axis=1,
        )
        C = mx.concatenate(
            [C, mx.zeros((Bs, pad_len, H, N))],
            axis=1,
        )
    L_pad = x.shape[1]
    n_chunks = L_pad // C_sz

    # Reshape into chunks: (B, n_chunks, C, ...)
    x_c = x.reshape(Bs, n_chunks, C_sz, H, d)
    dt_c = dt.reshape(Bs, n_chunks, C_sz, H)
    B_c = B.reshape(Bs, n_chunks, C_sz, H, N)
    C_c = C.reshape(Bs, n_chunks, C_sz, H, N)

    # dt * A per position: (B, nc, C, H)
    dtA = dt_c * A[None, None, None, :]

    # ── Step 1: Intra-chunk attention ──
    # Build causal decay matrix L: (B, nc, H, C, C)
    # dtA: (B, nc, C, H) -> transpose to (B, nc, H, C)
    dtA_t = mx.transpose(dtA, (0, 1, 3, 2))
    L = _segsum(dtA_t)  # (B, nc, H, C, C)
    L = mx.exp(L)

    # Intra-chunk: Y_intra[i] = sum_j L[i,j] * (B[j]^T x[j]) . C[i]
    # B_c: (B, nc, C, H, N), x_c: (B, nc, C, H, d)
    # BT_x: (B, nc, C, H, N, d) -- outer product B^T @ x
    # But for SSD dual: (x @ B^T) gives (C, H, d) @ (C, H, N)^T
    # Actually: attention = C @ (L * (B^T x))
    # Compute (B^T x) as einsum: (B, nc, C, H, N) * (B, nc, C, H, d)
    # -> need (B, nc, H, C, N) and (B, nc, H, C, d)
    B_t = mx.transpose(B_c, (0, 1, 3, 2, 4))  # (B,nc,H,C,N)
    x_t = mx.transpose(x_c, (0, 1, 3, 2, 4))  # (B,nc,H,C,d)
    C_t = mx.transpose(C_c, (0, 1, 3, 2, 4))  # (B,nc,H,C,N)

    # Scale x by dt for discretization: x * dt
    dt_t = mx.transpose(dt_c, (0, 1, 3, 2))  # (B,nc,H,C)
    x_scaled = x_t * dt_t[:, :, :, :, None]  # (B,nc,H,C,d)

    # SSD dual form: Y = (C B^T * L) x_scaled
    # CB: (B, nc, H, C, C) via C @ B^T
    CB = C_t @ mx.transpose(
        B_t,
        (0, 1, 2, 4, 3),
    )  # (B,nc,H,C,C)

    # Element-wise multiply with decay: (CB * L)
    CB_L = CB * L  # (B, nc, H, C, C) -- causal attention

    # Y_intra = CB_L @ x_scaled: (B,nc,H,C,C) @ (B,nc,H,C,d)
    Y_intra = CB_L @ x_scaled  # (B, nc, H, C, d)

    # ── Step 2: Chunk-level state accumulation ──
    # Compute per-chunk SSM states by scanning across chunks.
    # For chunk c, the SSM state entering the chunk is S_c.
    # S_c+1 = decay_c * S_c + sum_t (dB_t outer x_t) within chunk
    # where decay_c = exp(sum of dtA over the chunk).

    # Chunk decay: exp(sum(dtA) per chunk)
    chunk_decay = mx.exp(
        mx.sum(dtA, axis=2),
    )  # (B, nc, H) -- total decay per chunk

    # Per-chunk contribution: sum of x_scaled outer B within chunk
    # (B, nc, H, d, N)
    # x_scaled: (B, nc, H, C, d), B_t: (B, nc, H, C, N)
    x_s_t = mx.transpose(
        x_scaled,
        (0, 1, 2, 4, 3),
    )  # (B, nc, H, d, C)
    chunk_contrib = x_s_t @ B_t  # (B, nc, H, d, N)

    # Sequential scan across chunks
    states = []
    S = initial_state  # (B, H, d, N)
    for c_idx in range(n_chunks):
        states.append(S)
        decay_c = chunk_decay[:, c_idx, :]  # (B, H)
        contrib_c = chunk_contrib[:, c_idx, :, :, :]  # (B,H,d,N)
        S = decay_c[:, :, None, None] * S + contrib_c

    # states: list of (B, H, d, N), stack -> (B, nc, H, d, N)
    states_stacked = mx.stack(states, axis=1)

    # ── Step 3: State-to-output ──
    # Y_state[i] = C[i] @ S_c (state entering chunk c)
    # with position-dependent decay within the chunk.

    # Decay from chunk start to each position within chunk:
    # cumulative dtA from start of chunk
    # dtA_t: (B, nc, H, C)
    cum_decay = mx.cumsum(dtA_t, axis=-1)  # (B,nc,H,C)
    pos_decay = mx.exp(cum_decay)  # (B, nc, H, C)

    # C_t: (B, nc, H, C, N)
    # states_stacked: (B, nc, H, d, N)
    # Y_state[i] = pos_decay[i] * C[i] @ S
    # -> (B, nc, H, C, N) @ (B, nc, H, N, d) = (B,nc,H,C,d)
    S_t = mx.transpose(
        states_stacked,
        (0, 1, 2, 4, 3),
    )  # (B, nc, H, N, d)
    Y_state_raw = C_t @ S_t  # (B, nc, H, C, d)
    Y_state = Y_state_raw * pos_decay[:, :, :, :, None]

    # ── Step 4: Combine ──
    Y = Y_intra + Y_state

    # D skip connection: Y += D * x
    Y = Y + D[None, None, :, None, None] * x_t

    # Transpose back: (B, nc, H, C, d) -> (B, nc, C, H, d)
    Y = mx.transpose(Y, (0, 1, 3, 2, 4))
    # Reshape: (B, L_pad, H, d)
    Y = Y.reshape(Bs, L_pad, H, d)

    # Trim padding
    if pad_len > 0:
        Y = Y[:, :seq_len, :, :]

    return Y, S


@attention_registry.register("mamba2")
class Mamba2(AttentionBase):
    """Mamba-2 structured state-space sequence mixer.

    Registered as an attention variant so it plugs into
    ConfigurableBlock's attention slot. Uses selective SSM
    with input-dependent discretization.

    The state has fixed size (n_heads, head_dim, ssm_state)
    regardless of sequence length, giving O(1) memory per
    token during inference.

    Cache format: (ssm_state, conv_state) where:
        ssm_state: (B, n_heads, head_dim, ssm_state_size)
        conv_state: (B, K-1, inner_dim)

    Args:
        config: Block configuration with mamba_n_heads,
            mamba_head_dim, ssm_state_size, mamba_expand.
    """

    def __init__(self, config: BlockConfig) -> None:
        # Skip AttentionBase.__init__ -- we don't use its
        # n_heads/head_dim (Mamba has its own).
        nn.Module.__init__(self)
        self.config = config
        self.d_model = config.d_model

        n_heads = config.mamba_n_heads or config.n_heads
        head_dim = config.mamba_head_dim or config.d_model // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.ssm_state_size = config.ssm_state_size
        expand = config.mamba_expand
        self.inner_dim = self.d_model * expand
        self.conv_kernel_size = config.conv_kernel_size
        self.n_groups = config.mamba_n_groups
        self.chunk_size = config.mamba_chunk_size

        # Validate dimensions
        if self.inner_dim != n_heads * head_dim:
            raise ValueError(
                f"inner_dim ({self.inner_dim}) != "
                f"n_heads * head_dim ({n_heads} * {head_dim})"
            )

        if n_heads % self.n_groups != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by "
                f"n_groups ({self.n_groups})"
            )

        # Input projection: x -> (z, x_conv_input, dt)
        # x_conv_input includes B and C components.
        # B/C are projected per group (like GQA shares KV).
        bc_dim = 2 * self.n_groups * self.ssm_state_size
        self.in_proj = nn.Linear(
            self.d_model,
            self.inner_dim + self.inner_dim + bc_dim + n_heads,
            bias=False,
        )

        # Short causal convolution (depthwise)
        conv_d = self.inner_dim + bc_dim
        self.conv_weight = (
            mx.random.normal(
                shape=(conv_d, self.conv_kernel_size),
            )
            * 0.02
        )
        self.conv_bias = mx.zeros((conv_d,))

        # SSM parameters
        # A: diagonal, log-space. Init as log(1..n_heads)
        # per reference (state-spaces/mamba).
        self.A_log = mx.log(
            mx.arange(1, n_heads + 1, dtype=mx.float32),
        )
        # D: skip connection (per head)
        self.D = mx.ones((n_heads,))

        # dt_bias: learned bias for dt discretization.
        # Init as inverse-softplus of log-uniform in
        # [dt_min, dt_max] per reference.
        dt_min, dt_max = 0.001, 0.1
        dt_init = mx.exp(
            mx.random.uniform(shape=(n_heads,))
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        # inverse softplus: log(exp(x) - 1)
        self.dt_bias = mx.log(mx.exp(dt_init) - 1.0)

        # Output norm (RMSNorm, applied after gating)
        self.norm = nn.RMSNorm(self.inner_dim)

        # Output projection
        self.out_proj = nn.Linear(
            self.inner_dim,
            self.d_model,
            bias=False,
        )

    def __call__(  # type: ignore[override]
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, ...] | None = None,
        rope: nn.Module | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...]]:
        """Forward pass with selective SSM.

        Args:
            x: Input (B, L, d_model).
            mask: Unused (SSM is inherently causal).
            cache: Tuple of (ssm_state, conv_state).
            rope: Unused (SSM doesn't use positional
                encoding).

        Returns:
            Tuple of (output, new_cache).
        """
        B, L, _ = x.shape
        N = self.ssm_state_size
        G = self.n_groups

        # Project input
        proj = self.in_proj(x)  # (B, L, proj_dim)

        # Split into components
        bc_dim = 2 * G * N
        z = proj[:, :, : self.inner_dim]
        x_bc = proj[:, :, self.inner_dim : 2 * self.inner_dim + bc_dim]
        dt_logits = proj[
            :, :, 2 * self.inner_dim + bc_dim :
        ]  # (B, L, n_heads)

        # Parse cache
        if cache is not None:
            ssm_state = cache[0]
            conv_state = cache[1]
        else:
            ssm_state = mx.zeros(
                (
                    B,
                    self.n_heads,
                    self.head_dim,
                    N,
                )
            )
            conv_state = None

        # Causal conv1d on x_bc
        x_bc, conv_state = _mamba_conv1d(
            x_bc,
            self.conv_weight,
            self.conv_bias,
            conv_state,
        )
        x_bc = nn.silu(x_bc)

        # Split conv output into x_ssm, B, C
        x_ssm = x_bc[:, :, : self.inner_dim]
        bc_raw = x_bc[:, :, self.inner_dim :]
        # bc_raw: (B, L, 2 * G * N)
        B_groups = bc_raw[:, :, : G * N]
        C_groups = bc_raw[:, :, G * N :]

        # Reshape to (B, L, G, N)
        B_groups = B_groups.reshape(B, L, G, N)
        C_groups = C_groups.reshape(B, L, G, N)

        # Expand groups to heads: (B, L, G, N) -> (B, L, H, N)
        heads_per_group = self.n_heads // G
        if G > 1:
            B_mat = mx.repeat(B_groups, heads_per_group, axis=2)
            C_mat = mx.repeat(C_groups, heads_per_group, axis=2)
        else:
            # G == 1: just squeeze the group dim
            B_mat = B_groups.reshape(B, L, N)
            C_mat = C_groups.reshape(B, L, N)

        # Discretize dt: softplus(dt + dt_bias)
        # dt_bias matches reference (state-spaces/mamba)
        dt = nn.softplus(
            dt_logits + self.dt_bias[None, None, :],
        )  # (B, L, n_heads)

        # SSM parameters
        A = -mx.exp(self.A_log)  # (n_heads,) -- negative
        D = self.D  # (n_heads,)

        # Reshape x_ssm for multi-head
        # (B, L, inner_dim) -> (B, L, n_heads, head_dim)
        x_ssm = x_ssm.reshape(
            B,
            L,
            self.n_heads,
            self.head_dim,
        )

        # Ensure B/C have per-head shape (B, L, H, N)
        # for both recurrent and chunked paths.
        if G == 1:
            B_mat = mx.broadcast_to(
                B_mat[:, :, None, :],
                (B, L, self.n_heads, N),
            )
            C_mat = mx.broadcast_to(
                C_mat[:, :, None, :],
                (B, L, self.n_heads, N),
            )

        # Dispatch: chunked SSD for long training sequences,
        # recurrent for inference (L=1) and short sequences.
        use_chunked = self.chunk_size <= L and cache is None

        if use_chunked:
            out, S = _ssd_chunk_scan(
                x_ssm,
                dt,
                A,
                B_mat,
                C_mat,
                D,
                self.chunk_size,
                ssm_state,
            )
        else:
            out, S = self._recurrent_scan(
                x_ssm,
                dt,
                A,
                B_mat,
                C_mat,
                D,
                ssm_state,
            )

        # Reshape to (B, L, inner_dim)
        out = out.reshape(B, L, self.inner_dim)

        # Gated output with RMSNorm.
        # Reference: RMSNorm(SiLU(z) * ssm_output)
        # Gate first, then normalize (norm_before_gate=False
        # in state-spaces/mamba and Nemotron 3).
        out = nn.silu(z) * out
        out = self.norm(out)

        # Output projection
        out = self.out_proj(out)

        new_cache = (S, conv_state)
        return out, new_cache

    def _recurrent_scan(
        self,
        x_ssm: mx.array,
        dt: mx.array,
        A: mx.array,
        B_mat: mx.array,
        C_mat: mx.array,
        D: mx.array,
        ssm_state: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Recurrent SSM scan (O(L) sequential).

        Args:
            x_ssm: (B, L, H, d).
            dt: (B, L, H).
            A: (H,) negative.
            B_mat: (B, L, H, N).
            C_mat: (B, L, H, N).
            D: (H,).
            ssm_state: (B, H, d, N).

        Returns:
            Tuple of (output, final_state).
        """
        Bs, L, H, d = x_ssm.shape

        outputs = []
        S = ssm_state  # (B, H, d, N)
        for t in range(L):
            x_t = x_ssm[:, t, :, :]  # (B, H, d)
            B_t = B_mat[:, t, :, :]  # (B, H, N)
            C_t = C_mat[:, t, :, :]  # (B, H, N)
            dt_t = dt[:, t, :]  # (B, H)

            # Discretize: dA = exp(A * dt)
            dA = mx.exp(A[None, :] * dt_t)
            # dB = dt * B: (B, H, 1) * (B, H, N)
            dB = dt_t[:, :, None] * B_t

            # State update: S = dA * S + x outer dB
            S = (
                dA[:, :, None, None] * S
                + x_t[:, :, :, None] * dB[:, :, None, :]
            )

            # Output: y = S @ C + D * x
            y_t = mx.sum(
                S * C_t[:, :, None, :],
                axis=-1,
            )
            y_t = y_t + D[None, :, None] * x_t
            outputs.append(y_t)

        # Stack: (B, L, H, d)
        out = mx.stack(outputs, axis=1)
        return out, S
