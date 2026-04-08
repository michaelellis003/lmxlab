"""Mamba-3 SSD layer with trapezoidal discretization.

Implements the Mamba-3 architecture improvements over Mamba-2:

1. **Trapezoidal discretization**: Two sequential SSD calls
   (forward Euler + backward Euler) instead of one, improving
   approximation of the continuous ODE.
2. **BCNorm**: RMSNorm applied to B and C projections,
   analogous to QK-norm for attention. Reduces sensitivity
   to input scale.
3. **Complex A**: Data-dependent RoPE applied to B and C,
   equivalent to complex eigenvalues in the SSM.

Reuses ``_ssd_chunk_scan``, ``_mamba_conv1d``, and recurrent
scan from ``mamba2.py``.

References:
- Mamba-3 (Dao & Gu, ICLR 2026 oral)

Cross-references:
- mamba3-minimal reference implementation
- state-spaces/mamba official repo
"""

import math

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.attention import AttentionBase, attention_registry
from lmxlab.core.config import BlockConfig
from lmxlab.core.mamba2 import _mamba_conv1d, _ssd_chunk_scan


def _apply_bc_rope(
    B_mat: mx.array,
    C_mat: mx.array,
    freqs: mx.array,
    seq_len: int,
) -> tuple[mx.array, mx.array]:
    """Apply data-dependent RoPE to B and C (complex A).

    Rotates B and C using learned frequencies, equivalent
    to complex eigenvalues in the SSM state matrix A.

    Args:
        B_mat: B projection (B, L, H, N).
        C_mat: C projection (B, L, H, N).
        freqs: Learned frequencies (N // 2,).
        seq_len: Sequence length L.

    Returns:
        Rotated (B, C) tuple.
    """
    positions = mx.arange(seq_len, dtype=mx.float32)
    # angles: (L, N//2)
    angles = positions[:, None] * freqs[None, :]
    cos_a = mx.cos(angles)  # (L, N//2)
    sin_a = mx.sin(angles)

    # Reshape for broadcast: (1, L, 1, N//2)
    cos_a = cos_a[None, :, None, :]
    sin_a = sin_a[None, :, None, :]

    d2 = B_mat.shape[-1] // 2
    B1, B2 = B_mat[..., :d2], B_mat[..., d2:]
    C1, C2 = C_mat[..., :d2], C_mat[..., d2:]

    B_rot = mx.concatenate(
        [B1 * cos_a - B2 * sin_a, B2 * cos_a + B1 * sin_a],
        axis=-1,
    )
    C_rot = mx.concatenate(
        [C1 * cos_a - C2 * sin_a, C2 * cos_a + C1 * sin_a],
        axis=-1,
    )
    return B_rot, C_rot


@attention_registry.register("mamba3")
class Mamba3(AttentionBase):
    """Mamba-3 structured state-space sequence mixer.

    Extends Mamba-2 with trapezoidal discretization, BCNorm,
    and complex A. Registered as ``"mamba3"`` in the
    attention registry.

    Cache format: same as Mamba-2 (ssm_state, conv_state).

    Args:
        config: Block configuration with Mamba SSM params
            plus ``mamba_trapezoidal``, ``mamba_bc_norm``,
            ``mamba_complex_a`` flags.
    """

    def __init__(self, config: BlockConfig) -> None:
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

        # Feature flags
        self.use_trapezoidal = config.mamba_trapezoidal
        self.use_bc_norm = config.mamba_bc_norm
        self.use_complex_a = config.mamba_complex_a

        # Validate dimensions
        if self.inner_dim != n_heads * head_dim:
            raise ValueError(
                f"inner_dim ({self.inner_dim}) != "
                f"n_heads * head_dim "
                f"({n_heads} * {head_dim})"
            )

        if n_heads % self.n_groups != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible "
                f"by n_groups ({self.n_groups})"
            )

        # Input projection (same as Mamba-2)
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

        # SSM parameters (same init as Mamba-2)
        self.A_log = mx.log(
            mx.arange(1, n_heads + 1, dtype=mx.float32),
        )
        self.D = mx.ones((n_heads,))

        # dt_bias
        dt_min, dt_max = 0.001, 0.1
        dt_init = mx.exp(
            mx.random.uniform(shape=(n_heads,))
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        self.dt_bias = mx.log(mx.exp(dt_init) - 1.0)

        # BCNorm: per-group RMSNorm on B and C
        N = self.ssm_state_size
        if self.use_bc_norm:
            self.b_norm = nn.RMSNorm(N)
            self.c_norm = nn.RMSNorm(N)

        # Complex A: learned frequencies for B/C RoPE
        if self.use_complex_a:
            self.bc_freqs = mx.zeros((N // 2,))

        # Output norm and projection (same as Mamba-2)
        self.norm = nn.RMSNorm(self.inner_dim)
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
        """Forward pass with Mamba-3 enhancements.

        Args:
            x: Input (B, L, d_model).
            mask: Unused (SSM is inherently causal).
            cache: Tuple of (ssm_state, conv_state).
            rope: Unused.

        Returns:
            Tuple of (output, new_cache).
        """
        B, L, _ = x.shape
        N = self.ssm_state_size
        G = self.n_groups

        # Project input
        proj = self.in_proj(x)

        # Split into components
        bc_dim = 2 * G * N
        z = proj[:, :, : self.inner_dim]
        x_bc = proj[:, :, self.inner_dim : 2 * self.inner_dim + bc_dim]
        dt_logits = proj[:, :, 2 * self.inner_dim + bc_dim :]

        # Parse cache
        if cache is not None:
            ssm_state = cache[0]
            conv_state = cache[1]
        else:
            ssm_state = mx.zeros((B, self.n_heads, self.head_dim, N))
            conv_state = None

        # Causal conv1d
        x_bc, conv_state = _mamba_conv1d(
            x_bc,
            self.conv_weight,
            self.conv_bias,
            conv_state,
        )
        x_bc = nn.silu(x_bc)

        # Split conv output
        x_ssm = x_bc[:, :, : self.inner_dim]
        bc_raw = x_bc[:, :, self.inner_dim :]
        B_groups = bc_raw[:, :, : G * N].reshape(B, L, G, N)
        C_groups = bc_raw[:, :, G * N :].reshape(B, L, G, N)

        # Apply BCNorm (per-group RMSNorm on B and C)
        if self.use_bc_norm:
            B_groups = self.b_norm(B_groups)
            C_groups = self.c_norm(C_groups)

        # Expand groups to heads
        heads_per_group = self.n_heads // G
        if G > 1:
            B_mat = mx.repeat(B_groups, heads_per_group, axis=2)
            C_mat = mx.repeat(C_groups, heads_per_group, axis=2)
        else:
            B_mat = B_groups.reshape(B, L, N)
            C_mat = C_groups.reshape(B, L, N)

        # Ensure per-head shape
        if G == 1:
            B_mat = mx.broadcast_to(
                B_mat[:, :, None, :],
                (B, L, self.n_heads, N),
            )
            C_mat = mx.broadcast_to(
                C_mat[:, :, None, :],
                (B, L, self.n_heads, N),
            )

        # Apply complex A (RoPE on B and C)
        if self.use_complex_a:
            B_mat, C_mat = _apply_bc_rope(B_mat, C_mat, self.bc_freqs, L)

        # Discretize dt
        dt = nn.softplus(dt_logits + self.dt_bias[None, None, :])

        A = -mx.exp(self.A_log)
        D = self.D

        # Reshape x_ssm for multi-head
        x_ssm = x_ssm.reshape(B, L, self.n_heads, self.head_dim)

        # Dispatch to scan
        use_chunked = self.chunk_size <= L and cache is None

        if self.use_trapezoidal:
            out, S = self._trapezoidal_scan(
                x_ssm,
                dt,
                A,
                B_mat,
                C_mat,
                D,
                ssm_state,
                use_chunked,
            )
        elif use_chunked:
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

        # Reshape and gate
        out = out.reshape(B, L, self.inner_dim)
        out = nn.silu(z) * out
        out = self.norm(out)
        out = self.out_proj(out)

        new_cache = (S, conv_state)
        return out, new_cache

    def _trapezoidal_scan(
        self,
        x_ssm: mx.array,
        dt: mx.array,
        A: mx.array,
        B_mat: mx.array,
        C_mat: mx.array,
        D: mx.array,
        ssm_state: mx.array,
        use_chunked: bool,
    ) -> tuple[mx.array, mx.array]:
        """Trapezoidal discretization via two SSD calls.

        Combines forward Euler and backward Euler for better
        ODE approximation:
            y = SSD_forward(x) + SSD_backward(x)

        The backward Euler uses shifted B and C matrices.

        Args:
            x_ssm: (B, L, H, d).
            dt: (B, L, H).
            A: (H,) negative.
            B_mat: (B, L, H, N).
            C_mat: (B, L, H, N).
            D: (H,).
            ssm_state: (B, H, d, N).
            use_chunked: Whether to use chunked SSD.

        Returns:
            Tuple of (output, final_state).
        """
        if use_chunked:
            # Forward Euler SSD
            y_fwd, S_fwd = _ssd_chunk_scan(
                x_ssm,
                dt,
                A,
                B_mat,
                C_mat,
                D,
                self.chunk_size,
                ssm_state,
            )
            # Backward Euler: shift B by 1 position
            # (use B[t+1] instead of B[t])
            B_shift = mx.concatenate(
                [B_mat[:, 1:, :, :], B_mat[:, -1:, :, :]],
                axis=1,
            )
            # D=0 for backward term (skip connection only
            # counted once)
            D_zero = mx.zeros_like(D)
            y_bwd, S_bwd = _ssd_chunk_scan(
                x_ssm,
                dt,
                A,
                B_shift,
                C_mat,
                D_zero,
                self.chunk_size,
                ssm_state,
            )
        else:
            y_fwd, S_fwd = self._recurrent_scan(
                x_ssm,
                dt,
                A,
                B_mat,
                C_mat,
                D,
                ssm_state,
            )
            B_shift = mx.concatenate(
                [B_mat[:, 1:, :, :], B_mat[:, -1:, :, :]],
                axis=1,
            )
            D_zero = mx.zeros_like(D)
            y_bwd, S_bwd = self._recurrent_scan(
                x_ssm,
                dt,
                A,
                B_shift,
                C_mat,
                D_zero,
                ssm_state,
            )

        # Average the two estimates (trapezoidal rule)
        y = 0.5 * (y_fwd + y_bwd)
        # Use forward state as final (both converge)
        return y, S_fwd

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
        """Recurrent SSM scan (same as Mamba-2).

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
        S = ssm_state
        for t in range(L):
            x_t = x_ssm[:, t, :, :]
            B_t = B_mat[:, t, :, :]
            C_t = C_mat[:, t, :, :]
            dt_t = dt[:, t, :]

            dA = mx.exp(A[None, :] * dt_t)
            dB = dt_t[:, :, None] * B_t

            S = (
                dA[:, :, None, None] * S
                + x_t[:, :, :, None] * dB[:, :, None, :]
            )

            y_t = mx.sum(S * C_t[:, :, None, :], axis=-1)
            y_t = y_t + D[None, :, None] * x_t
            outputs.append(y_t)

        out = mx.stack(outputs, axis=1)
        return out, S
