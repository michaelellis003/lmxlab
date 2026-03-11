"""Gated DeltaNet: linear attention with delta rule.

Implements the Gated Delta Network from "Gated Delta Networks:
Improving Mamba2 with Delta Rule" (Yang et al., ICLR 2025).

Key ideas:
- Delta rule: error-correcting state updates instead of blind
  accumulation. The state matrix S predicts v from k, then
  corrects itself based on the prediction error.
- Decay gate (alpha): learned selective forgetting
- Update gate (beta): learned correction strength
- Short causal convolutions: local context on Q, K, V
- L2 normalization on Q, K for numerical stability

This implementation uses the recurrent form, which is:
- O(d^2) per token during inference (constant, no KV cache growth)
- O(n * d^2) for training (sequential, no chunkwise parallelism)

The recurrent form is simpler and sufficient for educational
purposes. Production implementations use chunkwise parallelism
for training efficiency.
"""

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.core.attention import AttentionBase, attention_registry
from lmt_metal.core.config import BlockConfig


def _l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
    """L2 normalize along the last dimension."""
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


def _causal_conv1d(
    x: mx.array,
    weight: mx.array,
    conv_state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply causal 1D convolution per head.

    Args:
        x: Input (B, H, L, D).
        weight: Conv weights (H, K, D) where K = kernel size.
        conv_state: Previous state (B, H, K-1, D) for
            autoregressive inference.

    Returns:
        Tuple of (output, new_conv_state).
    """
    B, H, L, D = x.shape
    K = weight.shape[1]

    if conv_state is not None:
        # Inference: single token with cached state
        # x is (B, H, 1, D)
        # Prepend cached state
        full = mx.concatenate([conv_state, x], axis=2)
        # full is (B, H, K, D)
        # Weighted sum: sum over kernel dimension
        # weight is (H, K, D), full is (B, H, K, D)
        out = mx.sum(full * weight[None, :, :, :], axis=2, keepdims=True)
        new_state = full[:, :, 1:, :]  # Drop oldest
        return out, new_state

    # Training: full sequence with left-padding
    # Pad left with zeros: (B, H, K-1+L, D)
    pad = mx.zeros((B, H, K - 1, D))
    padded = mx.concatenate([pad, x], axis=2)

    # Sliding window convolution via stacking shifted views
    # For each position i, gather [i, i+1, ..., i+K-1]
    stacked = mx.stack(
        [padded[:, :, i : i + L, :] for i in range(K)],
        axis=3,
    )
    # stacked: (B, H, L, K, D)
    # weight: (H, K, D) -> (1, H, 1, K, D)
    out = mx.sum(stacked * weight[None, :, None, :, :], axis=3)
    # out: (B, H, L, D)

    # New conv state = last K-1 positions of x
    if L >= K - 1:
        new_state = x[:, :, -(K - 1) :, :]
    else:
        # Pad if sequence shorter than kernel
        pad_len = K - 1 - L
        new_state = mx.concatenate([mx.zeros((B, H, pad_len, D)), x], axis=2)

    return out, new_state


@attention_registry.register("gated_deltanet")
class GatedDeltaNet(AttentionBase):
    """Gated Delta Network for linear attention.

    Uses the delta rule for error-correcting state updates
    with learned decay and update gates. The state matrix S
    has fixed size (d_k, d_v) regardless of sequence length,
    giving O(1) memory per token during inference.

    Forward pass per token:
        1. Project x -> Q, K, V, decay_logits, update_logits
        2. Apply causal convolution on Q, K, V (local context)
        3. Compute gates: alpha = sigmoid(decay), beta = sigmoid(update)
        4. L2 normalize Q, K
        5. Delta update: S = alpha * S - beta * (S @ k - v) @ k^T
        6. Output: o = S^T @ q

    Cache format: (S, conv_state) where:
        S: (B, H, head_dim, head_dim) — the state matrix
        conv_state: (B, H, K-1, head_dim) — conv history
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Gate projections (per-head scalar gates)
        self.decay_proj = nn.Linear(self.d_model, self.n_heads, bias=True)
        self.update_proj = nn.Linear(self.d_model, self.n_heads, bias=True)

        # Output gate (per-head, applied to output)
        self.out_gate_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Short causal convolution weights (per head)
        self.conv_kernel_size = config.conv_kernel_size
        self.use_short_conv = config.use_short_conv
        if self.use_short_conv:
            # (n_heads, kernel_size, head_dim)
            self.q_conv_w = (
                mx.random.normal(
                    shape=(
                        self.n_heads,
                        self.conv_kernel_size,
                        self.head_dim,
                    )
                )
                * 0.02
            )
            self.k_conv_w = (
                mx.random.normal(
                    shape=(
                        self.n_heads,
                        self.conv_kernel_size,
                        self.head_dim,
                    )
                )
                * 0.02
            )
            self.v_conv_w = (
                mx.random.normal(
                    shape=(
                        self.n_heads,
                        self.conv_kernel_size,
                        self.head_dim,
                    )
                )
                * 0.02
            )

        # Initialize gate biases to negative values so gates
        # start near 0 (conservative updates at init)
        self.decay_proj.bias = mx.full((self.n_heads,), -3.0)
        self.update_proj.bias = mx.full((self.n_heads,), -3.0)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, ...] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...] | None]:
        """Forward pass with delta rule state updates.

        Args:
            x: Input (B, L, d_model).
            mask: Unused (DeltaNet is inherently causal via
                recurrent state).
            cache: Tuple of (S, q_conv, k_conv, v_conv) for
                autoregressive inference.

        Returns:
            Tuple of (output, new_cache).
        """
        B, L, _ = x.shape

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Gate logits
        decay_logits = self.decay_proj(x)  # (B, L, H)
        update_logits = self.update_proj(x)  # (B, L, H)

        # Output gate
        out_gate = mx.sigmoid(self.out_gate_proj(x))
        # (B, L, d_model)

        # Reshape to (B, H, L, head_dim)
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)

        # Parse cache
        if cache is not None:
            S_prev = cache[0]
            q_conv_state = cache[1] if self.use_short_conv else None
            k_conv_state = cache[2] if self.use_short_conv else None
            v_conv_state = cache[3] if self.use_short_conv else None
        else:
            S_prev = mx.zeros((B, self.n_heads, self.head_dim, self.head_dim))
            q_conv_state = None
            k_conv_state = None
            v_conv_state = None

        # Apply short causal convolutions (optional)
        if self.use_short_conv:
            q, q_conv_state = _causal_conv1d(q, self.q_conv_w, q_conv_state)
            k, k_conv_state = _causal_conv1d(k, self.k_conv_w, k_conv_state)
            v, v_conv_state = _causal_conv1d(v, self.v_conv_w, v_conv_state)
            q = nn.silu(q)

        # L2 normalize Q and K
        q = _l2_normalize(q)
        k = _l2_normalize(k)

        # Compute gates: (B, L, H) -> (B, H, L, 1)
        alpha = mx.sigmoid(decay_logits)
        alpha = alpha.transpose(0, 2, 1)[:, :, :, None]
        beta = mx.sigmoid(update_logits)
        beta = beta.transpose(0, 2, 1)[:, :, :, None]

        # Recurrent delta rule over sequence
        outputs = []
        S = S_prev
        for t in range(L):
            q_t = q[:, :, t, :]  # (B, H, d)
            k_t = k[:, :, t, :]  # (B, H, d)
            v_t = v[:, :, t, :]  # (B, H, d)
            a_t = alpha[:, :, t, :]  # (B, H, 1)
            b_t = beta[:, :, t, :]  # (B, H, 1)

            # Prediction: S @ k -> (B, H, d)
            # S is (B, H, d, d), k_t is (B, H, d)
            pred = mx.sum(S * k_t[:, :, None, :], axis=-1)

            # Error: predicted - actual
            error = pred - v_t  # (B, H, d)

            # Delta update: S = alpha * S - beta * error @ k^T
            # error @ k^T: (B, H, d, 1) * (B, H, 1, d)
            correction = error[:, :, :, None] * k_t[:, :, None, :]
            S = a_t[:, :, :, None] * S - b_t[:, :, :, None] * correction

            # Output: q^T @ S -> (B, H, d)
            o_t = mx.sum(q_t[:, :, None, :] * S, axis=-1)
            outputs.append(o_t)

        # Stack outputs: (B, H, L, d)
        out = mx.stack(outputs, axis=2)

        # Reshape to (B, L, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)

        # Apply output gate
        out = out * out_gate

        # Output projection
        out = self.o_proj(out)

        # Build new cache
        new_cache: tuple[mx.array, ...]
        if self.use_short_conv:
            new_cache = (S, q_conv_state, k_conv_state, v_conv_state)
        else:
            new_cache = (S, mx.array(0), mx.array(0), mx.array(0))

        return out, new_cache
