"""Positional encoding modules: RoPE, ALiBi, Sinusoidal."""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.config import BlockConfig
from lmxlab.core.registry import Registry

# Registry for position encoding variants
position_registry: Registry[type[nn.Module]] = Registry("position")


@position_registry.register("rope")
class RoPE(nn.Module):
    """Rotary Position Embedding wrapper.

    Wraps nn.RoPE with config-driven initialization.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self._rope = nn.RoPE(
            config.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor (batch, heads, seq, head_dim).
            k: Key tensor (batch, kv_heads, seq, head_dim).
            offset: Position offset for KV cache.

        Returns:
            Tuple of (rotated_q, rotated_k).
        """
        q = self._rope(q, offset=offset)
        k = self._rope(k, offset=offset)
        return q, k


@position_registry.register("sinusoidal")
class Sinusoidal(nn.Module):
    """Sinusoidal positional encoding (added to embeddings)."""

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self._embed = nn.SinusoidalPositionalEncoding(
            config.d_model,
            full_turns=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Add sinusoidal position encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Input with positional encoding added.
        """
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        pe = self._embed(positions)  # (seq_len, d_model)
        return x + pe


@position_registry.register("alibi")
class ALiBi(nn.Module):
    """Attention with Linear Biases (Press et al. ICLR 2022, arXiv:2108.12409).

    Adds head-specific distance-based biases to the attention
    mask. Each head gets a fixed slope from a geometric
    sequence, penalizing distant tokens more strongly. Replaces
    explicit positional embeddings — no PE is added to inputs.

    Applied to the attention mask before softmax, not to Q/K.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self._alibi = nn.ALiBi()
        self.n_heads = config.n_heads

    def __call__(
        self,
        mask: mx.array | None = None,
        seq_len: int = 0,
        cache_len: int = 0,
    ) -> mx.array:
        """Create ALiBi-biased attention mask.

        Args:
            mask: Optional causal mask (T_q, T_k).
            seq_len: Query sequence length.
            cache_len: Cached key sequence length (offset).

        Returns:
            Combined ALiBi bias + mask (1, H, T_q, T_k).
        """
        t_q = mask.shape[-2] if mask is not None else seq_len
        t_k = mask.shape[-1] if mask is not None else seq_len + cache_len
        dummy = mx.zeros((1, self.n_heads, t_q, t_k))
        return self._alibi(dummy, offset=cache_len, mask=mask)


@position_registry.register("none")
class NoPosition(nn.Module):
    """No positional encoding (identity).

    Used by architectures that get position information from
    other mechanisms (e.g. causal convolutions in DeltaNet).
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        """Return input unchanged."""
        return x


# Convenience factory functions
def rope(config: BlockConfig) -> RoPE:
    """Create a RoPE module from config."""
    return RoPE(config)


def sinusoidal(config: BlockConfig) -> Sinusoidal:
    """Create a Sinusoidal module from config."""
    return Sinusoidal(config)


def alibi(config: BlockConfig) -> ALiBi:
    """Create an ALiBi module from config."""
    return ALiBi(config)
