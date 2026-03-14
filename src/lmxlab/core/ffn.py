"""Feed-forward network modules: Standard and Gated (SwiGLU)."""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.config import BlockConfig
from lmxlab.core.registry import Registry

# Registry for FFN variants
ffn_registry: Registry[type["FFNBase"]] = Registry("ffn")


class FFNBase(nn.Module):
    """Base class for feed-forward modules."""

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self.config = config


@ffn_registry.register("standard")
class StandardFFN(FFNBase):
    """Standard two-layer feed-forward network with GELU activation.

    FFN(x) = W2 * GELU(W1 * x + b1) + b2
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.gelu(self.up(x)))


@ffn_registry.register("gated")
class GatedFFN(FFNBase):
    """Gated feed-forward network (SwiGLU variant).

    FFN(x) = W_down * (SiLU(W_gate * x) * W_up * x)
    Used in LLaMA, Mistral, etc.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


@ffn_registry.register("relu2")
class ReluSquaredFFN(FFNBase):
    """Non-gated FFN with squared ReLU activation.

    FFN(x) = W_down * ReLU(W_up * x)^2
    From Primer (So et al. 2021). Used in Nemotron 3 for
    both dense (*) layers and MoE expert FFNs.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        self.up = nn.Linear(
            config.d_model, config.d_ff, bias=config.bias,
        )
        self.down = nn.Linear(
            config.d_ff, config.d_model, bias=config.bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with squared ReLU."""
        h = nn.relu(self.up(x))
        return self.down(h * h)


@ffn_registry.register("gated_relu2")
class GatedReluSquaredFFN(FFNBase):
    """Gated feed-forward network with squared ReLU.

    FFN(x) = W_down * (ReLU(W_gate * x)^2 * W_up * x)
    SwiGLU-style gated variant with squared ReLU instead
    of SiLU. Distinct from Primer-style non-gated relu2.
    """

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        d_ff = config.d_ff
        self.gate = nn.Linear(
            config.d_model, d_ff, bias=config.bias,
        )
        self.up = nn.Linear(
            config.d_model, d_ff, bias=config.bias,
        )
        self.down = nn.Linear(
            d_ff, config.d_model, bias=config.bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with squared ReLU gating."""
        gate = nn.relu(self.gate(x))
        return self.down(gate * gate * self.up(x))


@ffn_registry.register("none")
class NoneFFN(FFNBase):
    """Identity FFN — returns input unchanged.

    Used in hybrid architectures where some layers don't
    need a feed-forward network (e.g. Mamba layers).
    """

    def __init__(self, config: BlockConfig) -> None:
        nn.Module.__init__(self)
        self.config = config

    def __call__(self, x: mx.array) -> mx.array:
        """Return input unchanged."""
        return x
