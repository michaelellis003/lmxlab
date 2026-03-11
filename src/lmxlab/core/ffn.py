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
