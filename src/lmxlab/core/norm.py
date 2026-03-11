"""Normalization wrappers for registry use."""

import mlx.nn as nn

from lmxlab.core.config import BlockConfig
from lmxlab.core.registry import Registry

# Registry for normalization variants
norm_registry: Registry[type[nn.Module]] = Registry("norm")


@norm_registry.register("rms_norm")
class RMSNorm(nn.RMSNorm):
    """RMSNorm wrapper that constructs from BlockConfig."""

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config.d_model, eps=config.norm_eps)


@norm_registry.register("layer_norm")
class LayerNorm(nn.LayerNorm):
    """LayerNorm wrapper that constructs from BlockConfig."""

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config.d_model, eps=config.norm_eps)


# Convenience factory functions
def rms_norm(config: BlockConfig) -> RMSNorm:
    """Create an RMSNorm from config."""
    return RMSNorm(config)


def layer_norm(config: BlockConfig) -> LayerNorm:
    """Create a LayerNorm from config."""
    return LayerNorm(config)
