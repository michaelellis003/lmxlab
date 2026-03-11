"""Post-training quantization for lmxlab models.

Wraps MLX's built-in quantization to provide a simple interface
for quantizing LanguageModel instances. MLX uses affine quantization
by default: weights are stored as packed integers with per-group
scales and biases.

Supported modes:
    - ``"affine"`` (default): Standard weight-only quantization.
      Bits: 2, 4, 8. Group size: 32, 64, 128.

Example::

    from lmxlab.models.base import LanguageModel
    from lmxlab.models.llama import llama_config
    from lmxlab.core.quantize import quantize_model

    config = llama_config()
    model = LanguageModel(config)
    model.load_weights(...)

    # Quantize to 4-bit (default)
    quantize_model(model, bits=4, group_size=64)

    # Model is now ~8x smaller in memory
    logits, cache = model(tokens)
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
) -> None:
    """Quantize all Linear and Embedding layers in-place.

    Uses MLX's native quantization. Linear layers become
    ``nn.QuantizedLinear``, Embedding layers become
    ``nn.QuantizedEmbedding``. Norm layers and other modules
    are left unchanged.

    Args:
        model: Model to quantize (modified in-place).
        bits: Bits per weight (2, 4, or 8).
        group_size: Quantization group size (32, 64, or 128).
        mode: Quantization mode. Default: ``"affine"``.
    """
    nn.quantize(model, group_size=group_size, bits=bits, mode=mode)


def dequantize_model(model: nn.Module) -> None:
    """Dequantize all QuantizedLinear layers back to Linear.

    Reconstructs float weights from quantized representation.
    Useful for fine-tuning after loading quantized weights.

    Args:
        model: Model to dequantize (modified in-place).
    """

    def _maybe_dequantize(_path: str, m: nn.Module) -> nn.Module:
        if isinstance(m, nn.QuantizedLinear):
            # Reconstruct float weight from quantized form
            weight = mx.dequantize(
                m.weight,
                m.scales,
                m.get("biases"),
                m.group_size,
                m.bits,
            )
            has_bias = "bias" in m
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=has_bias)
            linear.weight = weight
            if has_bias:
                linear.bias = m.bias
            return linear
        if isinstance(m, nn.QuantizedEmbedding):
            weight = mx.dequantize(
                m.weight,
                m.scales,
                m.get("biases"),
                m.group_size,
                m.bits,
            )
            embed = nn.Embedding(weight.shape[0], weight.shape[1])
            embed.weight = weight
            return embed
        return m

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(
        _maybe_dequantize, leaves, is_leaf=nn.Module.is_module
    )
    model.update_modules(leaves)
