"""QLoRA: Quantized LoRA for memory-efficient fine-tuning.

Combines 4-bit quantized base weights with trainable float16 LoRA
adapters. The base model stays in quantized form (saving ~8x memory)
while only the small LoRA matrices are stored in full precision and
updated during training.

This is the standard approach for fine-tuning large models that would
not fit in memory at full precision. On Apple Silicon with unified
memory, QLoRA lets you fine-tune models that nearly fill available
RAM by keeping base weights quantized.

Example::

    from lmt_metal.core.quantize import quantize_model
    from lmt_metal.core.qlora import apply_qlora

    model, config = load_from_hf('meta-llama/Llama-3.2-1B')
    quantize_model(model, bits=4)
    apply_qlora(model, rank=8, targets=['attention'])

    # Train only LoRA parameters (~0.1% of total)
    trainer = Trainer(model, train_config)
    trainer.train(data)
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path


class LoRAQuantizedLinear(nn.Module):
    """Quantized linear layer with low-rank adaptation.

    Computes: y = quantized_matmul(x, W_q) + bias + scaling * x @ A @ B

    where W_q is a frozen quantized weight and A, B are trainable
    float LoRA matrices. B is zero-initialized so the initial output
    equals the quantized layer's output.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        rank: LoRA rank (low-rank dimension).
        alpha: LoRA scaling factor. Effective scaling = alpha/rank.
        bias: Whether the layer has bias.
        group_size: Quantization group size.
        bits: Quantization bits.
        mode: Quantization mode.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ) -> None:
        super().__init__()

        self.rank = rank
        self.scaling = alpha / rank
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Quantized base weight (frozen) — placeholder; use from_quantized
        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((output_dims,))

        # LoRA matrices (trainable, float)
        self.lora_A = mx.random.normal((input_dims, rank)) * math.sqrt(
            2 / input_dims
        )
        self.lora_B = mx.zeros((rank, output_dims))

        # Freeze everything except LoRA
        self.freeze(
            keys=["weight", "scales", "biases", "bias"],
            recurse=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Quantized base: uses efficient quantized matmul
        y = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            y = y + self["bias"]
        # LoRA: full-precision low-rank update
        y = y + (x @ self.lora_A @ self.lora_B) * self.scaling
        return y

    @classmethod
    def from_quantized(
        cls,
        ql: nn.QuantizedLinear,
        rank: int = 8,
        alpha: float = 1.0,
    ) -> "LoRAQuantizedLinear":
        """Create from an existing QuantizedLinear layer.

        Copies the quantized weights and adds LoRA adapters.

        Args:
            ql: Quantized linear layer to wrap.
            rank: LoRA rank.
            alpha: LoRA scaling factor.

        Returns:
            LoRAQuantizedLinear with same quantized base weights.
        """
        # Infer dimensions from quantized weight
        out_dims = ql.weight.shape[0]
        in_dims = (ql.weight.shape[1] * 32) // ql.bits
        has_bias = "bias" in ql

        lora_ql = cls(
            in_dims,
            out_dims,
            rank,
            alpha,
            bias=has_bias,
            group_size=ql.group_size,
            bits=ql.bits,
            mode=ql.mode,
        )
        # Copy quantized state
        lora_ql.weight = ql.weight
        lora_ql.scales = ql.scales
        if ql.get("biases") is not None:
            lora_ql.biases = ql["biases"]
        if has_bias:
            lora_ql.bias = ql.bias

        # Re-freeze quantized params
        lora_ql.freeze(
            keys=["weight", "scales", "biases", "bias"],
            recurse=False,
        )
        return lora_ql


def apply_qlora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 1.0,
    targets: list[str] | None = None,
) -> None:
    """Apply QLoRA to a quantized model's layers in-place.

    Replaces targeted ``nn.QuantizedLinear`` layers with
    ``LoRAQuantizedLinear``, keeping base weights quantized and
    adding trainable float LoRA matrices.

    The model should already be quantized (via ``quantize_model``
    or ``nn.quantize``) before calling this.

    Args:
        model: Quantized model to modify (in-place).
        rank: LoRA rank for all adapted layers.
        alpha: LoRA scaling factor.
        targets: Which submodules to target. Options:
            ``'attention'`` (q/k/v/o projections),
            ``'ffn'`` (gate/up/down projections).
            Default: ``['attention']``.
    """
    if targets is None:
        targets = ["attention"]

    def _maybe_qlora(path: str, m: nn.Module) -> nn.Module:
        if not isinstance(m, nn.QuantizedLinear):
            return m
        for target in targets:
            if target in path:
                return LoRAQuantizedLinear.from_quantized(m, rank, alpha)
        return m

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(
        _maybe_qlora, leaves, is_leaf=nn.Module.is_module
    )
    model.update_modules(leaves)

    # Freeze everything, then unfreeze only LoRA params
    model.freeze()
    model.unfreeze(keys=["lora_A", "lora_B"])
