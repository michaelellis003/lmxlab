"""LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Implements LoRA as described in Hu et al. (2021): instead of
fine-tuning all weights W, learn a low-rank update W + BA where
B is (d_out, rank) and A is (d_in, rank), with rank << d_model.

Only the LoRA matrices A and B are trainable; the base weight W
is frozen. This reduces trainable parameters by 10-100x while
preserving most of the fine-tuning quality.

Example::

    from lmt_metal.core.lora import apply_lora, merge_lora
    from lmt_metal.models.convert import load_from_hf

    model, config = load_from_hf('meta-llama/Llama-3.2-1B')
    apply_lora(model, rank=8, targets=['attention'])

    # Train only LoRA parameters (~0.1% of total)
    trainer = Trainer(model, train_config)
    trainer.train(data)

    # Merge LoRA into base weights for inference
    merge_lora(model)
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path


class LoRALinear(nn.Module):
    """Linear layer with low-rank adaptation.

    Computes: y = xW^T + b + scaling * x @ A @ B^T

    where W is frozen and A, B are trainable low-rank matrices.
    B is zero-initialized so the initial output equals the base
    linear layer's output.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        rank: LoRA rank (low-rank dimension).
        alpha: LoRA scaling factor. Effective scaling = alpha/rank.
        bias: Whether the base layer has bias.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.rank = rank
        self.scaling = alpha / rank

        # Base weight (frozen)
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((output_dims,))

        # LoRA matrices (trainable)
        # A: Kaiming uniform init
        self.lora_A = mx.random.normal((input_dims, rank)) * math.sqrt(
            2 / input_dims
        )
        # B: zero init (so initial LoRA contribution is zero)
        self.lora_B = mx.zeros((rank, output_dims))

        # Freeze base weight, keep LoRA trainable
        self.freeze(keys=["weight", "bias"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Base: x @ W^T + bias
        y = x @ self.weight.T
        if "bias" in self:
            y = y + self.bias
        # LoRA: scaling * x @ A @ B
        y = y + (x @ self.lora_A @ self.lora_B) * self.scaling
        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 1.0,
    ) -> "LoRALinear":
        """Create a LoRALinear from an existing nn.Linear.

        Copies the base weight and bias, then adds LoRA matrices.

        Args:
            linear: Base linear layer to wrap.
            rank: LoRA rank.
            alpha: LoRA scaling factor.

        Returns:
            LoRALinear with the same base weights.
        """
        output_dims, input_dims = linear.weight.shape
        has_bias = "bias" in linear

        lora = cls(input_dims, output_dims, rank, alpha, bias=has_bias)
        lora.weight = linear.weight
        if has_bias:
            lora.bias = linear.bias
        # Re-freeze after setting weights
        lora.freeze(keys=["weight", "bias"], recurse=False)
        return lora

    def to_linear(self) -> nn.Linear:
        """Merge LoRA weights and return a plain nn.Linear.

        Computes W_merged = W + scaling * (A @ B)^T and returns
        a new nn.Linear with the merged weight.
        """
        has_bias = "bias" in self
        merged_weight = (
            self.weight + (self.lora_A @ self.lora_B).T * self.scaling
        )
        linear = nn.Linear(
            self.weight.shape[1],
            self.weight.shape[0],
            bias=has_bias,
        )
        linear.weight = merged_weight
        if has_bias:
            linear.bias = self.bias
        return linear


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 1.0,
    targets: list[str] | None = None,
) -> None:
    """Apply LoRA to a model's linear layers in-place.

    Replaces targeted ``nn.Linear`` layers with ``LoRALinear``,
    freezing the base weights and making only the LoRA matrices
    trainable.

    Args:
        model: Model to modify (in-place).
        rank: LoRA rank for all adapted layers.
        alpha: LoRA scaling factor.
        targets: Which submodules to target. Options:
            ``'attention'`` (q/k/v/o projections),
            ``'ffn'`` (gate/up/down projections).
            Default: ``['attention']``.
    """
    if targets is None:
        targets = ["attention"]

    def _maybe_lora(path, m):
        if not isinstance(m, nn.Linear):
            return m
        # Check if path matches any target
        for target in targets:
            if target in path:
                return LoRALinear.from_linear(m, rank, alpha)
        return m

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(
        _maybe_lora, leaves, is_leaf=nn.Module.is_module
    )
    model.update_modules(leaves)


def lora_parameters(model: nn.Module) -> dict:
    """Extract only LoRA parameters from a model.

    Useful for saving/loading just the LoRA adapter weights,
    which are much smaller than the full model.

    Args:
        model: Model with LoRA layers.

    Returns:
        Nested dict of only lora_A and lora_B parameters.
    """
    all_params = model.trainable_parameters()
    # trainable_parameters already returns only unfrozen params,
    # which for LoRA models means only lora_A and lora_B
    return all_params


def merge_lora(model: nn.Module) -> None:
    """Merge LoRA weights into base weights in-place.

    Replaces all ``LoRALinear`` layers with plain ``nn.Linear``
    layers whose weights include the LoRA contribution. After
    merging, the model has the same output but no LoRA overhead.

    Args:
        model: Model to merge (in-place).
    """

    def _maybe_merge(_path, m):
        if isinstance(m, LoRALinear):
            return m.to_linear()
        return m

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(
        _maybe_merge, leaves, is_leaf=nn.Module.is_module
    )
    model.update_modules(leaves)
