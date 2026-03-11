"""Core abstractions for transformer blocks and components."""

from lmxlab.core.attention import GQA, MHA, SlidingWindowGQA
from lmxlab.core.block import ConfigurableBlock
from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.core.deltanet import GatedDeltaNet
from lmxlab.core.ffn import GatedFFN, StandardFFN
from lmxlab.core.lora import (
    LoRALinear,
    apply_lora,
    load_lora_adapters,
    merge_lora,
    save_lora_adapters,
)
from lmxlab.core.mla import MLA
from lmxlab.core.norm import layer_norm, rms_norm
from lmxlab.core.position import alibi, rope, sinusoidal
from lmxlab.core.qlora import LoRAQuantizedLinear, apply_qlora
from lmxlab.core.quantize import dequantize_model, quantize_model
from lmxlab.core.registry import Registry

__all__ = [
    "BlockConfig",
    "ConfigurableBlock",
    "GQA",
    "GatedDeltaNet",
    "GatedFFN",
    "LoRALinear",
    "LoRAQuantizedLinear",
    "MHA",
    "MLA",
    "SlidingWindowGQA",
    "ModelConfig",
    "Registry",
    "StandardFFN",
    "apply_lora",
    "apply_qlora",
    "dequantize_model",
    "load_lora_adapters",
    "merge_lora",
    "save_lora_adapters",
    "alibi",
    "quantize_model",
    "layer_norm",
    "rope",
    "rms_norm",
    "sinusoidal",
]
