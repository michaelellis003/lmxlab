# Core

The core package provides the building blocks for all architectures.

## Configuration

::: lmxlab.core.config.BlockConfig
    options:
      members: true

::: lmxlab.core.config.ModelConfig
    options:
      members: true

## ConfigurableBlock

::: lmxlab.core.block.ConfigurableBlock
    options:
      members: true

## Attention

::: lmxlab.core.attention.MHA
    options:
      members: ["__init__", "__call__"]

::: lmxlab.core.attention.GQA
    options:
      members: ["__init__", "__call__"]

::: lmxlab.core.attention.SlidingWindowGQA
    options:
      members: ["__init__", "__call__"]

## Multi-Head Latent Attention

::: lmxlab.core.mla.MLA
    options:
      members: ["__init__", "__call__"]

## Gated DeltaNet

::: lmxlab.core.deltanet.GatedDeltaNet
    options:
      members: ["__init__", "__call__"]

## Feed-Forward Networks

::: lmxlab.core.ffn.StandardFFN
    options:
      members: ["__init__", "__call__"]

::: lmxlab.core.ffn.GatedFFN
    options:
      members: ["__init__", "__call__"]

## Mixture of Experts

::: lmxlab.core.moe.MoEFFN
    options:
      members: ["__init__", "__call__"]

::: lmxlab.core.moe.SharedExpertMoEFFN
    options:
      members: ["__init__", "__call__"]

## Normalization

::: lmxlab.core.norm

## Position Encoding

::: lmxlab.core.position

## Quantization

::: lmxlab.core.quantize.quantize_model

::: lmxlab.core.quantize.dequantize_model

## LoRA

::: lmxlab.core.lora.LoRALinear
    options:
      members: ["__init__", "__call__", "from_linear", "to_linear"]

::: lmxlab.core.lora.apply_lora

::: lmxlab.core.lora.merge_lora

::: lmxlab.core.lora.lora_parameters

::: lmxlab.core.lora.save_lora_adapters

::: lmxlab.core.lora.load_lora_adapters

## QLoRA

::: lmxlab.core.qlora.LoRAQuantizedLinear
    options:
      members: ["__init__", "__call__", "from_quantized"]

::: lmxlab.core.qlora.apply_qlora

## Registry

::: lmxlab.core.registry.Registry
    options:
      members: true
