# Core

The core package provides the building blocks for all architectures.

## Configuration

::: lmt_metal.core.config.BlockConfig
    options:
      members: true

::: lmt_metal.core.config.ModelConfig
    options:
      members: true

## ConfigurableBlock

::: lmt_metal.core.block.ConfigurableBlock
    options:
      members: true

## Attention

::: lmt_metal.core.attention.MHA
    options:
      members: ["__init__", "__call__"]

::: lmt_metal.core.attention.GQA
    options:
      members: ["__init__", "__call__"]

::: lmt_metal.core.attention.SlidingWindowGQA
    options:
      members: ["__init__", "__call__"]

## Multi-Head Latent Attention

::: lmt_metal.core.mla.MLA
    options:
      members: ["__init__", "__call__"]

## Gated DeltaNet

::: lmt_metal.core.deltanet.GatedDeltaNet
    options:
      members: ["__init__", "__call__"]

## Feed-Forward Networks

::: lmt_metal.core.ffn.StandardFFN
    options:
      members: ["__init__", "__call__"]

::: lmt_metal.core.ffn.GatedFFN
    options:
      members: ["__init__", "__call__"]

## Mixture of Experts

::: lmt_metal.core.moe.MoEFFN
    options:
      members: ["__init__", "__call__"]

::: lmt_metal.core.moe.SharedExpertMoEFFN
    options:
      members: ["__init__", "__call__"]

## Normalization

::: lmt_metal.core.norm

## Position Encoding

::: lmt_metal.core.position

## Quantization

::: lmt_metal.core.quantize.quantize_model

::: lmt_metal.core.quantize.dequantize_model

## LoRA

::: lmt_metal.core.lora.LoRALinear
    options:
      members: ["__init__", "__call__", "from_linear", "to_linear"]

::: lmt_metal.core.lora.apply_lora

::: lmt_metal.core.lora.merge_lora

::: lmt_metal.core.lora.lora_parameters

## QLoRA

::: lmt_metal.core.qlora.LoRAQuantizedLinear
    options:
      members: ["__init__", "__call__", "from_quantized"]

::: lmt_metal.core.qlora.apply_qlora

## Registry

::: lmt_metal.core.registry.Registry
    options:
      members: true
