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

## Registry

::: lmt_metal.core.registry.Registry
    options:
      members: true
