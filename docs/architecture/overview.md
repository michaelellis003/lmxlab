# Architecture Overview

lmt-metal's central claim is that GPT, LLaMA, and DeepSeek are not different
architectures -- they are different *configurations* of the same four building
blocks. This page explains how the library makes that idea concrete.

## The key insight: configs, not subclasses

Most ML codebases define one class per architecture: `GPTModel`, `LlamaModel`,
`DeepSeekModel`. Each duplicates the transformer skeleton (embed, blocks, norm,
head) with minor variations in the block internals.

lmt-metal takes a different approach. There is one `LanguageModel` class and
one `ConfigurableBlock` class. Architecture variants are expressed as
**config factories** -- plain functions that return a `ModelConfig`:

```python
# These three calls produce the same type: ModelConfig
from lmt_metal.models.llama import llama_config
from lmt_metal.models.deepseek import deepseek_config
from lmt_metal.core.config import BlockConfig, ModelConfig

gpt_config = ModelConfig(
    block=BlockConfig(attention='mha', ffn='standard', norm='layer_norm', position='sinusoidal'),
    vocab_size=50257,
    n_layers=12,
)

llama = llama_config(d_model=4096, n_heads=32, n_kv_heads=8, n_layers=32)
deepseek = deepseek_config(d_model=5120, n_heads=128, n_layers=60, kv_lora_rank=512)
```

All three are `ModelConfig` values. All three build via `LanguageModel(config)`.
The differences are in the string names (`'mha'` vs `'gqa'` vs `'mla'`) and
numeric parameters.

## The four component registries

A `BlockConfig` names its components as strings. Those strings are resolved at
construction time by four typed registries:

| Registry | Key | Class | Used by |
|----------|-----|-------|---------|
| `attention_registry` | `'mha'` | `MHA` | GPT |
| | `'gqa'` | `GQA` | LLaMA, Mistral |
| | `'mla'` | `MLA` | DeepSeek V2/V3 |
| `ffn_registry` | `'standard'` | `StandardFFN` | GPT |
| | `'gated'` | `GatedFFN` (SwiGLU) | LLaMA, DeepSeek |
| `norm_registry` | `'layer_norm'` | `LayerNorm` | GPT |
| | `'rms_norm'` | `RMSNorm` | LLaMA, DeepSeek |
| `position_registry` | `'rope'` | `RoPE` | LLaMA, DeepSeek |
| | `'sinusoidal'` | `Sinusoidal` | GPT |
| | `'alibi'` | `ALiBi` | BLOOM |

Each registry is a `Registry[T]` instance -- a typed dictionary with a
decorator-based registration API:

```python
from lmt_metal.core.registry import Registry
from lmt_metal.core.attention import attention_registry

@attention_registry.register('gqa')
class GQA(AttentionBase):
    ...

# Later, at construction time:
attn_cls = attention_registry.get('gqa')  # Returns the GQA class
```

If you ask for a key that doesn't exist, you get a clear error listing the
available options.

## How a model is built

Here is the full construction chain, from config to runnable model:

```
ModelConfig
  |
  v
LanguageModel.__init__()
  |-- nn.Embedding(vocab_size, d_model)
  |
  |-- for each layer:
  |     ConfigurableBlock(block_config)
  |       |-- attention_registry.get(config.attention)(config)
  |       |-- ffn_registry.get(config.ffn)(config)
  |       |-- norm_registry.get(config.norm)(config)  x2
  |       |-- position_registry.get(config.position)(config)
  |
  |-- norm_registry.get(config.norm)(config)   # final norm
  |-- nn.Linear (or tied embedding)            # output head
```

Every component's constructor takes a `BlockConfig`. This uniform interface
is what makes the registry pattern work -- you can swap any component without
changing the wiring code.

## Config factories as architecture specifications

The factory functions in `lmt_metal.models` are deliberately simple. Here is
`llama_config` in its entirety:

```python
def llama_config(
    vocab_size=32000, d_model=4096, n_heads=32,
    n_kv_heads=8, n_layers=32, d_ff=11008,
    max_seq_len=4096, rope_theta=10000.0,
    tie_embeddings=False,
) -> ModelConfig:
    block = BlockConfig(
        attention='gqa', ffn='gated', norm='rms_norm',
        position='rope', d_model=d_model, n_heads=n_heads,
        n_kv_heads=n_kv_heads, d_ff=d_ff, bias=False,
        rope_theta=rope_theta, max_seq_len=max_seq_len,
        pre_norm=True,
    )
    return ModelConfig(
        block=block, vocab_size=vocab_size,
        n_layers=n_layers, tie_embeddings=tie_embeddings,
    )
```

There is no class to subclass, no abstract methods to override. If you want
a new architecture, write a function that returns a `ModelConfig`. If your
architecture needs a new attention mechanism, register it and reference it
by name.

## Per-layer block overrides

Some architectures use different block configurations at different layers
(for example, interleaving dense and MoE layers). `ModelConfig` supports
this via the `block_configs` field:

```python
dense_block = BlockConfig(attention='gqa', ffn='gated', ...)
moe_block = BlockConfig(attention='gqa', ffn='moe', ...)

config = ModelConfig(
    block=dense_block,  # default (used if block_configs is None)
    n_layers=4,
    block_configs=(dense_block, moe_block, dense_block, moe_block),
)
```

The `get_block_config(layer_idx)` method returns the per-layer config if
provided, otherwise falls back to the shared `block` config.

## What this buys you

1. **Readability.** The "architecture" of LLaMA fits in a single config dict.
   No hunting through class hierarchies.
2. **Composability.** Mix and match: GQA attention with LayerNorm, or MHA with
   RMSNorm. The registries don't care.
3. **Extensibility.** Add a new attention variant by implementing `AttentionBase`,
   registering it, and referencing it by name. No changes to `LanguageModel` or
   `ConfigurableBlock`.
4. **Comparison.** To understand the difference between LLaMA and DeepSeek,
   compare two config dicts. The structural similarities and differences are
   immediately visible.

## Next steps

- **[Configurable Block](configurable-block.md)** -- How `ConfigurableBlock`
  assembles components and handles pre-norm vs post-norm.
- **[MLX Idioms](mlx-idioms.md)** -- How the training loop and model internals
  use MLX-specific patterns.
