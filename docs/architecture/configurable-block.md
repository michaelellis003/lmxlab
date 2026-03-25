# Configurable Block

`ConfigurableBlock` is a single `nn.Module` that assembles a complete
transformer block from registry components based on a `BlockConfig`.
This page describes its structure and extension points.

## Anatomy of a block

Every transformer block has the same skeleton:

1. **Attention** -- compute token interactions (MHA, GQA, or MLA)
2. **Feed-forward network** -- per-token nonlinear transformation
3. **Normalization** -- stabilize activations (LayerNorm or RMSNorm)
4. **Position encoding** -- inject sequence order information (RoPE, sinusoidal, ALiBi)
5. **Residual connections** -- let gradients flow through

`ConfigurableBlock` builds these from its config at construction time:

```python
class ConfigurableBlock(nn.Module):
    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self.config = config

        # Look up classes by name, then instantiate
        attn_cls = attention_registry.get(config.attention)
        ffn_cls = ffn_registry.get(config.ffn)
        norm_cls = norm_registry.get(config.norm)

        self.attention = attn_cls(config)
        self.ffn = ffn_cls(config)
        self.attn_norm = norm_cls(config)
        self.ffn_norm = norm_cls(config)
        self.position = position_registry.get(config.position)(config)
```

The pattern is always the same: `registry.get(name)` returns a class, and
that class is constructed with the `BlockConfig`. This uniform constructor
signature is the contract that makes the registry system work.

## Pre-norm vs post-norm

The `pre_norm` flag in `BlockConfig` controls where normalization is applied
relative to the sublayer and residual connection. The choice has a
measurable effect on training stability and final quality.

### Pre-norm (LLaMA, GPT-2, most modern models)

```
residual = x
x = norm(x)          # normalize first
x = sublayer(x)      # attention or FFN
x = residual + x     # then add residual
```

```python
def _pre_norm_forward(self, x, mask, cache):
    residual = x
    h = self.attn_norm(x)
    h, new_cache = self.attention(h, mask=mask, cache=cache)
    x = residual + h

    residual = x
    h = self.ffn_norm(x)
    h = self.ffn(h)
    x = residual + h
    return x, new_cache
```

Pre-norm is more stable during training because the residual stream passes
through without normalization, preserving gradient magnitude. This is why
most modern LLMs use it.

### Post-norm (original Transformer, BERT)

```
x = sublayer(x)      # attention or FFN first
x = norm(x + residual)  # then residual + normalize
```

```python
def _post_norm_forward(self, x, mask, cache):
    h, new_cache = self.attention(x, mask=mask, cache=cache)
    x = self.attn_norm(x + h)

    h = self.ffn(x)
    x = self.ffn_norm(x + h)
    return x, new_cache
```

Post-norm can achieve slightly better final quality but requires careful
learning rate warmup to avoid early training instability.

## The forward pass

The block's `__call__` method dispatches to the appropriate path:

```python
def __call__(self, x, mask=None, cache=None):
    if self.config.pre_norm:
        return self._pre_norm_forward(x, mask, cache)
    return self._post_norm_forward(x, mask, cache)
```

Inputs and outputs follow a consistent interface:

- **Input:** `x` of shape `(batch, seq_len, d_model)`, optional `mask`, optional `cache`
- **Output:** tuple of `(output, updated_cache)` where `output` has the same shape as `x`

The cache is a tuple of two arrays (keys and values for MHA/GQA, or compressed
latents for MLA). It is `None` during training and populated during generation.

## How to add a new component

Adding a new component requires three steps. No changes to `ConfigurableBlock`
or `LanguageModel`.

### Example: adding a new attention variant

As an example, consider implementing sliding window attention:

**Step 1: Implement the module.**

```python
# src/lmxlab/core/sliding_attention.py
import mlx.core as mx
import mlx.nn as nn
from lmxlab.core.attention import AttentionBase, attention_registry
from lmxlab.core.config import BlockConfig

@attention_registry.register('sliding_window')
class SlidingWindowAttention(AttentionBase):
    """Attention with a fixed-size sliding window."""

    def __init__(self, config: BlockConfig) -> None:
        super().__init__(config)
        self.window_size = config.max_seq_len  # or add a dedicated field
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x, mask=None, cache=None):
        # Your implementation here
        ...
```

The key requirements:

- Inherit from `AttentionBase` (or at minimum, `nn.Module`)
- Accept a `BlockConfig` in `__init__`
- Match the `__call__` signature: `(x, mask, cache) -> (output, cache)`
- Register with `@attention_registry.register('your_name')`

**Step 2: Import it so the registration runs.**

```python
# In your config factory or an __init__.py:
from lmxlab.core.sliding_attention import SlidingWindowAttention  # noqa: F401
```

**Step 3: Reference it in a config.**

```python
config = BlockConfig(
    attention='sliding_window',
    ffn='gated',
    norm='rms_norm',
    position='rope',
    d_model=512,
    n_heads=8,
)
```

`ConfigurableBlock` will look up `'sliding_window'` from the
attention registry and instantiate it. The same pattern applies to FFN,
norm, and position encoding registries.

### The component contract

All registry components share a common constructor contract:

| Component | Base class | Constructor | `__call__` signature |
|-----------|-----------|-------------|---------------------|
| Attention | `AttentionBase` | `(config: BlockConfig)` | `(x, mask, cache) -> (output, cache)` |
| FFN | `FFNBase` | `(config: BlockConfig)` | `(x) -> output` |
| Norm | `nn.Module` | `(config: BlockConfig)` | `(x) -> output` |
| Position | `nn.Module` | `(config: BlockConfig)` | varies by type |

Any component that follows this contract will work with
`ConfigurableBlock` without changes to existing code.

## Registries vs. subclassing

An alternative design would define `LlamaBlock(TransformerBlock)` and
override methods. The registry approach was chosen for three reasons:

1. With 3 attention types, 2 FFN types, 2 norm types, and 3 position
   encodings, subclassing would require 36 classes to cover every
   combination. The registry approach handles all combinations with
   zero subclasses.

2. `BlockConfig(attention='gqa', ffn='gated')` states what the block does
   directly. With inheritance, the class hierarchy must be traced.

3. Each component is independent. GQA can be tested without knowing
   about the block that will contain it, and MLA can be added without
   modifying any existing attention code.
