# lmt-metal

An educational MLX library for transformer language models on Apple Silicon.

## Why lmt-metal?

Most transformer implementations optimize for production at the cost of readability.
lmt-metal takes the opposite approach: every layer is implemented from scratch in
[MLX](https://ml-explore.github.io/mlx/), with the explicit goal of helping you
understand how modern language models work.

The core insight is that GPT, LLaMA, and DeepSeek are not fundamentally different
architectures. They are different *configurations* of the same building blocks:
attention, feed-forward networks, normalization, and positional encoding. lmt-metal
makes this concrete by using **config factories** instead of class hierarchies.

```python
from lmt_metal.models.llama import llama_config
from lmt_metal.models.deepseek import deepseek_config
from lmt_metal.models.base import LanguageModel

# Same LanguageModel class, different configs
llama = LanguageModel(llama_config(d_model=512, n_heads=8, n_kv_heads=4, n_layers=6))
deepseek = LanguageModel(deepseek_config(d_model=512, n_heads=8, n_layers=6, kv_lora_rank=64))
```

No subclassing. No `LlamaModel` vs `DeepSeekModel`. One `LanguageModel` class,
assembled from registry components based on what the config asks for.

## Design principles

- **Educational first.** Code is written for clarity, not maximum performance.
  Comments explain *why*, not just *what*.
- **MLX-native.** No PyTorch translation layer. Uses MLX idioms directly:
  `nn.value_and_grad`, `mx.compile`, `mx.fast.scaled_dot_product_attention`,
  unified memory.
- **Config factories, not subclasses.** Architecture variants are configs, not
  class hierarchies. A `BlockConfig` names its components by string, and a
  typed `Registry` resolves them at construction time.
- **Progressive complexity.** Start with standard MHA and LayerNorm (GPT-style),
  then swap in GQA + RMSNorm + SwiGLU (LLaMA-style), then try MLA
  (DeepSeek-style). Same model class throughout.

## Installation

```bash
pip install lmt-metal
```

Or from source:

```bash
git clone https://github.com/michaelellis003/lmt-metal.git
cd lmt-metal
pip install -e ".[dev]"
```

Requires Python 3.12+ and an Apple Silicon Mac (M1 or later) for GPU acceleration.
MLX will still run on Intel Macs and Linux using CPU, but the performance
characteristics will differ.

## Documentation overview

- **[Quickstart](getting-started/quickstart.md)** -- Build and run a model in
  under 20 lines.
- **[Architecture Overview](architecture/overview.md)** -- How configs, registries,
  and `ConfigurableBlock` fit together.
- **[Configurable Block](architecture/configurable-block.md)** -- Deep dive into
  the block that assembles components from registries.
- **[MLX Idioms](architecture/mlx-idioms.md)** -- How MLX differs from PyTorch
  and why it matters for this library.
- **[API Reference](api/index.md)** -- Auto-generated module documentation.
