# lmxlab

Transformer language models on Apple Silicon, built with [MLX](https://ml-explore.github.io/mlx/).

A single `LanguageModel` class handles all architectures (GPT, LLaMA, DeepSeek, Gemma, Qwen, Mixtral, and more). Switching architectures is a config change — no subclassing needed.

```python
from lmxlab.models.llama import llama_config
from lmxlab.models.base import LanguageModel

model = LanguageModel(llama_config(d_model=512, n_heads=8, n_kv_heads=4, n_layers=6))
```

## Getting started

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)

## API Reference

- [Core](api/core.md) — blocks, attention, FFN, normalization, position encodings, LoRA
- [Models](api/models.md) — `LanguageModel`, config factories, generation
- [Training](api/training.md) — trainer, optimizers, checkpoints, callbacks
- [Data](api/data.md) — tokenizers, datasets, batching
- [Eval](api/eval.md) — perplexity, bits-per-byte
- [Inference](api/inference.md) — sampling, speculative decoding
