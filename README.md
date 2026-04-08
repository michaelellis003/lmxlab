# lmxlab

Transformer language models on Apple Silicon, built with [MLX](https://ml-explore.github.io/mlx/).

[![CI](https://github.com/michaelellis003/lmxlab/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/lmxlab/actions/workflows/ci.yml)
[![Docs](https://github.com/michaelellis003/lmxlab/actions/workflows/docs.yml/badge.svg)](https://michaelellis003.github.io/lmxlab/)

## Install

```bash
pip install lmxlab
```

Requires Python 3.12+ and Apple Silicon (M1+). MLX runs on Intel/Linux too, but CPU-only.

## Usage

```python
import mlx.core as mx
from lmxlab.models.llama import llama_config
from lmxlab.models.base import LanguageModel

config = llama_config(vocab_size=32000, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6)
model = LanguageModel(config)
mx.eval(model.parameters())

tokens = mx.array([[1, 234, 567]])
logits, caches = model(tokens)
```

Architecture variants (GPT, LLaMA, DeepSeek, Gemma, Qwen, Mixtral, etc.) are config factories — same `LanguageModel` class, different settings.

## CLI

```bash
lmxlab list                    # Show available architectures
lmxlab info llama --tiny       # Config details
lmxlab count deepseek --detail # Parameter breakdown
```

## Docs

Full API docs at [michaelellis003.github.io/lmxlab](https://michaelellis003.github.io/lmxlab/).

## Development

```bash
git clone https://github.com/michaelellis003/lmxlab.git
cd lmxlab
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

## License

MIT
