# Installation

## From PyPI

```bash
pip install lmxlab
```

## From source (development)

```bash
git clone https://github.com/michaelellis003/lmxlab.git
cd lmxlab
pip install -e ".[dev]"
```

## Requirements

- Python 3.12+
- Apple Silicon Mac (M1/M2/M3/M4) for GPU acceleration

MLX also runs on Intel Macs and Linux using CPU, but performance
will differ.

## Optional dependencies

Install extras for additional functionality:

```bash
# BPE tokenization (tiktoken)
pip install lmxlab[tokenizers]

# HuggingFace model loading
pip install lmxlab[hf]

# Experiment tracking (MLflow)
pip install lmxlab[experiments]

# Everything for development
pip install -e ".[dev]"
```

## Verify installation

```python
import mlx.core as mx
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.base import LanguageModel

config = gpt_tiny()
model = LanguageModel(config)
mx.eval(model.parameters())

tokens = mx.array([[1, 2, 3, 4]])
logits, _ = model(tokens)
mx.eval(logits)
print(f"Output shape: {logits.shape}")  # (1, 4, vocab_size)
print("Installation OK!")
```

Or use the CLI:

```bash
lmxlab list
```

## Troubleshooting

`ImportError: libmlx.so` on Linux/Intel Mac:
MLX requires Apple Silicon for GPU support. On other platforms it falls
back to CPU, but the shared library must still be available. Ensure
`mlx>=0.25` is installed correctly: `pip install mlx`.

`ModuleNotFoundError: No module named 'tiktoken'`:
Install the tokenizers extra: `pip install lmxlab[tokenizers]`.

Slow first run:
MLX compiles computation graphs on first execution. Subsequent runs
are faster. This is expected behavior.
