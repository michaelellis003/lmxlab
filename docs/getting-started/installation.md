# Installation

## From PyPI

```bash
pip install lmt-metal
```

## From source (development)

```bash
git clone https://github.com/michaelellis003/lmt-metal.git
cd lmt-metal
pip install -e ".[dev]"
```

## Requirements

- **Python 3.12+**
- **Apple Silicon Mac** (M1/M2/M3/M4) for GPU acceleration

MLX will also run on Intel Macs and Linux using CPU, but performance
will differ significantly.

## Optional dependencies

Install extras for additional functionality:

```bash
# BPE tokenization (tiktoken)
pip install lmt-metal[tokenizers]

# HuggingFace model loading
pip install lmt-metal[hf]

# Experiment tracking (MLflow)
pip install lmt-metal[experiments]

# Everything for development
pip install -e ".[dev]"
```

## Verify installation

```python
import mlx.core as mx
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.models.base import LanguageModel

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
lmt-metal list
```

## Troubleshooting

**`ImportError: libmlx.so` on Linux/Intel Mac:**
MLX requires Apple Silicon for GPU support. On other platforms it falls
back to CPU, but the shared library must still be available. Ensure
`mlx>=0.25` installed correctly: `pip install mlx`.

**`ModuleNotFoundError: No module named 'tiktoken'`:**
Install the tokenizers extra: `pip install lmt-metal[tokenizers]`.

**Slow first run:**
MLX compiles computation graphs on first execution. Subsequent runs
will be faster. This is normal.
