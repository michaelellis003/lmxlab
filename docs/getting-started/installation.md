# Installation

## From PyPI

```bash
pip install lmxlab
```

## From source

```bash
git clone https://github.com/michaelellis003/lmxlab.git
cd lmxlab
pip install -e ".[dev]"
```

## Requirements

- Python 3.12+
- Apple Silicon Mac (M1/M2/M3/M4) for GPU acceleration

MLX also runs on Intel Macs and Linux (CPU only).

## Optional extras

```bash
pip install lmxlab[tokenizers]  # tiktoken BPE tokenization
pip install lmxlab[hf]          # HuggingFace model loading
```

## Verify

```python
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.base import LanguageModel
import mlx.core as mx

model = LanguageModel(gpt_tiny())
mx.eval(model.parameters())
logits, _ = model(mx.array([[1, 2, 3, 4]]))
print(logits.shape)  # (1, 4, vocab_size)
```

Or via CLI:

```bash
lmxlab list
```
