# Data

Tokenizers, datasets, and batching utilities for feeding data to models.

## Overview

The data pipeline follows a simple flow:

```
raw text → Tokenizer → token IDs → Dataset → batch_iterator → (x, y) batches
```

Three tokenizer implementations are provided:

- **CharTokenizer**: character-level tokenization (good for learning, no dependencies)
- **TiktokenTokenizer**: OpenAI's BPE tokenizer (GPT-2/GPT-4 vocabularies)
- **HFTokenizer**: wraps any HuggingFace `AutoTokenizer`

## Usage

```python
import mlx.core as mx
from lmxlab.data import CharTokenizer, TextDataset, batch_iterator

# Character-level tokenizer
tok = CharTokenizer("the quick brown fox")
ids = tok.encode("the fox")
print(tok.decode(ids))  # "the fox"

# Create dataset with next-token prediction targets
ds = TextDataset("the quick brown fox jumps over the lazy dog", tok, seq_len=16)
x, y = ds[0]  # x = input tokens, y = shifted targets

# Batch iterator for training
tokens = mx.array(tok.encode("..." * 1000), dtype=mx.int32)
for x_batch, y_batch in batch_iterator(tokens, batch_size=4, seq_len=32):
    # x_batch.shape == (4, 32), y_batch.shape == (4, 32)
    pass
```

## Tokenizer

::: lmxlab.data.tokenizer

## Datasets

::: lmxlab.data.dataset.TextDataset
    options:
      members: true

::: lmxlab.data.dataset.TokenDataset
    options:
      members: true

::: lmxlab.data.dataset.HFDataset
    options:
      members: true

## Batching

::: lmxlab.data.batching
