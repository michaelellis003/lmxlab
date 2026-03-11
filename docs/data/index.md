# Data Pipeline

lmt-metal's data pipeline follows a simple flow: raw text goes through
a tokenizer, gets wrapped in a dataset, and is yielded as batches for
training. No multiprocessing needed -- MLX's unified memory means data
is already on the GPU.

## Tokenizers

All tokenizers implement the `Tokenizer` protocol: `encode()`,
`decode()`, and `vocab_size`. This makes them interchangeable.

### CharTokenizer

Character-level tokenizer for testing and small experiments:

```python
from lmt_metal.data import CharTokenizer

# Build vocabulary from your text
text = open('data/shakespeare.txt').read()
tok = CharTokenizer(text)
print(f'Vocab size: {tok.vocab_size}')  # ~65 for Shakespeare

ids = tok.encode('To be or not to be')
print(tok.decode(ids))  # 'To be or not to be'
```

You can also create one with default ASCII characters (no text needed):

```python
tok = CharTokenizer()  # ASCII printable (95 chars)
```

### TiktokenTokenizer

BPE tokenizer using OpenAI's tiktoken (requires `pip install tiktoken`):

```python
from lmt_metal.data import TiktokenTokenizer

# GPT-2 encoding (50257 tokens, good default)
tok = TiktokenTokenizer('gpt2')
ids = tok.encode('Hello, world!')
print(ids)  # [15496, 11, 995, 0]

# GPT-4 encoding (larger vocabulary)
tok = TiktokenTokenizer('cl100k_base')
```

### HFTokenizer

Wraps a HuggingFace `AutoTokenizer` (requires `pip install transformers`):

```python
from lmt_metal.data import HFTokenizer

# Use the tokenizer from a pretrained model
tok = HFTokenizer('meta-llama/Llama-3.2-1B')
ids = tok.encode('Hello, world!')
print(tok.decode(ids))

# Access special tokens
print(f'EOS: {tok.eos_token_id}, BOS: {tok.bos_token_id}')
```

This is the tokenizer to use when working with models loaded via
`load_from_hf()`.

### Custom Tokenizers

Any object implementing the `Tokenizer` protocol works:

```python
from lmt_metal.data import Tokenizer

class MyTokenizer:
    @property
    def vocab_size(self) -> int:
        return 1000

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...
```

## Datasets

### TextDataset

Takes raw text and a tokenizer, creates sliding windows of
(input, target) pairs:

```python
from lmt_metal.data import TextDataset, CharTokenizer

text = open('data/shakespeare.txt').read()
tok = CharTokenizer(text)

dataset = TextDataset(text, tok, seq_len=128)
print(f'{len(dataset)} training windows')

# Get a single (input, target) pair
x, y = dataset[0]
# x: tokens[0:128], y: tokens[1:129]
```

The target is the input shifted by one position -- standard
next-token prediction.

### TokenDataset

If you already have token IDs (e.g., pre-tokenized data):

```python
import mlx.core as mx
from lmt_metal.data import TokenDataset

tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dataset = TokenDataset(tokens, seq_len=4)
x, y = dataset[0]
# x: [1, 2, 3, 4], y: [2, 3, 4, 5]
```

### HFDataset

Load data directly from HuggingFace datasets (requires `pip install datasets`):

```python
from lmt_metal.data import HFDataset, HFTokenizer

# Use a HuggingFace tokenizer with a HuggingFace dataset
tok = HFTokenizer('meta-llama/Llama-3.2-1B')
ds = HFDataset('wikitext', tok, seq_len=128, config_name='wikitext-2-raw-v1')

# Stream batches for training
for inputs, targets in ds.batch_iterator(batch_size=8, max_batches=100):
    # inputs shape: (8, 128)
    # targets shape: (8, 128)
    logits, _ = model(inputs)
    ...
```

HFDataset supports streaming mode for large datasets that don't fit in memory:

```python
ds = HFDataset(
    'wikitext', tok, seq_len=128,
    config_name='wikitext-2-raw-v1',
    streaming=True,
)
```

You can also iterate over individual tokens:

```python
for token_id in ds.token_iterator():
    print(token_id)
```

## Batching

`batch_iterator` creates non-overlapping windows from a flat token
array and yields shuffled batches:

```python
from lmt_metal.data import batch_iterator, CharTokenizer

text = open('data/shakespeare.txt').read()
tok = CharTokenizer(text)
tokens = mx.array(tok.encode(text))

# Iterate through batches
for x, y in batch_iterator(tokens, batch_size=32, seq_len=128):
    # x shape: (32, 128)
    # y shape: (32, 128)
    logits, _ = model(x)
    ...
```

The iterator:

1. Splits the token array into non-overlapping sequences of length `seq_len`
2. Shuffles the sequences (if `shuffle=True`, the default)
3. Yields batches of `batch_size` sequences

!!! note "No DataLoader needed"
    Unlike PyTorch, there's no need for `DataLoader` with
    `num_workers`. MLX uses unified memory -- the same arrays live
    on CPU and GPU simultaneously. A simple Python iterator is
    sufficient.

## End-to-End Example

Putting it all together to train a small model:

```python
import mlx.core as mx
from lmt_metal.data import CharTokenizer, batch_iterator
from lmt_metal.models import LanguageModel
from lmt_metal.models.gpt import gpt_config
from lmt_metal.training import Trainer, TrainConfig

# 1. Load and tokenize text
text = open('data/shakespeare.txt').read()
tok = CharTokenizer(text)

# 2. Create model matching tokenizer vocab
config = gpt_config(
    vocab_size=tok.vocab_size,
    d_model=128, n_heads=4, n_layers=4,
)
model = LanguageModel(config)

# 3. Set up training
tokens = mx.array(tok.encode(text))
train_config = TrainConfig(
    learning_rate=1e-3,
    max_steps=500,
    batch_size=32,
)
trainer = Trainer(model, train_config)

# 4. Train
batches = batch_iterator(tokens, batch_size=32, seq_len=128)
history = trainer.train(batches)

# 5. Generate
from lmt_metal.models import stream_generate

prompt = mx.array([tok.encode('HAMLET:\n')])
for token_id in stream_generate(
    model, prompt, max_tokens=200,
    temperature=0.8,
):
    print(tok.decode([token_id]), end='', flush=True)
```

## Choosing a Tokenizer

| Tokenizer | Vocab Size | Best For |
|---|---|---|
| `CharTokenizer` | ~65-95 | Testing, tiny experiments, debugging |
| `TiktokenTokenizer('gpt2')` | 50,257 | General text, GPT-style models |
| `TiktokenTokenizer('cl100k_base')` | 100,256 | Large models, multilingual |
| `HFTokenizer('repo-id')` | Varies | Pretrained HuggingFace models |

For real training, BPE tokenizers produce better results because
they capture subword patterns. Use `HFTokenizer` when working with
pretrained models from HuggingFace (loaded via `load_from_hf`).
Character tokenizers are useful for fast iteration and testing.
