# First Training Run

This guide walks through training a small language model from
scratch on Apple Silicon. The result is a model that memorizes a
short text and generates completions from it.

!!! info "Prerequisites"
    - Preparing text data for training
    - Using lmxlab's `Trainer` with `mx.compile`
    - Monitoring loss curves and generating text
    - MLX lazy evaluation and unified memory in the training loop

## The full script

The complete training script is shown below. The sections that
follow explain each part.

```python
from dataclasses import replace

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.gpt import gpt_tiny
from lmxlab.training.callbacks import MetricsLogger, ThroughputMonitor
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

mx.random.seed(42)

# 1. Prepare data
text = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles, "
    "And by opposing end them."
)
tokenizer = CharTokenizer(text)
tokens = mx.array(tokenizer.encode(text), dtype=mx.int32)

# 2. Build model
config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)
model = LanguageModel(config)
mx.eval(model.parameters())

# 3. Train
train_config = TrainConfig(
    learning_rate=1e-3,
    max_steps=200,
    batch_size=4,
    log_interval=25,
    compile_step=False,
    warmup_steps=10,
)
trainer = Trainer(
    model,
    train_config,
    callbacks=[
        MetricsLogger(log_interval=25),
        ThroughputMonitor(log_interval=25, tokens_per_step=4 * 32),
    ],
)
history = trainer.train(
    batch_iterator(tokens, batch_size=4, seq_len=32)
)

# 4. Generate
prompt = mx.array([tokenizer.encode("To be")])
output = generate(model, prompt, max_tokens=60, temperature=0.8)
print(tokenizer.decode(output[0].tolist()))
```

Run it:

```bash
uv run python my_first_train.py
```

## Step-by-step walkthrough

### 1. Prepare text data

```python
from lmxlab.data.tokenizer import CharTokenizer

text = "To be, or not to be, that is the question..."
tokenizer = CharTokenizer(text)
tokens = mx.array(tokenizer.encode(text), dtype=mx.int32)
```

`CharTokenizer` maps each unique character to an integer ID.
It builds its vocabulary from the input text, so `vocab_size`
equals the number of distinct characters. For real training,
a BPE tokenizer such as `TiktokenTokenizer('gpt2')` is more
appropriate.

The `batch_iterator` takes this flat token array and creates
sliding windows of `(input, target)` pairs:

```python
from lmxlab.data.batching import batch_iterator

for x, y in batch_iterator(tokens, batch_size=4, seq_len=32):
    # x shape: (4, 32) -- 4 sequences of 32 tokens
    # y shape: (4, 32) -- shifted by 1 position
    break
```

Each target token is the next token after the corresponding
input token. This is the standard language modeling objective:
predict the next token at every position.

### 2. Build the model

```python
from dataclasses import replace
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.base import LanguageModel

config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)
model = LanguageModel(config)
mx.eval(model.parameters())
```

`gpt_tiny()` returns a `ModelConfig` with small dimensions
(d_model=64, 2 layers, 4 heads). The `vocab_size` is overridden to
match the tokenizer using `dataclasses.replace`.

The `mx.eval(model.parameters())` call materializes the
weights. MLX is lazy by default: without this call, the
random weight tensors would not be computed until
first use. Evaluating them up front establishes a clean baseline.

!!! tip "Swapping architectures"
    To use LLaMA instead of GPT, replace the config factory.
    The rest of the code is identical:

    ```python
    from lmxlab.models.llama import llama_tiny
    config = replace(llama_tiny(), vocab_size=tokenizer.vocab_size)
    ```

    The same `LanguageModel` class handles both because
    architecture differences live in `BlockConfig` string names
    (`'mha'` vs `'gqa'`, `'layer_norm'` vs `'rms_norm'`, etc.),
    resolved by the registry at construction time.

### 3. Configure and run training

```python
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

train_config = TrainConfig(
    learning_rate=1e-3,
    max_steps=200,
    batch_size=4,
    log_interval=25,
    compile_step=False,
    warmup_steps=10,
)
trainer = Trainer(model, train_config)
history = trainer.train(
    batch_iterator(tokens, batch_size=4, seq_len=32)
)
```

The `Trainer` performs the following steps:

1. Wraps the loss function with `nn.value_and_grad` for
   functional gradient computation
2. Optionally wraps the full step (forward + backward + update)
   with `mx.compile` for hardware-fused execution
3. Each step: computes loss and gradients, clips gradients,
   updates weights via AdamW
4. Calls `mx.eval(loss, model.parameters(), optimizer.state)`
   at the eval boundary, which triggers actual GPU computation

!!! note "`compile_step=False` for tiny models"
    Compilation is disabled here because the tiny model runs in
    microseconds and compilation overhead dominates. For real
    models (millions of parameters), set `compile_step=True`
    for a significant speedup. See
    [Compiled Training](../architecture/compiled-training.md).

Training config options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Peak learning rate |
| `optimizer` | `'adamw'` | Also: `'lion'`, `'adafactor'`, `'sgd'` |
| `lr_schedule` | `'cosine'` | Also: `'linear'`, `'constant'` |
| `warmup_steps` | 100 | Linear warmup before decay |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `grad_accumulation_steps` | 1 | Micro-batches per update |
| `compile_step` | True | Use `mx.compile` |

### 4. Monitor training

Add callbacks to observe training dynamics:

```python
from lmxlab.training.callbacks import (
    MetricsLogger,
    ThroughputMonitor,
)

trainer = Trainer(
    model,
    train_config,
    callbacks=[
        MetricsLogger(log_interval=25),
        ThroughputMonitor(
            log_interval=25,
            tokens_per_step=4 * 32,  # batch_size * seq_len
        ),
    ],
)
```

`MetricsLogger` prints loss and learning rate at each interval.
`ThroughputMonitor` reports tokens per second, which is useful for
comparing compiled vs uncompiled steps or different model sizes.

Expected training behavior:

- Loss starts high (approximately 4-5 for character-level,
  10-11 for BPE with large vocab), corresponding to
  `-log(1/vocab_size)`.
- Loss drops in the first 50 or so steps as the model
  learns character frequencies.
- Loss plateaus around 1.0-2.0 for this tiny dataset.
  With more data and a larger model, it would continue decreasing.

### 5. Generate text

```python
from lmxlab.models.generate import generate

prompt = mx.array([tokenizer.encode("To be")])
output = generate(
    model, prompt,
    max_tokens=60,
    temperature=0.8,
    top_k=10,
)
print(tokenizer.decode(output[0].tolist()))
```

Generation uses KV caching: the prompt is processed in one
forward pass (prefill), then each new token reuses cached
key/value projections. This reduces generation from O(n^2)
to O(n).

Sampling parameters:

| Parameter | Effect |
|-----------|--------|
| `temperature=0.0` | Greedy (always pick the most likely token) |
| `temperature=0.8` | Balanced creativity |
| `temperature=1.5` | More random, more diverse |
| `top_k=10` | Only consider the top 10 most likely tokens |
| `top_p=0.95` | Nucleus sampling (dynamic vocabulary cutoff) |
| `repetition_penalty=1.2` | Penalize tokens already generated |

For a tiny model trained on a single paragraph, the output will
approximately reproduce the training text with some variation.

## Evaluating the model

After training, measure quality with perplexity or
bits-per-byte:

```python
from lmxlab.eval.metrics import perplexity, bits_per_byte

# Create eval batches
eval_batches = list(
    batch_iterator(tokens, batch_size=2, seq_len=32, shuffle=False)
)

ppl = perplexity(model, [mx.concatenate([x, y[:, -1:]], axis=1)
                          for x, y in eval_batches])
print(f"Perplexity: {ppl:.2f}")
```

Lower perplexity indicates higher confidence in predictions.
A model that perfectly memorized the training data would
approach perplexity 1.0.

## Scaling up

The following modifications extend this example to
realistic training settings.

Larger model:

```python
from lmxlab.models.llama import llama_config

config = llama_config(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_kv_heads=4,
    n_layers=6,
    d_ff=1376,
)
```

BPE tokenizer:

```python
from lmxlab.data.tokenizer import TiktokenTokenizer

tokenizer = TiktokenTokenizer('gpt2')  # 50257 tokens
```

Real text data:

```python
from lmxlab.data.dataset import TextDataset

dataset = TextDataset('path/to/text.txt', tokenizer, seq_len=128)
```

Compiled training (for larger models):

```python
train_config = TrainConfig(
    compile_step=True,  # Fuse forward+backward+update
    learning_rate=3e-4,
    max_steps=5000,
    batch_size=32,
)
```

Gradient accumulation (when a batch does not fit in memory):

```python
train_config = TrainConfig(
    batch_size=8,
    grad_accumulation_steps=4,  # Effective batch = 32
)
```

Checkpointing:

```python
from lmxlab.training.checkpoints import save_checkpoint

save_checkpoint(model, trainer.optimizer, trainer.step, 'ckpt/')
```

## Next steps

- **[Quickstart](quickstart.md)** -- Forward passes and
  generation without training
- **[Compiled Training](../architecture/compiled-training.md)**
  -- How `mx.compile` speeds up the training loop
- **[MLX Idioms](../architecture/mlx-idioms.md)** -- Lazy
  evaluation, eval boundaries, and unified memory
- **[Recipes](../recipes/index.md)** -- 30+ scripts for
  training, fine-tuning, and evaluation
