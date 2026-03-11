# Your First Training Run

This guide walks through training a small language model from
scratch on Apple Silicon. By the end you will have a model that
memorizes a short text and generates completions from it.

!!! info "What you'll learn"
    - How to prepare text data for training
    - How lmxlab's `Trainer` works with `mx.compile`
    - How to monitor loss curves and generate text
    - How MLX's lazy evaluation and unified memory simplify the loop

## The full script

Here is the complete training script. The sections below explain
each part.

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
equals the number of distinct characters. For real training
you would use a BPE tokenizer like `TiktokenTokenizer('gpt2')`.

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
(d_model=64, 2 layers, 4 heads). We override `vocab_size` to
match our tokenizer using `dataclasses.replace`.

The `mx.eval(model.parameters())` call materializes the
weights. MLX is lazy by default -- without this call, the
random weight tensors would not actually be computed until
first use. Evaluating them up front gives a clean baseline.

!!! tip "Why not LLaMA?"
    `gpt_tiny()` is fine for a first test. To try LLaMA instead,
    swap in `llama_tiny()` -- the rest of the code is identical:

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

**What happens inside `Trainer`:**

1. Wraps the loss function with `nn.value_and_grad` for
   functional gradient computation
2. Optionally wraps the full step (forward + backward + update)
   with `mx.compile` for hardware-fused execution
3. Each step: compute loss and gradients, clip gradients,
   update weights via AdamW
4. Calls `mx.eval(loss, model.parameters(), optimizer.state)`
   at the eval boundary -- this is where MLX actually runs the
   computation graph on the GPU

!!! note "`compile_step=False` for tiny models"
    We disable compilation here because the tiny model runs in
    microseconds and compilation overhead dominates. For real
    models (millions of parameters), set `compile_step=True`
    for a significant speedup. See
    [Compiled Training](../architecture/compiled-training.md).

**Training config options:**

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

Add callbacks to see what is happening:

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
`ThroughputMonitor` reports tokens per second -- useful for
comparing compiled vs uncompiled steps, or different model
sizes on your hardware.

**What to expect:**

- **Loss starts high** (~4-5 for character-level, ~10-11 for
  BPE with large vocab). This is `-log(1/vocab_size)`.
- **Loss drops quickly** in the first ~50 steps as the model
  learns character frequencies.
- **Loss plateaus** around 1.0-2.0 for this tiny dataset.
  With more data and a larger model, it would continue dropping.

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
key/value projections. This makes generation O(n) instead
of O(n^2).

**Sampling parameters:**

| Parameter | Effect |
|-----------|--------|
| `temperature=0.0` | Greedy (always pick the most likely token) |
| `temperature=0.8` | Balanced creativity |
| `temperature=1.5` | More random, more diverse |
| `top_k=10` | Only consider the top 10 most likely tokens |
| `top_p=0.95` | Nucleus sampling (dynamic vocabulary cutoff) |
| `repetition_penalty=1.2` | Penalize tokens already generated |

For a tiny model trained on a paragraph, expect the output to
roughly reproduce the training text with some variations.

## Evaluating your model

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

Lower perplexity means the model is more confident in its
predictions. A perfect model that memorized the training data
would approach perplexity 1.0.

## Scaling up

Once the basics work, here are natural next steps:

**Bigger model:**

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

**BPE tokenizer:**

```python
from lmxlab.data.tokenizer import TiktokenTokenizer

tokenizer = TiktokenTokenizer('gpt2')  # 50257 tokens
```

**Real text data:**

```python
from lmxlab.data.dataset import TextDataset

dataset = TextDataset('path/to/text.txt', tokenizer, seq_len=128)
```

**Compiled training (for real models):**

```python
train_config = TrainConfig(
    compile_step=True,  # Fuse forward+backward+update
    learning_rate=3e-4,
    max_steps=5000,
    batch_size=32,
)
```

**Gradient accumulation (when batch doesn't fit in memory):**

```python
train_config = TrainConfig(
    batch_size=8,
    grad_accumulation_steps=4,  # Effective batch = 32
)
```

**Checkpointing:**

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
- **[Recipes](../recipes/index.md)** -- 30+ ready-to-run
  scripts for training, fine-tuning, and evaluation
