# Quickstart

This guide walks through building a transformer language model, running a
forward pass, and generating text. Everything runs on a single Apple Silicon
GPU via MLX -- no CUDA, no `.to(device)` calls, no boilerplate.

## 1. Create a model config

Every model starts with a `ModelConfig`, which contains a `BlockConfig`
describing the transformer block components. The simplest way is to use a
preset factory function:

```python
from lmt_metal.models.llama import llama_config

# A small LLaMA-style model for experimentation
config = llama_config(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_kv_heads=4,     # GQA: 4 KV heads shared across 8 query heads
    n_layers=6,
    d_ff=1376,
    max_seq_len=512,
)
```

Under the hood, `llama_config` returns a `ModelConfig` whose `BlockConfig`
selects the right components by name:

```python
# This is what llama_config builds internally:
from lmt_metal.core.config import BlockConfig, ModelConfig

block = BlockConfig(
    attention='gqa',       # Grouped-Query Attention
    ffn='gated',           # SwiGLU feed-forward
    norm='rms_norm',       # RMSNorm (not LayerNorm)
    position='rope',       # Rotary Position Embedding
    d_model=512,
    n_heads=8,
    n_kv_heads=4,
    d_ff=1376,
    bias=False,            # LLaMA uses no bias
    pre_norm=True,         # Norm before attention/FFN
)
config = ModelConfig(block=block, vocab_size=32000, n_layers=6)
```

You can also build a config from scratch if you want a non-standard combination.
Want MHA with RMSNorm and a standard FFN? Just change the string names:

```python
block = BlockConfig(
    attention='mha',       # Standard Multi-Head Attention
    ffn='standard',        # GELU FFN (no gating)
    norm='rms_norm',
    position='rope',
    d_model=256,
    n_heads=4,
)
```

## 2. Build the model

Once you have a config, build the model:

```python
from lmt_metal.models.base import LanguageModel

model = LanguageModel(config)
print(f'Parameters: {model.count_parameters():,}')
```

`LanguageModel` constructs an embedding layer, `n_layers` `ConfigurableBlock`
instances (each assembled from registry components), a final norm, and an
output head. If `tie_embeddings=True` (the default), the output projection
reuses the embedding weight matrix.

## 3. Forward pass

The model takes token IDs and returns logits plus KV caches:

```python
import mlx.core as mx

# Batch of 2 sequences, each 16 tokens long
tokens = mx.random.randint(0, 32000, shape=(2, 16))

logits, caches = model(tokens)
# logits shape: (2, 16, 32000) -- one distribution per position
# caches: list of (K, V) tuples, one per layer
```

Note that we never called `.to(device)` or `.cuda()`. MLX uses unified
memory -- the same arrays live on CPU and GPU simultaneously. Computation
happens on the GPU automatically.

!!! note "Lazy evaluation"
    MLX operations are lazy by default. The `logits` array above is only
    *described*, not yet computed. Call `mx.eval(logits)` to force
    evaluation, or let it happen implicitly when you read a value
    (e.g., `logits.shape` is available immediately, but `.item()` triggers
    evaluation).

## 4. Greedy text generation

For autoregressive generation, feed one token at a time and reuse the
KV cache:

```python
def generate(model, prompt_tokens, max_new_tokens=50):
    """Simple greedy generation loop."""
    tokens = prompt_tokens  # (1, prompt_len)
    cache = None

    for _ in range(max_new_tokens):
        logits, cache = model(tokens, cache=cache)
        # Take the last token's logits, pick the argmax
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)  # Force evaluation before next step
        tokens = next_token  # Feed only the new token (cache has history)

        yield next_token.item()

# Usage:
prompt = mx.array([[1, 234, 567]])  # Example token IDs
for token_id in generate(model, prompt, max_new_tokens=20):
    print(token_id, end=' ')
```

The first call processes the full prompt. Subsequent calls process only the
new token, because the KV cache stores the key/value projections from
previous positions. This is why generation is O(n) total work instead of
O(n^2).

## 5. Try a different architecture

The power of config factories: switch to DeepSeek-style MLA without changing
any model code.

```python
from lmt_metal.models.deepseek import deepseek_config

ds_config = deepseek_config(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=1376,
    kv_lora_rank=64,    # Compress KV to 64-dim latent
    q_lora_rank=128,    # Compress Q too
    rope_dim=16,        # 16 dims for RoPE, rest for nope
)

ds_model = LanguageModel(ds_config)
logits, caches = ds_model(tokens)
```

Same `LanguageModel`, same `ConfigurableBlock`, same forward pass interface.
The only difference is which attention module the registry resolves: `'gqa'`
for LLaMA, `'mla'` for DeepSeek.

## 6. Training (preview)

lmt-metal includes a `Trainer` that handles the MLX training loop:

```python
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer

train_config = TrainConfig(
    learning_rate=3e-4,
    max_steps=1000,
    batch_size=32,
    compile_step=True,   # mx.compile the training step
)

trainer = Trainer(model, train_config)
# trainer.train(train_data) to run the loop
```

The trainer uses `nn.value_and_grad` for functional gradient computation
and `mx.eval` for explicit evaluation boundaries. See
[MLX Idioms](../architecture/mlx-idioms.md) for why these patterns matter.

## 7. CLI tools

lmt-metal includes a CLI for quick architecture exploration:

```bash
# List all architectures
lmt-metal list

# Show config details
lmt-metal info llama --tiny

# Count parameters with breakdown
lmt-metal count deepseek --detail
```

## 8. Recipes

Ready-to-run scripts in the `recipes/` directory:

```bash
# Train a tiny GPT on Shakespeare
uv run python recipes/train_tiny_gpt.py

# Train LLaMA with BPE tokenization
uv run python recipes/train_llama_shakespeare.py

# Compare architectures side-by-side
uv run python recipes/compare_training.py

# Run structured experiments with logging
uv run python recipes/run_experiment.py --arch llama --seeds 3
```

## Next steps

- **[Architecture Overview](../architecture/overview.md)** -- Understand the
  config/registry/block design in depth.
- **[MLX Idioms](../architecture/mlx-idioms.md)** -- Learn the MLX patterns
  that differ from PyTorch.
- **[Models](../models/index.md)** -- Compare all 8 architectures side-by-side.
