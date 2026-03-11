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

## 4. Text generation

lmt-metal provides built-in generation with KV caching, sampling strategies,
and stop tokens:

```python
from lmt_metal.models import generate

prompt = mx.array([[1, 234, 567]])  # Token IDs

# Greedy decoding (temperature=0)
output = generate(model, prompt, max_tokens=20, temperature=0.0)
# output shape: (1, 23) -- 3 prompt + 20 generated

# Top-k sampling with temperature
output = generate(
    model, prompt, max_tokens=50,
    temperature=0.8, top_k=40,
)

# Nucleus (top-p) sampling
output = generate(
    model, prompt, max_tokens=50,
    temperature=0.9, top_p=0.95,
)

# Stop at specific token IDs (e.g., EOS)
output = generate(
    model, prompt, max_tokens=100,
    stop_tokens=[0, 2],  # Stop when token 0 or 2 is generated
)

# Repetition penalty (> 1.0 discourages repeats)
output = generate(
    model, prompt, max_tokens=50,
    temperature=0.8, repetition_penalty=1.2,
)
```

### Streaming generation

For interactive applications, `stream_generate` yields tokens one at a
time as they are produced:

```python
from lmt_metal.models import stream_generate

prompt = mx.array([[1, 234, 567]])
for token_id in stream_generate(
    model, prompt, max_tokens=50,
    temperature=0.8, stop_tokens=[0],
):
    print(token_id, end=' ', flush=True)
```

Both functions use KV caching internally -- the first call processes the
full prompt (prefill), then each subsequent token reuses cached key/value
projections. This makes generation O(n) total work instead of O(n^2).

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

# Ablation study: GPT → LLaMA one feature at a time
uv run python recipes/ablation_gpt_to_llama.py --steps 200

# Load a pretrained HuggingFace model (requires huggingface_hub)
uv run python recipes/load_pretrained.py --repo meta-llama/Llama-3.2-1B

# Fine-tune with LoRA (parameter-efficient, ~0.1% trainable)
uv run python recipes/finetune_lora.py --rank 8 --steps 200

# Fine-tune with QLoRA (4-bit base + LoRA, maximum memory efficiency)
uv run python recipes/finetune_qlora.py --rank 8 --bits 4

# Train a Mixture of Experts model (dense vs MoE comparison)
uv run python recipes/train_moe.py --experts 4 --top-k 2

# Speculative decoding (draft-then-verify generation)
uv run python recipes/speculative_decoding.py --draft-tokens 4

# Profile all architectures (memory, throughput, generation speed)
uv run python recipes/profile_models.py

# Benchmark mx.compile speedup on training steps
uv run python recipes/benchmark_compile.py

# Evaluate models with perplexity and BPB metrics
uv run python recipes/evaluate_model.py
```

## Next steps

- **[Architecture Overview](../architecture/overview.md)** -- Understand the
  config/registry/block design in depth.
- **[MLX Idioms](../architecture/mlx-idioms.md)** -- Learn the MLX patterns
  that differ from PyTorch.
- **[Models](../models/index.md)** -- Compare all 8 architectures side-by-side.
