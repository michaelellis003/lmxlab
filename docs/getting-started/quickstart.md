# Quickstart

## Create a model

Every model starts with a config. Use a preset factory or build one manually:

```python
from lmxlab.models.llama import llama_config
from lmxlab.models.base import LanguageModel

config = llama_config(
    vocab_size=32000, d_model=512, n_heads=8,
    n_kv_heads=4, n_layers=6, d_ff=1376,
)
model = LanguageModel(config)
```

Or build a config from scratch:

```python
from lmxlab.core.config import BlockConfig, ModelConfig

block = BlockConfig(
    attention='mha', ffn='standard', norm='rms_norm',
    position='rope', d_model=256, n_heads=4,
)
config = ModelConfig(block=block, vocab_size=32000, n_layers=4)
model = LanguageModel(config)
```

## Forward pass

```python
import mlx.core as mx

tokens = mx.random.randint(0, 32000, shape=(2, 16))
logits, caches = model(tokens)
# logits: (2, 16, 32000)
```

No `.to(device)` needed — MLX uses unified memory.

## Generation

```python
from lmxlab import generate, stream_generate

prompt = mx.array([[1, 234, 567]])

# Greedy
output = generate(model, prompt, max_tokens=20, temperature=0.0)

# Sampling
output = generate(model, prompt, max_tokens=50, temperature=0.8, top_k=40)

# Streaming
for token_id in stream_generate(model, prompt, max_tokens=50, temperature=0.8):
    print(token_id, end=' ', flush=True)
```

## Training

```python
from lmxlab import Trainer, TrainConfig

trainer = Trainer(model, TrainConfig(learning_rate=3e-4, max_steps=1000))
# trainer.train(data_iterator)
```

## Switch architectures

Same model class, different config:

```python
from lmxlab.models.deepseek import deepseek_config

ds = LanguageModel(deepseek_config(
    d_model=512, n_heads=8, n_layers=6, kv_lora_rank=64,
))
```

## CLI

```bash
lmxlab list                    # All architectures
lmxlab info llama --tiny       # Config details
lmxlab count deepseek --detail # Parameter breakdown
```
