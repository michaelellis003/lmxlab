# Training

lmt-metal provides a compiled training loop built on MLX idioms: `nn.value_and_grad` for functional gradients, `mx.compile` for the full training step, and explicit `mx.eval()` boundaries.

## Quick Start

```python
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer

model = LanguageModel(gpt_tiny())
config = TrainConfig(learning_rate=1e-3, max_steps=100)
trainer = Trainer(model, config)

# Train on batches of (input_tokens, target_tokens)
history = trainer.train(data_iterator)
```

## TrainConfig

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | 3e-4 | Peak learning rate |
| `weight_decay` | 0.01 | Weight decay coefficient |
| `warmup_steps` | 100 | Linear warmup steps |
| `max_steps` | 1000 | Maximum training steps |
| `batch_size` | 32 | Training batch size |
| `grad_accumulation_steps` | 1 | Micro-batches per optimizer step |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `optimizer` | "adamw" | Optimizer name |
| `lr_schedule` | "cosine" | Learning rate schedule |
| `compile_step` | True | Whether to `mx.compile` the step |

## Gradient Accumulation

Simulate larger batch sizes by accumulating gradients over multiple
micro-batches before each optimizer update:

```python
config = TrainConfig(
    batch_size=8,                  # Each micro-batch has 8 samples
    grad_accumulation_steps=4,     # Accumulate 4 micro-batches
    # Effective batch size: 8 * 4 = 32
)
```

Gradients are averaged across micro-batches, then clipped and applied
in a single optimizer step. This is useful when the effective batch
size you want doesn't fit in memory.

## Compiled Training Step

The training step is compiled by default for maximum performance:

```python
# Inside Trainer.__init__:
self._loss_and_grad = nn.value_and_grad(model, loss_fn)

if config.compile_step:
    self._step_fn = mx.compile(
        self._single_step,
        inputs=model.trainable_parameters(),
        outputs=model.trainable_parameters(),
    )
```

**Key MLX pattern:** `mx.compile` needs explicit `inputs` and `outputs` so it knows which arrays to trace. The model parameters serve as both.

## Optimizers

Three optimizers are available:

- **AdamW** (default): Standard adaptive optimizer with weight decay
- **Lion**: Sign-based optimizer, lower memory than Adam
- **Adafactor**: Memory-efficient adaptive optimizer

All use MLX's built-in learning rate schedules (`cosine_decay`, `linear_decay`).

## Advanced Training

### DPO (Direct Preference Optimization)

Train on preference pairs without reward modeling:

```python
from lmt_metal.training.dpo import dpo_loss

loss = dpo_loss(model, ref_model, chosen, rejected, beta=0.1)
```

### GRPO (Group Relative Policy Optimization)

Policy gradient with group-relative rewards:

```python
from lmt_metal.training.grpo import grpo_loss

loss = grpo_loss(model, ref_model, prompts, completions, rewards)
```

### Multi-Token Prediction

Train the model to predict multiple future tokens simultaneously:

```python
from lmt_metal.training.mtp import MultiTokenPrediction

mtp = MultiTokenPrediction(model, n_predict=2, mtp_weight=0.3)
logits, losses = mtp(input_ids, target_ids)
# losses["total_loss"] = main_loss + 0.3 * avg(mtp_losses)
```

### Curriculum Learning

Gradually increase training difficulty:

```python
from lmt_metal.training.curriculum import length_curriculum

batches = length_curriculum(
    tokens, batch_size=32,
    min_seq_len=64, max_seq_len=512,
    n_stages=4, batches_per_stage=100,
)
```

## LoRA Fine-Tuning

LoRA (Low-Rank Adaptation) freezes the base model and trains small
low-rank matrices on top. This reduces trainable parameters by 10-100x
while preserving most fine-tuning quality.

```python
from lmt_metal.core.lora import apply_lora, merge_lora
from lmt_metal.core.lora import save_lora_adapters, load_lora_adapters

# 1. Apply LoRA to attention layers
apply_lora(model, rank=8, alpha=16.0, targets=['attention'])

# 2. Train (only LoRA params are trainable)
trainer = Trainer(model, train_config)
trainer.train(data)

# 3. Save just the adapter (~MBs vs GBs for full model)
save_lora_adapters('adapters/my-lora', model, rank=8, alpha=16.0)

# 4. Later: load adapter onto a fresh base model
load_lora_adapters('adapters/my-lora', model)

# 5. Merge LoRA into base weights for inference (no overhead)
merge_lora(model)
```

Target options:

- `'attention'` — q/k/v/o projections (default, most common)
- `'ffn'` — gate/up/down projections
- Both: `targets=['attention', 'ffn']`

### QLoRA

QLoRA combines 4-bit quantized base weights with float16 LoRA adapters
for maximum memory efficiency:

```python
from lmt_metal.core.quantize import quantize_model
from lmt_metal.core.qlora import apply_qlora

# Quantize base model to 4 bits
quantize_model(model, bits=4)

# Add LoRA on top of quantized layers
apply_qlora(model, rank=8, targets=['attention'])

# Train normally — base stays int4, LoRA trains in float16
trainer.train(data)
```

## Checkpoints

Save and load via safetensors:

```python
from lmt_metal.training.checkpoints import save_checkpoint, load_checkpoint

save_checkpoint("checkpoints/step_100", model, optimizer, step=100)
metadata = load_checkpoint("checkpoints/step_100", model, optimizer)
```

### LoRA Adapter Checkpoints

For LoRA models, save only the adapter weights (much smaller):

```python
from lmt_metal.core.lora import save_lora_adapters, load_lora_adapters

# Save only LoRA weights (~100KB-10MB)
save_lora_adapters("adapters/my-lora", model, rank=8, alpha=16.0)

# Load onto a new model (must have apply_lora called first)
load_lora_adapters("adapters/my-lora", new_model)
```
