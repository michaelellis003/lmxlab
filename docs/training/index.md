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
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `optimizer` | "adamw" | Optimizer name |
| `lr_schedule` | "cosine" | Learning rate schedule |
| `compile_step` | True | Whether to `mx.compile` the step |

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

## Checkpoints

Save and load via safetensors:

```python
from lmt_metal.training.checkpoints import save_checkpoint, load_checkpoint

save_checkpoint("checkpoints/step_100", model, optimizer, step=100)
metadata = load_checkpoint("checkpoints/step_100", model, optimizer)
```
