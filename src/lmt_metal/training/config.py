"""Training configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for the training loop.

    Args:
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        warmup_steps: Linear warmup steps.
        max_steps: Maximum training steps.
        batch_size: Training batch size.
        grad_accumulation_steps: Number of micro-batches
            to accumulate before an optimizer step.
        max_grad_norm: Maximum gradient norm for clipping.
        eval_interval: Steps between evaluations.
        log_interval: Steps between logging.
        checkpoint_interval: Steps between checkpoints.
        optimizer: Optimizer name ('adamw', 'lion', 'adafactor').
        lr_schedule: Learning rate schedule ('cosine', 'linear', 'constant').
        compile_step: Whether to mx.compile the training step.
        seed: Random seed.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    batch_size: int = 32
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 500
    optimizer: str = "adamw"
    lr_schedule: str = "cosine"
    compile_step: bool = True
    seed: int = 42
