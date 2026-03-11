"""Optimizer and learning rate schedule factories."""

import mlx.optimizers as optim

from lmt_metal.training.config import TrainConfig


def create_schedule(
    config: TrainConfig,
) -> optim.schedulers.SchedulerBase:
    """Create a learning rate schedule.

    Supports warmup + decay patterns.

    Args:
        config: Training configuration.

    Returns:
        Learning rate scheduler.
    """
    warmup = optim.schedulers.linear_schedule(
        init=1e-7,
        end=config.learning_rate,
        steps=config.warmup_steps,
    )
    if config.lr_schedule == "cosine":
        decay = optim.schedulers.cosine_decay(
            init=config.learning_rate,
            decay_steps=config.max_steps - config.warmup_steps,
        )
    elif config.lr_schedule == "linear":
        decay = optim.schedulers.linear_schedule(
            init=config.learning_rate,
            end=0.0,
            steps=config.max_steps - config.warmup_steps,
        )
    elif config.lr_schedule == "constant":
        decay = config.learning_rate
    else:
        raise ValueError(f"Unknown schedule: {config.lr_schedule!r}")

    return optim.schedulers.join_schedules(
        schedules=[warmup, decay],
        boundaries=[config.warmup_steps],
    )


def create_optimizer(
    config: TrainConfig,
) -> optim.Optimizer:
    """Create an optimizer with learning rate schedule.

    Args:
        config: Training configuration.

    Returns:
        Configured optimizer.
    """
    schedule = create_schedule(config)

    if config.optimizer == "adamw":
        return optim.AdamW(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "lion":
        return optim.Lion(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adafactor":
        return optim.Adafactor(
            learning_rate=schedule,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer!r}")
