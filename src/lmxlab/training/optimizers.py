"""Optimizer and learning rate schedule factories."""

from collections.abc import Callable

import mlx.optimizers as optim

from lmxlab.training.config import TrainConfig


def create_schedule(
    config: TrainConfig,
) -> Callable[[int], float]:
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
    elif config.optimizer == "sgd":
        return optim.SGD(
            learning_rate=schedule,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer!r}")


def create_mup_optimizer(
    config: TrainConfig,
    width_mult: float,
) -> optim.Optimizer:
    """Create optimizer with μP learning rate scaling.

    Uses MLX's MultiOptimizer to assign different learning
    rates to different parameter groups:
    - Embedding parameters: base LR (no scaling)
    - Hidden/head parameters: base LR / width_mult

    When width_mult == 1.0, all groups use the same LR.

    Args:
        config: Training configuration.
        width_mult: d_model / base_d_model ratio.

    Returns:
        MultiOptimizer with per-layer LR groups.
    """
    embed_schedule = create_schedule(config)
    scaled_lr = config.learning_rate / width_mult
    scaled_config = TrainConfig(
        learning_rate=scaled_lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        lr_schedule=config.lr_schedule,
    )
    hidden_schedule = create_schedule(scaled_config)

    embed_opt = _create_single_optimizer(
        config.optimizer,
        embed_schedule,
        config.weight_decay,
    )
    hidden_opt = _create_single_optimizer(
        config.optimizer,
        hidden_schedule,
        config.weight_decay,
    )

    return optim.MultiOptimizer(
        optimizers=[embed_opt, hidden_opt],
        filters=[lambda k, g: 'embed' in k],
    )


def _create_single_optimizer(
    name: str,
    schedule: object,
    weight_decay: float,
) -> optim.Optimizer:
    """Create a single optimizer instance.

    Args:
        name: Optimizer name.
        schedule: Learning rate schedule.
        weight_decay: Weight decay coefficient.

    Returns:
        Configured optimizer.
    """
    if name == 'adamw':
        return optim.AdamW(
            learning_rate=schedule,
            weight_decay=weight_decay,
        )
    elif name == 'lion':
        return optim.Lion(
            learning_rate=schedule,
            weight_decay=weight_decay,
        )
    elif name == 'adafactor':
        return optim.Adafactor(learning_rate=schedule)
    elif name == 'sgd':
        return optim.SGD(
            learning_rate=schedule,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f'Unknown optimizer: {name!r}')
