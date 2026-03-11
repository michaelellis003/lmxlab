# Training

Compiled training loop, optimizers, and training utilities.

## Trainer

::: lmt_metal.training.trainer.Trainer
    options:
      members: true

## Training Config

::: lmt_metal.training.config.TrainConfig
    options:
      members: true

## Optimizers

::: lmt_metal.training.optimizers.create_optimizer

::: lmt_metal.training.optimizers.create_schedule

## Checkpoints

::: lmt_metal.training.checkpoints.save_checkpoint

::: lmt_metal.training.checkpoints.load_checkpoint

## Callbacks

::: lmt_metal.training.callbacks.Callback

::: lmt_metal.training.callbacks.MetricsLogger

::: lmt_metal.training.callbacks.EarlyStopping

## DPO

::: lmt_metal.training.dpo

## GRPO

::: lmt_metal.training.grpo

## Multi-Token Prediction

::: lmt_metal.training.mtp.MTPHead
    options:
      members: ["__init__", "__call__"]

::: lmt_metal.training.mtp.MultiTokenPrediction
    options:
      members: ["__init__", "__call__"]

## Curriculum Learning

::: lmt_metal.training.curriculum
