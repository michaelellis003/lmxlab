# Training

Compiled training loop, optimizers, and training utilities.

## Trainer

::: lmxlab.training.trainer.Trainer
    options:
      members: true

## Training Config

::: lmxlab.training.config.TrainConfig
    options:
      members: true

## Optimizers

::: lmxlab.training.optimizers.create_optimizer

::: lmxlab.training.optimizers.create_schedule

## Checkpoints

::: lmxlab.training.checkpoints.save_checkpoint

::: lmxlab.training.checkpoints.load_checkpoint

## Callbacks

::: lmxlab.training.callbacks.Callback

::: lmxlab.training.callbacks.MetricsLogger

::: lmxlab.training.callbacks.EarlyStopping

::: lmxlab.training.callbacks.ThroughputMonitor

## DPO

::: lmxlab.training.dpo

## GRPO

::: lmxlab.training.grpo

## Multi-Token Prediction

::: lmxlab.training.mtp.MTPHead
    options:
      members: ["__init__", "__call__"]

::: lmxlab.training.mtp.MultiTokenPrediction
    options:
      members: ["__init__", "__call__"]

## Curriculum Learning

::: lmxlab.training.curriculum

## Knowledge Distillation

::: lmxlab.training.distillation
