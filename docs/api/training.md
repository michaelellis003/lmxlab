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

## GRPO Trainer

::: lmxlab.training.grpo_trainer.GRPOConfig
    options:
      members: true

::: lmxlab.training.grpo_trainer.GRPOTrainer
    options:
      members: true

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

## Hardware Detection

::: lmxlab.training.hardware.detect_peak_tflops

## Metric Callbacks

::: lmxlab.training.metric_callbacks.GradientStatsCallback

::: lmxlab.training.metric_callbacks.WeightStatsCallback

::: lmxlab.training.metric_callbacks.ActivationStatsCallback

::: lmxlab.training.metric_callbacks.AttentionEntropyCallback

::: lmxlab.training.metric_callbacks.LossCurvatureCallback

::: lmxlab.training.metric_callbacks.EffectiveRankCallback
