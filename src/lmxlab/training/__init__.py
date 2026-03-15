"""Training infrastructure: trainer, optimizers, checkpoints, callbacks."""

from lmxlab.training.callbacks import (
    Callback,
    EarlyStopping,
    FLOPCounter,
    HardwareMonitor,
    MetricsLogger,
    ThroughputMonitor,
    ValTracker,
    standard_callbacks,
)
from lmxlab.training.checkpoints import load_checkpoint, save_checkpoint
from lmxlab.training.config import TrainConfig
from lmxlab.training.curriculum import (
    difficulty_curriculum,
    length_curriculum,
)
from lmxlab.training.distillation import (
    distillation_loss,
    soft_target_loss,
)
from lmxlab.training.dpo import dpo_loss
from lmxlab.training.grpo import grpo_loss
from lmxlab.training.grpo_trainer import GRPOConfig, GRPOTrainer
from lmxlab.training.hardware import detect_peak_tflops
from lmxlab.training.metric_callbacks import (
    ActivationStatsCallback,
    AttentionEntropyCallback,
    EffectiveRankCallback,
    GradientStatsCallback,
    LossCurvatureCallback,
    WeightStatsCallback,
)
from lmxlab.training.mtp import MTPHead, MultiTokenPrediction
from lmxlab.training.optimizers import create_optimizer, create_schedule
from lmxlab.training.trainer import Trainer

__all__ = [
    "ActivationStatsCallback",
    "AttentionEntropyCallback",
    "Callback",
    "EarlyStopping",
    "EffectiveRankCallback",
    "FLOPCounter",
    "GradientStatsCallback",
    "HardwareMonitor",
    "LossCurvatureCallback",
    "MetricsLogger",
    "ThroughputMonitor",
    "ValTracker",
    "WeightStatsCallback",
    "detect_peak_tflops",
    "standard_callbacks",
    "Trainer",
    "TrainConfig",
    "create_optimizer",
    "create_schedule",
    "difficulty_curriculum",
    "distillation_loss",
    "dpo_loss",
    "GRPOConfig",
    "GRPOTrainer",
    "grpo_loss",
    "MTPHead",
    "MultiTokenPrediction",
    "length_curriculum",
    "load_checkpoint",
    "save_checkpoint",
    "soft_target_loss",
]

try:
    from lmxlab.experiments.mlflow import MLflowCallback

    __all__ += ["MLflowCallback"]
except ImportError:
    pass
