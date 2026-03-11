"""lmt-metal: Educational MLX library for transformer LMs on Apple Silicon."""

from lmt_metal.core.config import BlockConfig, ModelConfig
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.generate import generate, stream_generate
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer

__version__ = "0.1.0"

__all__ = [
    "BlockConfig",
    "LanguageModel",
    "ModelConfig",
    "TrainConfig",
    "Trainer",
    "__version__",
    "generate",
    "stream_generate",
]
