"""lmxlab: Research platform for LM experimentation on Apple Silicon."""

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate, stream_generate
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

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
