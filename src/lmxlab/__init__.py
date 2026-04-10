"""lmxlab: Transformer language models on Apple Silicon with MLX."""

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate, stream_generate
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

__version__ = "0.3.1"

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
