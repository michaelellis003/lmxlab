"""lmt-metal: Educational MLX library for transformer LMs on Apple Silicon."""

from lmt_metal.core.config import BlockConfig, ModelConfig
from lmt_metal.models.base import LanguageModel

__version__ = "0.1.0"

__all__ = [
    "BlockConfig",
    "LanguageModel",
    "ModelConfig",
    "__version__",
]
