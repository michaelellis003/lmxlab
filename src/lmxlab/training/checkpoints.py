"""Checkpoint save/load using safetensors."""

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import mlx.utils

from lmxlab.models.base import LanguageModel


def save_checkpoint(
    path: str | Path,
    model: LanguageModel,
    optimizer: optim.Optimizer | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint.

    Saves model weights as safetensors and metadata as JSON.

    Args:
        path: Directory to save checkpoint.
        model: Model to save.
        optimizer: Optional optimizer to save state.
        step: Current training step.
        metadata: Additional metadata to save.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)

    # Save optimizer state if provided
    if optimizer is not None:
        opt_state = dict(mlx.utils.tree_flatten(optimizer.state))
        if opt_state:
            mx.save_safetensors(str(path / "optimizer.safetensors"), opt_state)

    # Save metadata
    meta = {
        "step": step,
        **(metadata or {}),
    }
    (path / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_checkpoint(
    path: str | Path,
    model: LanguageModel,
    optimizer: optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Directory containing checkpoint.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.

    Returns:
        Metadata dict from the checkpoint.
    """
    path = Path(path)

    # Load model weights
    weights = mx.load(str(path / "model.safetensors"))
    model.load_weights(list(weights.items()))

    # Load optimizer state if available
    opt_path = path / "optimizer.safetensors"
    if optimizer is not None and opt_path.exists():
        opt_state = mx.load(str(opt_path))
        # Reconstruct nested state from flat keys
        optimizer.state = mlx.utils.tree_unflatten(list(opt_state.items()))

    # Load metadata
    meta_path = path / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}
