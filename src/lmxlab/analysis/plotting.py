"""Visualization utilities for model analysis.

All functions return ``matplotlib.figure.Figure`` objects
that can be saved or displayed in notebooks.

Requires matplotlib: ``uv sync --extra analysis``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

import mlx.core as mx


def _import_plt():  # noqa: ANN202
    """Import matplotlib.pyplot, raising helpful error."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: uv sync --extra analysis"
        ) from e


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Training Loss",
) -> matplotlib.figure.Figure:
    """Plot training and optional validation loss curves.

    Args:
        train_losses: Per-step training losses.
        val_losses: Optional validation losses (same length
            or shorter).
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(train_losses, label="train", alpha=0.7)
    if val_losses is not None:
        # Val losses may be evaluated less frequently
        if len(val_losses) < len(train_losses):
            step = len(train_losses) // len(val_losses)
            x = list(range(0, len(train_losses), step))[: len(val_losses)]
        else:
            x = list(range(len(val_losses)))
        ax.plot(x, val_losses, label="val", alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.close(fig)
    return fig


def plot_layer_norms(
    activations: dict[str, mx.array],
    title: str = "Activation L2 Norms by Layer",
) -> matplotlib.figure.Figure:
    """Plot L2 norm of activations per layer.

    Args:
        activations: Dict from ActivationCapture, keyed by
            ``"layer_N/input"`` or ``"layer_N/output"``.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    plt = _import_plt()

    # Extract output norms, sorted by layer index
    items = []
    for key, val in activations.items():
        if key.endswith("/output"):
            layer = int(key.split("_")[1].split("/")[0])
            mx.eval(val)
            norm = mx.sqrt(mx.sum(val * val)).item()
            items.append((layer, norm))

    items.sort(key=lambda x: x[0])
    if not items:
        fig, ax = plt.subplots()
        ax.set_title("No output activations found")
        plt.close(fig)
        return fig

    layers, norms = zip(*items, strict=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(layers, norms, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.close(fig)
    return fig


def plot_attention_heatmap(
    weights: mx.array,
    tokens: list[str] | None = None,
    head: int = 0,
    layer: int = 0,
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot attention weight heatmap for one head.

    Args:
        weights: Attention weights of shape
            (batch, heads, seq, seq). Uses first batch item.
        tokens: Optional token strings for axis labels.
        head: Head index to visualize.
        layer: Layer index (for title only).
        title: Custom title. Defaults to
            ``"Layer {layer}, Head {head}"``.

    Returns:
        Matplotlib Figure.
    """
    plt = _import_plt()
    import numpy as np

    mx.eval(weights)
    w = np.array(weights[0, head])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)

    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title or f"Layer {layer}, Head {head}")
    plt.close(fig)
    return fig


def plot_gradient_flow(
    model: object,
    title: str = "Gradient Flow",
) -> matplotlib.figure.Figure:
    """Plot gradient magnitude per parameter group.

    Call after a backward pass while gradients are available.
    Groups parameters by top-level module name.

    Args:
        model: Model with ``parameters()`` method. Parameters
            should have gradient information from a recent
            backward pass.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    plt = _import_plt()
    import mlx.utils

    leaves = mlx.utils.tree_flatten(model.parameters())  # type: ignore[attr-defined]
    groups: dict[str, list[float]] = {}
    for path, p in leaves:
        if p.ndim < 2:
            continue
        top = path.split(".")[0]
        mx.eval(p)
        norm = mx.sqrt(mx.sum(p * p)).item()
        groups.setdefault(top, []).append(norm)

    # Average norm per group
    names = sorted(groups.keys())
    avg_norms = [sum(groups[n]) / len(groups[n]) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), avg_norms, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Avg Parameter Norm")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.close(fig)
    return fig
