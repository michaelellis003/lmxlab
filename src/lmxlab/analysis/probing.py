"""Linear probing classifiers for representation analysis.

Train lightweight probes on frozen model activations to
measure what information each layer encodes.

Example:
    >>> probe = LinearProbe(d_model=64, num_classes=256)
    >>> # Train on activations from a specific layer
    >>> train_probe(model, data_iter, layer=2, probe=probe)
    >>> acc = probe_accuracy(model, eval_iter, layer=2, probe=probe)
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from lmxlab.analysis.activations import ActivationCapture
from lmxlab.models.base import LanguageModel


class LinearProbe(nn.Module):
    """Single linear layer for probing experiments.

    Args:
        input_dim: Dimension of input activations.
        num_classes: Number of output classes.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (..., input_dim).

        Returns:
            Logits of shape (..., num_classes).
        """
        return self.linear(x)


def train_probe(
    model: LanguageModel,
    data: Iterator[tuple[mx.array, mx.array]],
    layer: int,
    probe: LinearProbe,
    steps: int = 200,
    lr: float = 1e-3,
) -> list[float]:
    """Train a probe on frozen model activations.

    Captures activations at the specified layer, then trains
    the probe to predict target tokens from those activations.

    Args:
        model: Frozen language model (not modified).
        data: Iterator of (input_tokens, target_tokens).
        layer: Layer index to probe (uses output activations).
        probe: LinearProbe to train.
        steps: Training steps.
        lr: Learning rate.

    Returns:
        List of per-step loss values.
    """
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(probe, _probe_loss)

    losses: list[float] = []

    for step, (x, y) in enumerate(data):
        if step >= steps:
            break

        # Get activations from frozen model
        with ActivationCapture(model) as cap:
            model(x)
        key = f"layer_{layer}/output"
        acts = cap.activations[key]

        # Train probe
        loss, grads = loss_and_grad(probe, acts, y)
        optimizer.update(probe, grads)
        mx.eval(loss, probe.parameters(), optimizer.state)

        losses.append(loss.item())

    return losses


def _probe_loss(
    probe: LinearProbe,
    activations: mx.array,
    targets: mx.array,
) -> mx.array:
    """Cross-entropy loss for probe training.

    Args:
        probe: Linear probe module.
        activations: Hidden states (batch, seq_len, d_model).
        targets: Target token IDs (batch, seq_len).

    Returns:
        Scalar loss.
    """
    logits = probe(activations)
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def probe_accuracy(
    model: LanguageModel,
    data: Iterator[tuple[mx.array, mx.array]],
    layer: int,
    probe: LinearProbe,
    max_batches: int = 50,
) -> float:
    """Evaluate probe accuracy on data.

    Args:
        model: Frozen language model.
        data: Iterator of (input_tokens, target_tokens).
        layer: Layer index to probe.
        probe: Trained LinearProbe.
        max_batches: Maximum batches to evaluate.

    Returns:
        Accuracy as a float in [0, 1].
    """
    correct = 0
    total = 0

    for i, (x, y) in enumerate(data):
        if i >= max_batches:
            break

        with ActivationCapture(model) as cap:
            model(x)
        key = f"layer_{layer}/output"
        acts = cap.activations[key]

        logits = probe(acts)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)

        correct += (preds == y).sum().item()
        total += y.size

    return correct / max(total, 1)
