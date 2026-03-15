"""Experiment-specific metric callbacks.

Pre-built callbacks that inject ``exp_*`` prefixed metrics into
the training metrics dict. MLflow routes these to the
``4_experiment/`` group automatically.

Each callback implements the Callback protocol from
``training.callbacks``.

Usage:
    >>> from lmxlab.training.metric_callbacks import (
    ...     GradientStatsCallback,
    ...     WeightStatsCallback,
    ... )
    >>> cbs = standard_callbacks(...) + [
    ...     GradientStatsCallback(model, loss_fn, log_interval=100),
    ...     WeightStatsCallback(model, log_interval=100),
    ... ]
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from lmxlab.training.config import TrainConfig


class GradientStatsCallback:
    """Tracks per-layer gradient norm statistics.

    Computes gradient norms via a separate forward+backward pass
    on a stored probe batch at measurement intervals.

    Args:
        model: Model to measure gradients on.
        loss_fn: Loss function ``(model, x, y) -> scalar``.
        log_interval: Steps between measurements.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Any,
        log_interval: int = 100,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.log_interval = log_interval
        self._probe_batch: tuple[mx.array, mx.array] | None = None

    def on_train_begin(self, config: TrainConfig) -> None:
        """No action on train begin."""

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Compute gradient stats at log_interval."""
        if self._probe_batch is None:
            return
        if step % self.log_interval != 0:
            return

        was_training = self.model.training
        self.model.eval()
        try:
            loss_and_grad = nn.value_and_grad(self.model, self.loss_fn)
            x, y = self._probe_batch
            _, grads = loss_and_grad(self.model, x, y)
        finally:
            if was_training:
                self.model.train()

        norms = []
        for _, g in tree_flatten(grads):
            norm = mx.sqrt(mx.sum(g * g))
            mx.eval(norm)
            norms.append(norm.item())

        if norms:
            mean_norm = sum(norms) / len(norms)
            std_norm = (
                sum((n - mean_norm) ** 2 for n in norms) / len(norms)
            ) ** 0.5
            max_idx = max(range(len(norms)), key=norms.__getitem__)
            metrics["exp_grad_norm_mean"] = mean_norm
            metrics["exp_grad_norm_std"] = std_norm
            metrics["exp_grad_norm_max_layer"] = float(max_idx)

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """No action on eval."""

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """No action on train end."""

    def set_probe_batch(self, batch: tuple[mx.array, mx.array]) -> None:
        """Store a probe batch for gradient measurement.

        Args:
            batch: Tuple of (input, target) arrays.
        """
        self._probe_batch = batch


class WeightStatsCallback:
    """Tracks weight norm and weight delta statistics.

    Stores initial norms in ``on_train_begin`` and computes
    delta (change from initial) at measurement intervals.

    Args:
        model: Model to measure weights on.
        log_interval: Steps between measurements.
    """

    def __init__(
        self,
        model: nn.Module,
        log_interval: int = 100,
    ) -> None:
        self.model = model
        self.log_interval = log_interval
        self._initial_norm: float = 0.0

    def on_train_begin(self, config: TrainConfig) -> None:
        """Store initial weight norm."""
        self._initial_norm = self._compute_weight_norm()

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Compute weight stats at log_interval."""
        if step % self.log_interval != 0:
            return
        current = self._compute_weight_norm()
        metrics["exp_weight_norm"] = current
        metrics["exp_weight_delta"] = abs(current - self._initial_norm)

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """No action on eval."""

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """No action on train end."""

    def _compute_weight_norm(self) -> float:
        """Compute total L2 norm of trainable parameters."""
        total = mx.array(0.0)
        for _, p in tree_flatten(self.model.trainable_parameters()):
            total = total + mx.sum(p * p)
        mx.eval(total)
        return mx.sqrt(total).item()


class ActivationStatsCallback:
    """Tracks activation norm ratios and sparsity.

    Uses ``ActivationCapture`` from the analysis module to
    capture layer activations on a probe batch.

    Args:
        model: Language model to instrument.
        probe_batch: Input tokens for activation capture.
        eval_interval: Steps between measurements.
        eps: Threshold for sparsity (fraction |x| < eps).
    """

    def __init__(
        self,
        model: Any,
        probe_batch: mx.array,
        eval_interval: int = 500,
        eps: float = 1e-3,
    ) -> None:
        self.model = model
        self.probe_batch = probe_batch
        self.eval_interval = eval_interval
        self.eps = eps

    def on_train_begin(self, config: TrainConfig) -> None:
        """No action on train begin."""

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Compute activation stats at eval_interval."""
        if step % self.eval_interval != 0:
            return

        from lmxlab.analysis.activations import (
            ActivationCapture,
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with ActivationCapture(self.model) as cap:
                self.model(self.probe_batch)
        finally:
            if was_training:
                self.model.train()

        # Compute per-layer output norms
        output_norms = []
        sparsities = []
        for key, val in sorted(cap.activations.items()):
            if not key.endswith("/output"):
                continue
            mx.eval(val)
            norm = mx.sqrt(mx.sum(val * val)).item()
            output_norms.append(norm)
            # Sparsity: fraction of elements near zero
            sparse_frac = mx.mean((mx.abs(val) < self.eps).astype(mx.float32))
            mx.eval(sparse_frac)
            sparsities.append(sparse_frac.item())

        if len(output_norms) >= 2:
            metrics["exp_act_norm_ratio"] = output_norms[-1] / max(
                output_norms[0], 1e-10
            )
        if sparsities:
            metrics["exp_act_sparsity_mean"] = sum(sparsities) / len(
                sparsities
            )

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """No action on eval."""

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """No action on train end."""


class AttentionEntropyCallback:
    """Tracks Shannon entropy of attention weights.

    Uses ``extract_attention_maps`` from the analysis module
    to get per-head attention weights.

    Args:
        model: Language model with attention layers.
        probe_batch: Input tokens for attention extraction.
        eval_interval: Steps between measurements.
    """

    def __init__(
        self,
        model: Any,
        probe_batch: mx.array,
        eval_interval: int = 500,
    ) -> None:
        self.model = model
        self.probe_batch = probe_batch
        self.eval_interval = eval_interval

    def on_train_begin(self, config: TrainConfig) -> None:
        """No action on train begin."""

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Compute attention entropy at eval_interval."""
        if step % self.eval_interval != 0:
            return

        from lmxlab.analysis.attention import (
            extract_attention_maps,
        )

        was_training = self.model.training
        self.model.eval()
        try:
            maps = extract_attention_maps(self.model, self.probe_batch)
        finally:
            if was_training:
                self.model.train()

        entropies = []
        for weights in maps.values():
            # weights: (batch, heads, seq, seq)
            # Shannon entropy per head, averaged over batch/seq
            # Clamp to avoid log(0)
            w = mx.clip(weights, 1e-10, 1.0)
            h = -mx.sum(w * mx.log(w), axis=-1)  # per position
            mean_h = mx.mean(h)
            mx.eval(mean_h)
            entropies.append(mean_h.item())

        if entropies:
            mean_ent = sum(entropies) / len(entropies)
            std_ent = (
                sum((e - mean_ent) ** 2 for e in entropies) / len(entropies)
            ) ** 0.5
            metrics["exp_attn_entropy_mean"] = mean_ent
            metrics["exp_attn_entropy_std"] = std_ent

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """No action on eval."""

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """No action on train end."""


class LossCurvatureCallback:
    """Tracks gradient noise scale from grad_norm history.

    Maintains a running window of ``grad_norm`` values from
    the metrics dict and computes gradient noise scale =
    std(grad_norms) / mean(grad_norms).

    Args:
        window_size: Number of recent grad_norms to track.
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self._window: deque[float] = deque(maxlen=window_size)

    def on_train_begin(self, config: TrainConfig) -> None:
        """Reset window."""
        self._window.clear()

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Update window and compute noise scale."""
        grad_norm = metrics.get("grad_norm")
        if grad_norm is None:
            return
        self._window.append(float(grad_norm))
        if len(self._window) >= 2:
            vals = list(self._window)
            mean = sum(vals) / len(vals)
            if mean > 1e-10:
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                metrics["exp_grad_noise_scale"] = std / mean

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """No action on eval."""

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """No action on train end."""


class EffectiveRankCallback:
    """Tracks effective rank of weight matrices.

    Computes SVD of the largest weight matrix per layer, then
    effective rank = exp(entropy of normalized singular values).

    Args:
        model: Model to analyze.
        eval_interval: Steps between measurements.
    """

    def __init__(
        self,
        model: nn.Module,
        eval_interval: int = 500,
    ) -> None:
        self.model = model
        self.eval_interval = eval_interval

    def on_train_begin(self, config: TrainConfig) -> None:
        """No action on train begin."""

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Compute effective rank at eval_interval."""
        if step % self.eval_interval != 0:
            return

        ranks = []
        for _, p in tree_flatten(self.model.trainable_parameters()):
            if p.ndim != 2:
                continue
            # Only process matrices above a size threshold
            if min(p.shape) < 4:
                continue
            sv = mx.linalg.svd(p, compute_uv=False, stream=mx.cpu)
            mx.eval(sv)
            # Normalize singular values
            sv_sum = mx.sum(sv)
            if sv_sum.item() < 1e-10:
                continue
            p_sv = sv / sv_sum
            # Entropy of normalized singular values
            p_sv = mx.clip(p_sv, 1e-10, 1.0)
            entropy = -mx.sum(p_sv * mx.log(p_sv))
            mx.eval(entropy)
            eff_rank = math.exp(entropy.item())
            ranks.append(eff_rank)

        if ranks:
            metrics["exp_effective_rank_mean"] = sum(ranks) / len(ranks)

    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None:
        """No action on eval."""

    def on_train_end(self, history: list[dict[str, Any]]) -> None:
        """No action on train end."""
