"""Per-layer activation capture for model analysis.

Captures intermediate hidden states during a forward pass
by replacing blocks in the model's block list with thin
wrappers. Uses a context manager pattern so originals are
always restored on exit.

Example:
    >>> model = LanguageModel(config)
    >>> tokens = mx.array([[1, 2, 3]])
    >>> with ActivationCapture(model) as cap:
    ...     logits, _ = model(tokens)
    >>> cap.activations["layer_0/output"].shape
    (1, 3, 64)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from lmxlab.models.base import LanguageModel


class _CaptureWrapper(nn.Module):
    """Thin wrapper that delegates to a block and records I/O.

    Args:
        block: Original ConfigurableBlock to wrap.
        layer_idx: Layer index for activation key naming.
        store: Dict to write captured activations into.
    """

    def __init__(
        self,
        block: nn.Module,
        layer_idx: int,
        store: dict[str, mx.array],
    ) -> None:
        super().__init__()
        self.inner = block
        self._idx = layer_idx
        self._store = store
        # Expose config for compatibility with model forward
        self.config = block.config
        self.position = block.position

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
    ) -> tuple[mx.array, Any]:
        """Forward with activation recording."""
        self._store[f"layer_{self._idx}/input"] = x
        out, new_cache = self.inner(x, mask=mask, cache=cache)
        self._store[f"layer_{self._idx}/output"] = out
        return out, new_cache


class ActivationCapture:
    """Captures per-layer activations during forward pass.

    Replaces blocks with thin wrappers that record input
    and output tensors. Restores originals on context exit.

    Captured keys per layer N:
    - ``layer_N/input``: block input (pre-norm hidden state)
    - ``layer_N/output``: block output (after FFN sublayer)

    Args:
        model: Language model to instrument.
    """

    def __init__(self, model: LanguageModel) -> None:
        self.model = model
        self.activations: dict[str, mx.array] = {}
        self._originals: list[nn.Module] = []

    def __enter__(self) -> ActivationCapture:
        """Replace blocks with capturing wrappers."""
        self.activations.clear()
        self._originals = list(self.model.blocks)

        for i, block in enumerate(self._originals):
            wrapper = _CaptureWrapper(block, i, self.activations)
            self.model.blocks[i] = wrapper  # type: ignore[call-overload]

        return self

    def __exit__(self, *exc: Any) -> None:
        """Restore original blocks."""
        for i, block in enumerate(self._originals):
            self.model.blocks[i] = block  # type: ignore[call-overload]
        self._originals.clear()

    def layer_norms(self) -> dict[str, float]:
        """Compute L2 norm of each captured activation.

        Returns:
            Dict mapping activation key to its L2 norm.
        """
        norms: dict[str, float] = {}
        for key, val in self.activations.items():
            mx.eval(val)
            norm = mx.sqrt(mx.sum(val * val)).item()
            norms[key] = float(norm)
        return norms
