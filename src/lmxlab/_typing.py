"""Shared type aliases for lmxlab."""

from typing import Any

import mlx.core as mx

# Common array type
Array = mx.array

# Shape type
Shape = tuple[int, ...]

# Generic nested dict (for configs, state dicts, etc.)
NestedDict = dict[str, Any]
