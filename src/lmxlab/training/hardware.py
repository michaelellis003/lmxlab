"""Hardware detection utilities for Apple Silicon."""

import mlx.core as mx

# Known FP32 peak TFLOP/s for Apple Silicon chips.
# Source: Apple tech specs and Geekbench ML benchmarks.
_APPLE_SILICON_TFLOPS: dict[str, float] = {
    "applegpu_g13s": 4.1,  # M3
    "applegpu_g13g": 4.1,  # M3 (variant)
    "applegpu_g15s": 6.5,  # M3 Pro (11-core)
    "applegpu_g15d": 6.5,  # M3 Pro (14-core)
    "applegpu_g15g": 6.5,  # M3 Pro (variant)
    "applegpu_g16s": 14.2,  # M3 Max
    "applegpu_g16g": 14.2,  # M3 Max (variant)
    "applegpu_g11p": 3.6,  # M2
    "applegpu_g14s": 7.0,  # M2 Pro
    "applegpu_g14g": 7.0,  # M2 Pro (variant)
    "applegpu_g14p": 13.6,  # M2 Max
    "applegpu_g13p": 2.6,  # M1
    "applegpu_g13d": 4.8,  # M1 Pro
}


def detect_peak_tflops() -> float | None:
    """Detect FP32 peak TFLOP/s for the current GPU.

    Uses ``mx.device_info()`` to identify the Apple Silicon
    architecture and returns a known peak value. Returns None
    if the architecture is not recognized.

    Returns:
        Peak FP32 TFLOP/s, or None if unknown.
    """
    info = mx.device_info()
    arch = info.get("architecture", "")
    return _APPLE_SILICON_TFLOPS.get(arch)
