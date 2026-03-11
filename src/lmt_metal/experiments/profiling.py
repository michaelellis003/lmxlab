"""MLX-specific profiling utilities.

Tools for benchmarking models on Apple Silicon: timing, memory
estimation, parameter breakdowns, and forward pass throughput.

Example::

    from lmt_metal.experiments.profiling import (
        benchmark_fn, memory_estimate, profile_forward,
    )

    model = LanguageModel(config)
    tokens = mx.array([[1, 2, 3, 4]])

    # Time a function
    timing = benchmark_fn(lambda: model(tokens), n_iter=10)
    print(f"Forward: {timing['mean_ms']:.2f} ms")

    # Memory estimate
    mem = memory_estimate(model)
    print(f"Model: {mem['total_mb']:.1f} MB")

    # Forward pass profiling
    result = profile_forward(model, tokens)
    print(f"Throughput: {result['tokens_per_sec']:.0f} tok/s")
"""

import math
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils


def benchmark_fn(
    fn,
    n_warmup: int = 3,
    n_iter: int = 10,
) -> dict[str, float]:
    """Time a function with warmup iterations.

    Runs the function n_warmup times (discarded), then n_iter
    times (timed). Returns timing statistics.

    Args:
        fn: Callable to benchmark (should include mx.eval).
        n_warmup: Number of warmup iterations.
        n_iter: Number of timed iterations.

    Returns:
        Dict with mean_ms, std_ms, min_ms, max_ms, n_iter.
    """
    # Warmup
    for _ in range(n_warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)

    return {
        "mean_ms": mean,
        "std_ms": math.sqrt(variance),
        "min_ms": min(times),
        "max_ms": max(times),
        "n_iter": n_iter,
    }


def memory_estimate(model: nn.Module) -> dict[str, Any]:
    """Estimate model memory usage from parameter shapes and dtypes.

    This is a static estimate based on parameter tensors. Actual
    memory usage during inference includes activations, KV cache,
    and MLX graph overhead.

    Args:
        model: Model to estimate.

    Returns:
        Dict with total_bytes, total_mb, param_count,
        and per-dtype breakdown.
    """
    flat = mlx.utils.tree_flatten(model.parameters())
    total_bytes = 0
    param_count = 0
    dtype_bytes = {}

    for _, p in flat:
        nbytes = p.nbytes
        total_bytes += nbytes
        param_count += p.size
        dtype_name = str(p.dtype)
        dtype_bytes[dtype_name] = dtype_bytes.get(dtype_name, 0) + nbytes

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "param_count": param_count,
        "by_dtype": dtype_bytes,
    }


def count_parameters_by_module(
    model: nn.Module,
) -> dict[str, int]:
    """Count parameters per top-level submodule.

    Returns a dict mapping module names to their parameter
    counts, useful for understanding where parameters are
    concentrated.

    Args:
        model: Model to analyze.

    Returns:
        Dict mapping module name to parameter count.
    """
    result = {}
    for name, child in model.children().items():
        flat = mlx.utils.tree_flatten(child)
        count = sum(p.size for _, p in flat)
        if count > 0:
            result[name] = count
    return result


def profile_forward(
    model: nn.Module,
    tokens: mx.array,
    n_warmup: int = 2,
    n_iter: int = 5,
) -> dict[str, Any]:
    """Profile forward pass throughput.

    Times the model's forward pass and computes tokens/second.

    Args:
        model: Language model to profile.
        tokens: Input token IDs (batch, seq_len).
        n_warmup: Warmup iterations.
        n_iter: Timed iterations.

    Returns:
        Dict with timing stats, tokens_per_sec, batch_size,
        seq_len.
    """
    batch_size, seq_len = tokens.shape

    def run():
        logits, _ = model(tokens)
        mx.eval(logits)

    timing = benchmark_fn(run, n_warmup=n_warmup, n_iter=n_iter)

    total_tokens = batch_size * seq_len
    tokens_per_sec = (
        total_tokens / (timing["mean_ms"] / 1000)
        if timing["mean_ms"] > 0
        else 0
    )

    return {
        **timing,
        "tokens_per_sec": tokens_per_sec,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def profile_generation(
    model: nn.Module,
    prompt: mx.array,
    max_tokens: int = 50,
) -> dict[str, Any]:
    """Profile autoregressive generation throughput.

    Measures time-to-first-token (prompt processing) and
    per-token generation speed.

    Args:
        model: Language model.
        prompt: Prompt token IDs (1, prompt_len).
        max_tokens: Number of tokens to generate.

    Returns:
        Dict with prefill_ms, decode_ms_per_token,
        total_ms, tokens_generated.
    """
    # Prefill: process the full prompt
    t0 = time.perf_counter()
    logits, cache = model(prompt)
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_token, *[c for pair in cache for c in pair])
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Decode: generate token by token
    tokens_generated = 0
    t0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits, cache = model(next_token, cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token, *[c for pair in cache for c in pair])
        tokens_generated += 1
    decode_ms = (time.perf_counter() - t0) * 1000

    total_generated = tokens_generated + 1  # include first token
    decode_per_token = decode_ms / max(tokens_generated, 1)

    return {
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": decode_per_token,
        "total_ms": prefill_ms + decode_ms,
        "tokens_generated": total_generated,
        "prompt_len": prompt.shape[1],
        "decode_tokens_per_sec": (
            1000 / decode_per_token if decode_per_token > 0 else 0
        ),
    }
