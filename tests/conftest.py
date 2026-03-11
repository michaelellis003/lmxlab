"""Shared test fixtures for lmt-metal."""

import mlx.core as mx
import pytest


@pytest.fixture
def tiny_vocab_size() -> int:
    """Small vocabulary for fast tests."""
    return 32


@pytest.fixture
def small_dims() -> dict[str, int]:
    """Small model dimensions for fast tests."""
    return {
        "d_model": 64,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_layers": 2,
        "d_ff": 128,
    }


@pytest.fixture
def batch_tokens(tiny_vocab_size: int) -> mx.array:
    """Small batch of random token IDs. Shape: (2, 16)."""
    return mx.random.randint(0, tiny_vocab_size, shape=(2, 16))


@pytest.fixture
def random_hidden() -> mx.array:
    """Random hidden states. Shape: (2, 16, 64)."""
    return mx.random.normal(shape=(2, 16, 64))
