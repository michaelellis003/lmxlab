"""Tests for MLX profiling utilities."""

import mlx.core as mx

from lmt_metal.experiments.profiling import (
    benchmark_fn,
    count_parameters_by_module,
    memory_estimate,
    profile_forward,
    profile_generation,
)
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.models.llama import llama_tiny


class TestBenchmarkFn:
    """Test the benchmark_fn timing utility."""

    def test_returns_timing_dict(self):
        def fn():
            x = mx.random.normal((32, 64))
            y = x @ x.T
            mx.eval(y)

        result = benchmark_fn(fn, n_warmup=1, n_iter=3)
        assert "mean_ms" in result
        assert "std_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result
        assert "n_iter" in result
        assert result["n_iter"] == 3

    def test_mean_is_positive(self):
        def fn():
            mx.eval(mx.zeros((10,)))

        result = benchmark_fn(fn, n_warmup=1, n_iter=5)
        assert result["mean_ms"] >= 0

    def test_custom_iterations(self):
        result = benchmark_fn(
            lambda: mx.eval(mx.zeros((1,))),
            n_warmup=0,
            n_iter=10,
        )
        assert result["n_iter"] == 10


class TestMemoryEstimate:
    """Test memory estimation for models."""

    def test_returns_dict(self):
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        est = memory_estimate(model)
        assert "total_bytes" in est
        assert "total_mb" in est
        assert "param_count" in est

    def test_positive_values(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        est = memory_estimate(model)
        assert est["total_bytes"] > 0
        assert est["param_count"] > 0

    def test_dtype_affects_size(self):
        """Float32 model should use more memory than float16."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        est = memory_estimate(model)
        # Default should be float32 (4 bytes per param)
        expected_bytes = est["param_count"] * 4
        # Allow some tolerance for metadata
        assert abs(est["total_bytes"] - expected_bytes) < 1000


class TestCountParametersByModule:
    """Test per-module parameter breakdown."""

    def test_returns_dict(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        breakdown = count_parameters_by_module(model)
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

    def test_contains_expected_keys(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        breakdown = count_parameters_by_module(model)
        # Should have embed, blocks, final_norm, head
        assert "embed" in breakdown
        assert "blocks" in breakdown

    def test_total_matches_model(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        breakdown = count_parameters_by_module(model)
        total_from_breakdown = sum(breakdown.values())
        total_from_model = model.count_parameters()

        assert total_from_breakdown == total_from_model


class TestProfileForward:
    """Test forward pass profiling."""

    def test_returns_timing(self):
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        tokens = mx.array([[1, 2, 3, 4]])
        result = profile_forward(model, tokens, n_iter=2)

        assert "mean_ms" in result
        assert "tokens_per_sec" in result
        assert result["seq_len"] == 4
        assert result["batch_size"] == 1

    def test_tokens_per_sec_positive(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        result = profile_forward(model, tokens, n_iter=3)

        assert result["tokens_per_sec"] > 0


class TestProfileGeneration:
    """Test autoregressive generation profiling."""

    def test_returns_expected_keys(self):
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3]])
        result = profile_generation(model, prompt, max_tokens=5)

        assert "prefill_ms" in result
        assert "decode_ms_per_token" in result
        assert "total_ms" in result
        assert "tokens_generated" in result
        assert "prompt_len" in result
        assert "decode_tokens_per_sec" in result

    def test_prompt_len_correct(self):
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3, 4, 5]])
        result = profile_generation(model, prompt, max_tokens=3)

        assert result["prompt_len"] == 5

    def test_tokens_generated(self):
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2]])
        result = profile_generation(model, prompt, max_tokens=10)

        assert result["tokens_generated"] == 10

    def test_timing_positive(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3]])
        result = profile_generation(model, prompt, max_tokens=5)

        assert result["prefill_ms"] > 0
        assert result["total_ms"] > 0
        assert result["decode_tokens_per_sec"] > 0
