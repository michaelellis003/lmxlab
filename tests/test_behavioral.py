"""Behavioral tests: invariance, directional, minimum functionality.

These tests verify *properties* of the system rather than exact
outputs, following the CheckList methodology (Ribeiro et al.).
"""

import pytest

from lmt_metal.core.config import BlockConfig, ModelConfig
from lmt_metal.core.registry import Registry
from lmt_metal.data.tokenizer import CharTokenizer

# -- Invariance tests: perturbations that should NOT change output ---------


class TestConfigInvariance:
    """Config properties should be invariant to construction order."""

    def test_head_dim_independent_of_other_fields(self):
        """head_dim depends only on d_model and n_heads."""
        c1 = BlockConfig(d_model=128, n_heads=4, d_ff=256)
        c2 = BlockConfig(d_model=128, n_heads=4, d_ff=512)
        assert c1.head_dim == c2.head_dim

    def test_model_config_block_lookup_stable(self):
        """get_block_config always returns same object for uniform."""
        block = BlockConfig(d_model=64)
        config = ModelConfig(block=block, n_layers=10)
        results = [config.get_block_config(i) for i in range(10)]
        assert all(r is block for r in results)


class TestTokenizerInvariance:
    """Tokenizer encode/decode should be stable."""

    def test_encode_deterministic(self):
        """Same input always produces same encoding."""
        tok = CharTokenizer("hello world")
        assert tok.encode("hello") == tok.encode("hello")

    def test_roundtrip_preserves_text(self):
        """encode -> decode is the identity function."""
        texts = ["hello", "a b c", "12345", "!@#"]
        tok = CharTokenizer("hello a b c 12345 !@#")
        for text in texts:
            assert tok.decode(tok.encode(text)) == text

    @pytest.mark.parametrize("text", ["a", "ab", "abc" * 100])
    def test_encode_length_equals_text_length(self, text):
        """Character tokenizer: len(encode(text)) == len(text)."""
        tok = CharTokenizer(text)
        assert len(tok.encode(text)) == len(text)


# -- Directional tests: expected direction of change ----------------------


class TestRegistryDirectional:
    """Registry behavior should change predictably with mutations."""

    def test_keys_grow_monotonically(self):
        """Adding entries always increases key count."""
        reg: Registry[int] = Registry("test")
        for i in range(5):
            assert len(reg.keys()) == i
            reg.register(f"key_{i}", i)
        assert len(reg.keys()) == 5

    def test_contains_only_after_register(self):
        """Key is not contained before registration, is after."""
        reg: Registry[int] = Registry("test")
        assert "x" not in reg
        reg.register("x", 1)
        assert "x" in reg


class TestConfigDirectional:
    """Config changes should affect derived properties predictably."""

    def test_more_heads_smaller_head_dim(self):
        """Doubling n_heads halves head_dim."""
        c1 = BlockConfig(d_model=256, n_heads=4)
        c2 = BlockConfig(d_model=256, n_heads=8)
        assert c2.head_dim == c1.head_dim // 2

    def test_explicit_kv_heads_always_used(self):
        """Setting n_kv_heads overrides default."""
        for kv in [1, 2, 4]:
            config = BlockConfig(n_heads=8, n_kv_heads=kv)
            assert config.effective_n_kv_heads == kv


# -- Minimum functionality tests: basic capabilities that must work --------


class TestMinimumFunctionality:
    """Core components must handle basic cases correctly."""

    def test_registry_single_entry(self):
        """Registry works with just one entry."""
        reg: Registry[str] = Registry("single")
        reg.register("only", "value")
        assert reg.get("only") == "value"

    def test_empty_registry_reports_no_keys(self):
        """Empty registry has no keys and good error message."""
        reg: Registry[int] = Registry("empty")
        assert reg.keys() == []
        with pytest.raises(KeyError, match="Available: \\[\\]"):
            reg.get("anything")

    def test_char_tokenizer_single_char(self):
        """Tokenizer handles single-character text."""
        tok = CharTokenizer("a")
        assert tok.vocab_size == 1
        assert tok.encode("a") == [0]
        assert tok.decode([0]) == "a"

    def test_block_config_all_defaults(self):
        """BlockConfig with all defaults is valid."""
        config = BlockConfig()
        assert config.head_dim > 0
        assert config.effective_n_kv_heads > 0
        assert config.d_ff > config.d_model

    def test_model_config_single_layer(self):
        """ModelConfig works with a single layer."""
        config = ModelConfig(n_layers=1)
        assert config.get_block_config(0) is config.block
