"""Behavioral tests: invariance, directional, minimum functionality.

These tests verify *properties* of the system rather than exact
outputs, following the CheckList methodology (Ribeiro et al.).
"""

from dataclasses import replace

import mlx.core as mx
import pytest

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.core.registry import Registry
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.deepseek import deepseek_tiny
from lmxlab.models.gemma import gemma_tiny
from lmxlab.models.gemma3 import gemma3_tiny
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.llama import llama_tiny
from lmxlab.models.mixtral import mixtral_tiny
from lmxlab.models.qwen35 import qwen35_tiny

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


# -- Architecture factory invariants ----------------------------------------

ALL_TINY_FACTORIES = [
    gpt_tiny,
    llama_tiny,
    gemma_tiny,
    mixtral_tiny,
    deepseek_tiny,
    gemma3_tiny,
    qwen35_tiny,
]


class TestArchitectureInvariance:
    """Properties that hold for ALL architecture configs."""

    @pytest.mark.parametrize(
        "factory",
        ALL_TINY_FACTORIES,
        ids=lambda f: f.__name__,
    )
    def test_tiny_configs_are_small(self, factory):
        """All tiny configs should have d_model <= 128."""
        config = factory()
        assert config.block.d_model <= 128

    @pytest.mark.parametrize(
        "factory",
        ALL_TINY_FACTORIES,
        ids=lambda f: f.__name__,
    )
    def test_tiny_configs_have_few_layers(self, factory):
        """All tiny configs should have <= 4 layers."""
        config = factory()
        assert config.n_layers <= 4

    @pytest.mark.parametrize(
        "factory",
        ALL_TINY_FACTORIES,
        ids=lambda f: f.__name__,
    )
    def test_vocab_size_positive(self, factory):
        """Every config must have a positive vocab size."""
        config = factory()
        assert config.vocab_size > 0

    @pytest.mark.parametrize(
        "factory",
        ALL_TINY_FACTORIES,
        ids=lambda f: f.__name__,
    )
    def test_head_dim_divides_d_model(self, factory):
        """d_model must be divisible by n_heads."""
        config = factory()
        assert config.block.d_model % config.block.n_heads == 0

    @pytest.mark.parametrize(
        "factory",
        ALL_TINY_FACTORIES,
        ids=lambda f: f.__name__,
    )
    def test_kv_heads_divide_query_heads(self, factory):
        """n_kv_heads must divide n_heads for GQA."""
        config = factory()
        kv = config.block.effective_n_kv_heads
        assert config.block.n_heads % kv == 0


class TestModelOutputInvariance:
    """Output properties that must hold regardless of architecture."""

    @pytest.mark.parametrize(
        "factory",
        ALL_TINY_FACTORIES,
        ids=lambda f: f.__name__,
    )
    def test_output_shape_matches_vocab(self, factory):
        """Logits last dim must equal vocab_size."""
        config = factory()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.random.randint(0, config.vocab_size, shape=(1, 8))
        logits, _ = model(tokens)
        mx.eval(logits)
        assert logits.shape == (1, 8, config.vocab_size)

    @pytest.mark.parametrize(
        "factory",
        ALL_TINY_FACTORIES,
        ids=lambda f: f.__name__,
    )
    def test_batch_independence(self, factory):
        """Adding to batch should not change existing outputs."""
        config = factory()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        tokens = mx.random.randint(0, config.vocab_size, shape=(1, 8))
        logits_single, _ = model(tokens)
        mx.eval(logits_single)

        # Same tokens duplicated in a batch of 2
        tokens_batch = mx.repeat(tokens, repeats=2, axis=0)
        logits_batch, _ = model(tokens_batch)
        mx.eval(logits_batch)

        diff = mx.abs(logits_single - logits_batch[0:1]).max()
        mx.eval(diff)
        assert diff.item() < 1e-4


# -- Directional tests for architecture scaling ----------------------------


class TestScalingDirectional:
    """Model properties should change predictably with scale."""

    def test_more_layers_more_parameters(self):
        """Increasing n_layers should increase parameter count."""
        config_small = replace(llama_tiny(), n_layers=2)
        config_large = replace(llama_tiny(), n_layers=4)
        model_small = LanguageModel(config_small)
        model_large = LanguageModel(config_large)
        assert model_large.count_parameters() > model_small.count_parameters()

    def test_wider_model_more_parameters(self):
        """Increasing d_model should increase parameter count."""
        block_narrow = BlockConfig(d_model=32, n_heads=2, d_ff=64)
        block_wide = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        c1 = ModelConfig(block=block_narrow, vocab_size=64, n_layers=2)
        c2 = ModelConfig(block=block_wide, vocab_size=64, n_layers=2)
        m1 = LanguageModel(c1)
        m2 = LanguageModel(c2)
        assert m2.count_parameters() > m1.count_parameters()

    def test_tied_embeddings_fewer_parameters(self):
        """Tied embeddings should reduce parameter count."""
        config_tied = ModelConfig(
            block=BlockConfig(d_model=64, n_heads=4, d_ff=128),
            vocab_size=256,
            n_layers=2,
            tie_embeddings=True,
        )
        config_untied = ModelConfig(
            block=BlockConfig(d_model=64, n_heads=4, d_ff=128),
            vocab_size=256,
            n_layers=2,
            tie_embeddings=False,
        )
        m_tied = LanguageModel(config_tied)
        m_untied = LanguageModel(config_untied)
        assert m_tied.count_parameters() < m_untied.count_parameters()


# -- Config factory directional tests --------------------------------------


class TestConfigFactoryDirectional:
    """Factory functions should produce distinct configs."""

    def test_gpt_uses_mha_not_gqa(self):
        """GPT uses standard MHA, not GQA."""
        config = gpt_tiny()
        assert config.block.attention == "mha"

    def test_llama_uses_gqa(self):
        """LLaMA uses grouped-query attention."""
        config = llama_tiny()
        assert config.block.attention == "gqa"

    def test_llama_has_no_bias(self):
        """LLaMA omits bias in linear layers."""
        config = llama_tiny()
        assert config.block.bias is False

    def test_gpt_has_bias(self):
        """GPT uses bias in linear layers."""
        config = gpt_tiny()
        assert config.block.bias is True

    def test_gpt_uses_layernorm(self):
        """GPT uses LayerNorm, not RMSNorm."""
        config = gpt_tiny()
        assert config.block.norm == "layer_norm"

    def test_llama_uses_rmsnorm(self):
        """LLaMA uses RMSNorm."""
        config = llama_tiny()
        assert config.block.norm == "rms_norm"

    def test_deepseek_uses_mla(self):
        """DeepSeek V2 uses Multi-Head Latent Attention."""
        config = deepseek_tiny()
        assert config.block.attention == "mla"
        assert config.block.kv_lora_rank is not None

    def test_mixtral_uses_moe(self):
        """Mixtral uses MoE feed-forward."""
        config = mixtral_tiny()
        assert config.block.n_experts is not None
        assert config.block.n_experts > 1

    def test_gemma3_has_per_layer_configs(self):
        """Gemma 3 uses different blocks for different layers."""
        config = gemma3_tiny()
        assert config.block_configs is not None
        attentions = {
            config.get_block_config(i).attention
            for i in range(config.n_layers)
        }
        # Should have both sliding window and global
        assert len(attentions) == 2

    def test_qwen35_has_hybrid_layers(self):
        """Qwen 3.5 mixes DeltaNet and GQA layers."""
        config = qwen35_tiny()
        assert config.block_configs is not None
        attentions = {
            config.get_block_config(i).attention
            for i in range(config.n_layers)
        }
        assert "gated_deltanet" in attentions
        assert "gqa" in attentions


# -- Frozen config immutability tests --------------------------------------


class TestConfigImmutability:
    """Frozen dataclasses should reject mutation."""

    def test_block_config_is_frozen(self):
        """Cannot modify BlockConfig attributes after creation."""
        config = BlockConfig()
        with pytest.raises(AttributeError):
            config.d_model = 999  # type: ignore[misc]

    def test_model_config_is_frozen(self):
        """Cannot modify ModelConfig attributes after creation."""
        config = ModelConfig()
        with pytest.raises(AttributeError):
            config.n_layers = 999  # type: ignore[misc]

    def test_replace_creates_new_config(self):
        """dataclasses.replace creates a new object."""
        c1 = ModelConfig(n_layers=4)
        c2 = replace(c1, n_layers=8)
        assert c1.n_layers == 4
        assert c2.n_layers == 8
        assert c1 is not c2
