"""Tests for lmt-metal core modules."""

import mlx.core as mx
import pytest

from lmt_metal.core.attention import GQA, MHA, attention_registry
from lmt_metal.core.block import ConfigurableBlock
from lmt_metal.core.config import BlockConfig, ModelConfig
from lmt_metal.core.ffn import GatedFFN, StandardFFN, ffn_registry
from lmt_metal.core.norm import LayerNorm, RMSNorm, norm_registry
from lmt_metal.core.position import RoPE, Sinusoidal, position_registry
from lmt_metal.core.registry import Registry

# -- Config tests ----------------------------------------------------------


class TestBlockConfig:
    def test_frozen(self):
        config = BlockConfig()
        with pytest.raises(AttributeError):
            config.d_model = 128  # type: ignore[misc]

    def test_defaults(self):
        config = BlockConfig()
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.pre_norm is True

    def test_head_dim(self):
        config = BlockConfig(d_model=256, n_heads=4)
        assert config.head_dim == 64

    def test_effective_n_kv_heads_default(self):
        config = BlockConfig(n_heads=8)
        assert config.effective_n_kv_heads == 8

    def test_effective_n_kv_heads_explicit(self):
        config = BlockConfig(n_heads=8, n_kv_heads=2)
        assert config.effective_n_kv_heads == 2


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig()
        assert config.vocab_size == 32000
        assert config.n_layers == 6
        assert config.tie_embeddings is True

    def test_get_block_config_uniform(self):
        block = BlockConfig(d_model=128)
        config = ModelConfig(block=block, n_layers=4)
        for i in range(4):
            assert config.get_block_config(i) is block

    def test_get_block_config_per_layer(self):
        blocks = tuple(BlockConfig(d_model=64 * (i + 1)) for i in range(3))
        config = ModelConfig(block=blocks[0], n_layers=3, block_configs=blocks)
        assert config.get_block_config(0).d_model == 64
        assert config.get_block_config(2).d_model == 192


# -- Registry tests ---------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self):
        reg: Registry[int] = Registry("test")
        reg.register("a", 1)
        assert reg.get("a") == 1

    def test_duplicate_raises(self):
        reg: Registry[int] = Registry("test")
        reg.register("a", 1)
        with pytest.raises(ValueError, match="already has key"):
            reg.register("a", 2)

    def test_missing_raises(self):
        reg: Registry[int] = Registry("test")
        with pytest.raises(KeyError, match="no key"):
            reg.get("missing")

    def test_keys(self):
        reg: Registry[int] = Registry("test")
        reg.register("b", 2)
        reg.register("a", 1)
        assert reg.keys() == ["a", "b"]

    def test_contains(self):
        reg: Registry[int] = Registry("test")
        reg.register("x", 0)
        assert "x" in reg
        assert "y" not in reg


# -- Attention registry populated -------------------------------------------


class TestAttentionRegistry:
    def test_mha_registered(self):
        assert "mha" in attention_registry

    def test_gqa_registered(self):
        assert "gqa" in attention_registry


# -- Attention forward pass tests -------------------------------------------


class TestMHA:
    def test_output_shape(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        mha = MHA(config)
        mx.eval(mha.parameters())
        out, cache = mha(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_cache_shape(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        mha = MHA(config)
        mx.eval(mha.parameters())
        _, cache = mha(random_hidden)
        mx.eval(cache[0], cache[1])
        B, L, _ = random_hidden.shape
        assert cache[0].shape == (B, small_dims["n_heads"], L, config.head_dim)


class TestGQA:
    def test_output_shape(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        gqa = GQA(config)
        mx.eval(gqa.parameters())
        out, cache = gqa(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_fewer_kv_heads(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        gqa = GQA(config)
        mx.eval(gqa.parameters())
        _, cache = gqa(random_hidden)
        mx.eval(cache[0], cache[1])
        B, L, _ = random_hidden.shape
        assert cache[0].shape == (
            B,
            small_dims["n_kv_heads"],
            L,
            config.head_dim,
        )


# -- FFN tests --------------------------------------------------------------


class TestFFN:
    def test_standard_shape(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        ffn = StandardFFN(config)
        mx.eval(ffn.parameters())
        out = ffn(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_gated_shape(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        ffn = GatedFFN(config)
        mx.eval(ffn.parameters())
        out = ffn(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_registry_populated(self):
        assert "standard" in ffn_registry
        assert "gated" in ffn_registry


# -- Norm tests --------------------------------------------------------------


class TestNorm:
    def test_rms_norm_shape(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        norm = RMSNorm(config)
        out = norm(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_layer_norm_shape(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        norm = LayerNorm(config)
        out = norm(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_registry_populated(self):
        assert "rms_norm" in norm_registry
        assert "layer_norm" in norm_registry


# -- Position encoding tests ------------------------------------------------


class TestPosition:
    def test_rope_shapes(self, small_dims):
        config = BlockConfig(**small_dims)
        pos = RoPE(config)
        B, L = 2, 16
        q = mx.random.normal(
            shape=(B, small_dims["n_heads"], L, config.head_dim)
        )
        k = mx.random.normal(
            shape=(B, small_dims["n_kv_heads"], L, config.head_dim)
        )
        rq, rk = pos(q, k)
        mx.eval(rq, rk)
        assert rq.shape == q.shape
        assert rk.shape == k.shape

    def test_sinusoidal_shape(self, small_dims, random_hidden):
        config = BlockConfig(**small_dims)
        pos = Sinusoidal(config)
        out = pos(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_registry_populated(self):
        assert "rope" in position_registry
        assert "sinusoidal" in position_registry
        assert "alibi" in position_registry


# -- ConfigurableBlock tests ------------------------------------------------


class TestConfigurableBlock:
    def test_gpt_style_block(self, small_dims, random_hidden):
        """GPT-style: LayerNorm + MHA + StandardFFN + Sinusoidal."""
        config = BlockConfig(
            attention="mha",
            ffn="standard",
            norm="layer_norm",
            position="sinusoidal",
            **small_dims,
        )
        block = ConfigurableBlock(config)
        mx.eval(block.parameters())
        out, cache = block(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_llama_style_block(self, small_dims, random_hidden):
        """LLaMA-style: RMSNorm + GQA + GatedFFN + RoPE."""
        config = BlockConfig(
            attention="gqa",
            ffn="gated",
            norm="rms_norm",
            position="rope",
            **small_dims,
        )
        block = ConfigurableBlock(config)
        mx.eval(block.parameters())
        out, cache = block(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_post_norm(self, small_dims, random_hidden):
        config = BlockConfig(
            pre_norm=False,
            **small_dims,
        )
        block = ConfigurableBlock(config)
        mx.eval(block.parameters())
        out, _ = block(random_hidden)
        mx.eval(out)
        assert out.shape == random_hidden.shape

    def test_with_cache(self, small_dims):
        """Test that KV cache grows on subsequent calls."""
        config = BlockConfig(**small_dims)
        block = ConfigurableBlock(config)
        mx.eval(block.parameters())

        # First pass
        x = mx.random.normal(shape=(1, 8, small_dims["d_model"]))
        _, cache = block(x)
        mx.eval(cache[0], cache[1])
        assert cache[0].shape[2] == 8  # seq_len in cache

        # Second pass (single token)
        x2 = mx.random.normal(shape=(1, 1, small_dims["d_model"]))
        _, cache2 = block(x2, cache=cache)
        mx.eval(cache2[0], cache2[1])
        assert cache2[0].shape[2] == 9  # cache grew
