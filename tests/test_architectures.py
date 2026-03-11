"""Tests for all model architecture config factories."""

import mlx.core as mx
import mlx.utils
import pytest

from lmt_metal.core.config import BlockConfig, ModelConfig
from lmt_metal.core.moe import MoEFFN
from lmt_metal.models.gemma import gemma_config, gemma_tiny
from lmt_metal.models.gpt import gpt_config, gpt_tiny
from lmt_metal.models.llama import llama_config, llama_tiny
from lmt_metal.models.mixtral import mixtral_config, mixtral_tiny
from lmt_metal.models.qwen import qwen_config, qwen_tiny

ALL_TINY_FACTORIES = [
    ("gpt", gpt_tiny),
    ("llama", llama_tiny),
    ("gemma", gemma_tiny),
    ("qwen", qwen_tiny),
    ("mixtral", mixtral_tiny),
]


@pytest.mark.parametrize(
    "name,factory",
    ALL_TINY_FACTORIES,
    ids=[t[0] for t in ALL_TINY_FACTORIES],
)
class TestArchitectureConfigs:
    """Common tests for all architecture config factories."""

    def test_returns_model_config(self, name, factory):
        config = factory()
        assert isinstance(config, ModelConfig)

    def test_tiny_has_small_dims(self, name, factory):
        config = factory()
        assert config.block.d_model <= 128
        assert config.n_layers <= 4

    def test_tiny_has_small_vocab(self, name, factory):
        config = factory()
        assert config.vocab_size <= 1024

    def test_block_config_valid(self, name, factory):
        config = factory()
        block = config.block
        assert block.head_dim > 0
        assert block.d_ff > 0
        assert block.effective_n_kv_heads > 0
        assert block.effective_n_kv_heads <= block.n_heads


class TestArchitectureDefaults:
    """Test that full-size configs have reasonable defaults."""

    def test_gpt_defaults(self):
        c = gpt_config()
        assert c.block.attention == "mha"
        assert c.block.norm == "layer_norm"
        assert c.block.bias is True

    def test_llama_defaults(self):
        c = llama_config()
        assert c.block.attention == "gqa"
        assert c.block.norm == "rms_norm"
        assert c.block.bias is False

    def test_gemma_defaults(self):
        c = gemma_config()
        assert c.block.attention == "gqa"
        assert c.block.n_kv_heads == 1  # multi-query
        assert c.tie_embeddings is True

    def test_qwen_defaults(self):
        c = qwen_config()
        assert c.block.bias is True  # Qwen uses bias
        assert c.block.rope_theta == 1000000.0  # high theta

    def test_mixtral_defaults(self):
        c = mixtral_config()
        assert c.block.ffn == "gated"
        assert c.block.rope_theta == 1000000.0


class TestArchitectureDifferences:
    """Comparative tests: verify architectures differ as expected."""

    def test_gpt_vs_llama_bias(self):
        gpt = gpt_config()
        llama = llama_config()
        assert gpt.block.bias is True
        assert llama.block.bias is False

    def test_gpt_vs_llama_norm(self):
        gpt = gpt_config()
        llama = llama_config()
        assert gpt.block.norm == "layer_norm"
        assert llama.block.norm == "rms_norm"

    def test_gemma_multi_query(self):
        gemma = gemma_config()
        llama = llama_config()
        assert gemma.block.n_kv_heads < llama.block.n_kv_heads


class TestMoEFFN:
    """Tests for MoE feed-forward network."""

    def test_output_shape(self):
        config = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        moe = MoEFFN(config, n_experts=4, top_k=2)
        mx.eval(moe.parameters())

        x = mx.random.normal(shape=(2, 8, 64))
        out = moe(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_more_experts_more_params(self):
        config = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        moe4 = MoEFFN(config, n_experts=4, top_k=2)
        moe8 = MoEFFN(config, n_experts=8, top_k=2)
        mx.eval(moe4.parameters(), moe8.parameters())

        p4 = sum(p.size for _, p in mlx.utils.tree_flatten(moe4.parameters()))
        p8 = sum(p.size for _, p in mlx.utils.tree_flatten(moe8.parameters()))
        assert p8 > p4
