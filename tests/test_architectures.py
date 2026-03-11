"""Tests for all model architecture config factories."""

import mlx.core as mx
import mlx.utils
import pytest

from lmt_metal.core.attention import SlidingWindowGQA, _apply_sliding_window
from lmt_metal.core.config import BlockConfig, ModelConfig
from lmt_metal.core.deltanet import GatedDeltaNet
from lmt_metal.core.mla import MLA
from lmt_metal.core.moe import MoEFFN, SharedExpertMoEFFN
from lmt_metal.models.deepseek import deepseek_config, deepseek_tiny
from lmt_metal.models.gemma import gemma_config, gemma_tiny
from lmt_metal.models.gemma3 import gemma3_config, gemma3_tiny
from lmt_metal.models.gpt import gpt_config, gpt_tiny
from lmt_metal.models.llama import llama_config, llama_tiny
from lmt_metal.models.mixtral import mixtral_config, mixtral_tiny
from lmt_metal.models.qwen import qwen_config, qwen_tiny
from lmt_metal.models.qwen35 import qwen35_config, qwen35_tiny

ALL_TINY_FACTORIES = [
    ("gpt", gpt_tiny),
    ("llama", llama_tiny),
    ("gemma", gemma_tiny),
    ("qwen", qwen_tiny),
    ("mixtral", mixtral_tiny),
    ("deepseek", deepseek_tiny),
    ("gemma3", gemma3_tiny),
    ("qwen35", qwen35_tiny),
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

    def test_deepseek_defaults(self):
        c = deepseek_config()
        assert c.block.attention == "mla"
        assert c.block.kv_lora_rank == 512
        assert c.block.q_lora_rank == 1536
        assert c.block.rope_dim == 64
        assert c.block.bias is False


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


class TestMLA:
    """Tests for Multi-Head Latent Attention."""

    def test_output_shape(self):
        config = BlockConfig(
            attention="mla",
            d_model=64,
            n_heads=4,
            d_ff=128,
            kv_lora_rank=16,
            q_lora_rank=32,
            rope_dim=8,
        )
        mla = MLA(config)
        mx.eval(mla.parameters())

        x = mx.random.normal(shape=(2, 8, 64))
        out, cache = mla(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_cache_is_compressed(self):
        """MLA cache should be smaller than full KV cache."""
        config = BlockConfig(
            attention="mla",
            d_model=64,
            n_heads=4,
            d_ff=128,
            kv_lora_rank=16,
            q_lora_rank=32,
            rope_dim=8,
        )
        mla = MLA(config)
        mx.eval(mla.parameters())

        x = mx.random.normal(shape=(1, 8, 64))
        _, cache = mla(x)
        mx.eval(cache[0], cache[1])

        # Cache[0] is compressed latent (B, L, kv_lora_rank)
        assert cache[0].shape == (1, 8, 16)  # kv_lora_rank

    def test_kv_cache_incremental(self):
        """Test that MLA KV cache works for autoregressive generation."""
        config = BlockConfig(
            attention="mla",
            d_model=64,
            n_heads=4,
            d_ff=128,
            kv_lora_rank=16,
            q_lora_rank=32,
            rope_dim=8,
        )
        mla = MLA(config)
        mx.eval(mla.parameters())

        # First pass: full sequence
        x = mx.random.normal(shape=(1, 4, 64))
        out1, cache = mla(x)
        mx.eval(out1, cache[0], cache[1])

        # Second pass: single token with cache
        x2 = mx.random.normal(shape=(1, 1, 64))
        out2, cache2 = mla(x2, cache=cache)
        mx.eval(out2, cache2[0], cache2[1])

        # Cache should grow by 1 in sequence dimension
        # c_kv cache is (B, L, kv_lora_rank)
        assert cache2[0].shape[1] == 5  # 4 + 1

    def test_requires_kv_lora_rank(self):
        """MLA should raise if kv_lora_rank is not set."""
        config = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        with pytest.raises(ValueError, match="kv_lora_rank"):
            MLA(config)

    def test_no_rope_dim(self):
        """MLA should work without decoupled RoPE (rope_dim=0)."""
        config = BlockConfig(
            attention="mla",
            d_model=64,
            n_heads=4,
            d_ff=128,
            kv_lora_rank=16,
            rope_dim=0,
        )
        mla = MLA(config)
        mx.eval(mla.parameters())

        x = mx.random.normal(shape=(1, 4, 64))
        out, cache = mla(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_no_q_compression(self):
        """MLA should work without Q compression."""
        config = BlockConfig(
            attention="mla",
            d_model=64,
            n_heads=4,
            d_ff=128,
            kv_lora_rank=16,
            rope_dim=8,
        )
        mla = MLA(config)
        mx.eval(mla.parameters())

        x = mx.random.normal(shape=(1, 4, 64))
        out, cache = mla(x)
        mx.eval(out)
        assert out.shape == x.shape


class TestSharedExpertMoEFFN:
    """Tests for MoE with shared experts."""

    def test_output_shape(self):
        """Output shape must match input shape."""
        config = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        moe = SharedExpertMoEFFN(config, n_experts=4, top_k=2, n_shared=1)
        mx.eval(moe.parameters())

        x = mx.random.normal(shape=(2, 8, 64))
        out = moe(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_shared_expert_always_active(self):
        """Shared expert should contribute even for a single token."""
        config = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        moe = SharedExpertMoEFFN(config, n_experts=4, top_k=1, n_shared=1)
        mx.eval(moe.parameters())

        x = mx.random.normal(shape=(1, 1, 64))
        out = moe(x)
        mx.eval(out)

        # Output should not be zero -- shared expert always
        # contributes regardless of routing
        assert mx.any(out != 0.0).item()

    def test_more_experts_more_params(self):
        """More routed experts means more parameters."""
        config = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        moe4 = SharedExpertMoEFFN(config, n_experts=4, top_k=2, n_shared=1)
        moe8 = SharedExpertMoEFFN(config, n_experts=8, top_k=2, n_shared=1)
        mx.eval(moe4.parameters(), moe8.parameters())

        p4 = sum(p.size for _, p in mlx.utils.tree_flatten(moe4.parameters()))
        p8 = sum(p.size for _, p in mlx.utils.tree_flatten(moe8.parameters()))
        assert p8 > p4

    def test_bias_routing_preserves_shape(self):
        """Bias-based routing should not change output shape."""
        config = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        moe = SharedExpertMoEFFN(config, n_experts=4, top_k=2, n_shared=1)
        mx.eval(moe.parameters())

        # Set non-zero bias to exercise the bias path
        moe.expert_bias = mx.array([1.0, -1.0, 0.5, -0.5])

        x = mx.random.normal(shape=(2, 8, 64))
        out = moe(x)
        mx.eval(out)
        assert out.shape == x.shape


class TestSlidingWindowGQA:
    """Tests for Sliding Window GQA attention."""

    def test_output_shape(self):
        """Output shape must match input shape."""
        config = BlockConfig(
            attention="sliding_window_gqa",
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            window_size=4,
        )
        attn = SlidingWindowGQA(config)
        mx.eval(attn.parameters())

        x = mx.random.normal(shape=(2, 8, 64))
        mask = mx.zeros((8, 8))
        out, cache = attn(x, mask=mask)
        mx.eval(out)
        assert out.shape == x.shape

    def test_sliding_window_mask_restricts_range(self):
        """Window mask should block tokens beyond window_size."""
        window_size = 3
        seq_len = 6
        mask = _apply_sliding_window(
            mask=None,
            window_size=window_size,
            seq_len=seq_len,
            cache_len=0,
        )
        mx.eval(mask)

        # For each row i, columns < max(0, i - window_size + 1)
        # should be masked (-1e9)
        for i in range(seq_len):
            for j in range(seq_len):
                val = mask[i, j].item()
                if j > i:
                    # Future tokens: should be masked by causal
                    # (not enforced here, only window)
                    pass
                elif j < i - window_size + 1:
                    # Beyond window: should be masked
                    assert val < -1e8, f"row={i}, col={j} should be masked"
                else:
                    # Within window: should be 0
                    assert val == 0.0, f"row={i}, col={j} should be 0"

    def test_requires_window_size(self):
        """SlidingWindowGQA should raise without window_size."""
        config = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        with pytest.raises(ValueError, match="window_size"):
            SlidingWindowGQA(config)


class TestGemma3Config:
    """Tests for Gemma 3 configuration factory."""

    def test_gemma3_defaults(self):
        """Full-size config should have expected defaults."""
        c = gemma3_config()
        assert c.block.attention == "sliding_window_gqa"
        assert c.block.norm == "rms_norm"
        assert c.block.ffn == "gated"
        assert c.block.position == "rope"
        assert c.block.bias is False
        assert c.block.window_size == 4096
        assert c.n_layers == 26
        assert c.tie_embeddings is True

    def test_interleaved_attention_types(self):
        """Gemma 3 should have different attention across layers."""
        c = gemma3_config()
        assert c.block_configs is not None
        attn_types = [c.block_configs[i].attention for i in range(c.n_layers)]
        # Should have both types
        assert "gqa" in attn_types
        assert "sliding_window_gqa" in attn_types
        # Every 6th layer (0-indexed: 5, 11, 17, 23) is global
        for i in range(c.n_layers):
            if (i + 1) % 6 == 0:
                assert c.block_configs[i].attention == "gqa"
                assert c.block_configs[i].window_size is None
            else:
                assert c.block_configs[i].attention == "sliding_window_gqa"
                assert c.block_configs[i].window_size == 4096

    def test_gemma3_tiny_small_dims(self):
        """Tiny config should have small dimensions."""
        c = gemma3_tiny()
        assert c.block.d_model <= 128
        assert c.n_layers <= 6
        assert c.vocab_size <= 1024

    def test_gemma3_tiny_has_window(self):
        """Tiny config should use sliding window."""
        c = gemma3_tiny()
        assert c.block.window_size == 16
        assert c.block_configs is not None
        # Layer 3 (index 3, every 4th) should be global
        assert c.block_configs[3].attention == "gqa"
        # Layer 0 should be sliding window
        assert c.block_configs[0].attention == "sliding_window_gqa"


class TestGatedDeltaNet:
    """Tests for Gated DeltaNet attention."""

    def test_output_shape(self):
        """Output shape must match input shape."""
        config = BlockConfig(
            attention="gated_deltanet",
            d_model=64,
            n_heads=4,
            d_ff=128,
        )
        dn = GatedDeltaNet(config)
        mx.eval(dn.parameters())

        x = mx.random.normal(shape=(2, 8, 64))
        out, cache = dn(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_state_is_fixed_size(self):
        """DeltaNet state should be (B, H, d, d) regardless
        of sequence length."""
        config = BlockConfig(
            attention="gated_deltanet",
            d_model=64,
            n_heads=4,
            d_ff=128,
        )
        dn = GatedDeltaNet(config)
        mx.eval(dn.parameters())

        # Short sequence
        x_short = mx.random.normal(shape=(1, 4, 64))
        _, cache_short = dn(x_short)
        mx.eval(cache_short[0])

        # Long sequence
        x_long = mx.random.normal(shape=(1, 32, 64))
        _, cache_long = dn(x_long)
        mx.eval(cache_long[0])

        # State shape should be the same
        assert cache_short[0].shape == cache_long[0].shape
        assert cache_short[0].shape == (1, 4, 16, 16)

    def test_incremental_generation(self):
        """DeltaNet should support token-by-token generation."""
        config = BlockConfig(
            attention="gated_deltanet",
            d_model=64,
            n_heads=4,
            d_ff=128,
        )
        dn = GatedDeltaNet(config)
        mx.eval(dn.parameters())

        # Full sequence
        x = mx.random.normal(shape=(1, 4, 64))
        _, cache = dn(x)
        mx.eval(cache[0])

        # Single token with cache
        x2 = mx.random.normal(shape=(1, 1, 64))
        out2, cache2 = dn(x2, cache=cache)
        mx.eval(out2, cache2[0])

        assert out2.shape == (1, 1, 64)
        # State shape unchanged
        assert cache2[0].shape == cache[0].shape

    def test_with_short_conv(self):
        """DeltaNet with causal convolutions."""
        config = BlockConfig(
            attention="gated_deltanet",
            d_model=64,
            n_heads=4,
            d_ff=128,
            use_short_conv=True,
            conv_kernel_size=4,
        )
        dn = GatedDeltaNet(config)
        mx.eval(dn.parameters())

        x = mx.random.normal(shape=(2, 8, 64))
        out, cache = dn(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_gates_start_conservative(self):
        """Gate biases should initialize negative (near-zero gates)."""
        config = BlockConfig(
            attention="gated_deltanet",
            d_model=64,
            n_heads=4,
            d_ff=128,
        )
        dn = GatedDeltaNet(config)
        mx.eval(dn.parameters())

        # Decay and update gate biases should be negative
        mx.eval(dn.decay_proj.bias, dn.update_proj.bias)
        assert mx.all(dn.decay_proj.bias < 0).item()
        assert mx.all(dn.update_proj.bias < 0).item()


class TestQwen35Config:
    """Tests for Qwen 3.5 configuration factory."""

    def test_qwen35_defaults(self):
        """Full-size config should have expected defaults."""
        c = qwen35_config()
        assert c.block.attention == "gated_deltanet"
        assert c.block.norm == "rms_norm"
        assert c.block.ffn == "gated"
        assert c.block.use_short_conv is True
        assert c.n_layers == 28
        assert c.tie_embeddings is False

    def test_hybrid_attention_pattern(self):
        """Qwen 3.5 should have 3:1 DeltaNet:GQA pattern."""
        c = qwen35_config()
        assert c.block_configs is not None
        attn_types = [c.block_configs[i].attention for i in range(c.n_layers)]
        # Should have both types
        assert "gated_deltanet" in attn_types
        assert "gqa" in attn_types
        # Every 4th layer (0-indexed: 3, 7, 11, ...) is GQA
        for i in range(c.n_layers):
            if (i + 1) % 4 == 0:
                assert c.block_configs[i].attention == "gqa"
            else:
                assert c.block_configs[i].attention == "gated_deltanet"

    def test_qwen35_tiny_small_dims(self):
        """Tiny config should have small dimensions."""
        c = qwen35_tiny()
        assert c.block.d_model <= 128
        assert c.n_layers <= 4
        assert c.vocab_size <= 1024

    def test_qwen35_tiny_has_hybrid(self):
        """Tiny config should have hybrid attention."""
        c = qwen35_tiny()
        assert c.block_configs is not None
        # Layer 3 (index 3, every 4th) should be GQA
        assert c.block_configs[3].attention == "gqa"
        # Layer 0 should be DeltaNet
        assert c.block_configs[0].attention == "gated_deltanet"
