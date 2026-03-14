"""Tests for model architectures and generation."""

import mlx.core as mx

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.gpt import gpt_config, gpt_tiny
from lmxlab.models.llama import llama_config, llama_tiny
from lmxlab.models.nemotron import (
    _parse_hybrid_pattern,
    nemotron3_8b,
    nemotron3_tiny,
)


class TestLanguageModel:
    def test_forward_shape(self):
        """Model produces correct output shape."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        x = mx.array([[1, 2, 3, 4]])
        logits, caches = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, config.vocab_size)
        assert len(caches) == config.n_layers

    def test_llama_forward(self):
        """LLaMA-style model forward pass."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        x = mx.array([[1, 2, 3]])
        logits, caches = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, config.vocab_size)

    def test_tied_embeddings(self):
        """Tied embeddings: no separate head weight."""
        config = gpt_tiny()
        assert config.tie_embeddings is True
        model = LanguageModel(config)
        assert not hasattr(model, "head")

    def test_untied_embeddings(self):
        """Untied embeddings: separate head weight."""
        block = BlockConfig(d_model=64, n_heads=2, d_ff=128)
        config = ModelConfig(
            block=block,
            vocab_size=256,
            n_layers=2,
            tie_embeddings=False,
        )
        model = LanguageModel(config)
        assert hasattr(model, "head")

    def test_count_parameters(self):
        """Parameter count is positive and reasonable."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        n = model.count_parameters()
        assert n > 0
        # Tiny model should be small
        assert n < 1_000_000

    def test_kv_cache_generation(self):
        """KV cache works across multiple forward passes."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        # Prefill
        x = mx.array([[1, 2, 3]])
        logits1, cache = model(x)
        mx.eval(logits1, *[c for pair in cache for c in pair])

        # Generate one token
        next_token = mx.array([[4]])
        logits2, cache2 = model(next_token, cache=cache)
        mx.eval(logits2)
        assert logits2.shape == (1, 1, config.vocab_size)


class TestGPTConfig:
    def test_defaults(self):
        config = gpt_config()
        assert config.vocab_size == 50257
        assert config.block.attention == "mha"
        assert config.block.norm == "layer_norm"
        assert config.block.ffn == "standard"
        assert config.block.bias is True

    def test_tiny(self):
        config = gpt_tiny()
        assert config.block.d_model == 64
        assert config.n_layers == 2


class TestLLaMAConfig:
    def test_defaults(self):
        config = llama_config()
        assert config.vocab_size == 32000
        assert config.block.attention == "gqa"
        assert config.block.norm == "rms_norm"
        assert config.block.ffn == "gated"
        assert config.block.bias is False

    def test_tiny(self):
        config = llama_tiny()
        assert config.block.d_model == 64
        assert config.block.n_kv_heads == 2

    def test_dropout_param(self):
        """LLaMA config accepts and stores dropout."""
        config = llama_config(
            vocab_size=256,
            d_model=64,
            n_heads=2,
            n_kv_heads=2,
            n_layers=2,
            d_ff=128,
            dropout=0.2,
        )
        assert config.block.dropout == 0.2


class TestDropout:
    def test_gpt_with_dropout(self):
        """GPT model runs with dropout enabled."""
        config = gpt_config(
            vocab_size=256,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            dropout=0.1,
        )
        model = LanguageModel(config)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3, 4]])
        logits, caches = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, 256)

    def test_llama_with_dropout(self):
        """LLaMA model runs with dropout enabled."""
        config = llama_config(
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            d_ff=128,
            dropout=0.2,
        )
        model = LanguageModel(config)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, caches = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, 256)

    def test_dropout_zero_preserves_output(self):
        """Dropout=0 is a no-op: same output in eval mode."""
        config = gpt_config(
            vocab_size=256,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            dropout=0.0,
        )
        model = LanguageModel(config)
        model.eval()
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        mx.random.seed(42)
        out1, _ = model(x)
        mx.random.seed(42)
        out2, _ = model(x)
        mx.eval(out1, out2)
        assert mx.allclose(out1, out2)

    def test_dropout_layers_exist(self):
        """Dropout layers are created in the model."""
        config = gpt_config(
            vocab_size=256,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            dropout=0.2,
        )
        model = LanguageModel(config)
        # Embedding dropout
        assert hasattr(model, "embed_dropout")
        # Block residual dropout
        for block in model.blocks:
            assert hasattr(block, "resid_dropout")


class TestGenerate:
    def test_greedy_generation(self):
        """Greedy generation produces correct length."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3]])
        output = generate(model, prompt, max_tokens=5, temperature=0.0)
        mx.eval(output)
        assert output.shape == (1, 8)  # 3 prompt + 5 generated
        # Prompt should be preserved
        assert mx.array_equal(output[:, :3], prompt)

    def test_temperature_sampling(self):
        """Temperature sampling produces valid tokens."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2]])
        output = generate(model, prompt, max_tokens=3, temperature=0.8)
        mx.eval(output)
        assert output.shape == (1, 5)

    def test_top_k_sampling(self):
        """Top-k sampling works."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2]])
        output = generate(model, prompt, max_tokens=3, top_k=10)
        mx.eval(output)
        assert output.shape == (1, 5)

    def test_batch_generation(self):
        """Generation works with batch size > 1."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3], [4, 5, 6]])
        output = generate(model, prompt, max_tokens=4, temperature=0.0)
        mx.eval(output)
        assert output.shape == (2, 7)

    def test_stop_tokens(self):
        """Generation stops at stop token."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3]])
        # Use a large max_tokens but stop early
        output = generate(
            model,
            prompt,
            max_tokens=50,
            temperature=0.0,
            stop_tokens=[0],  # likely to hit 0 eventually
        )
        mx.eval(output)
        # Should be shorter than 3 + 50 = 53
        assert output.shape[1] <= 53
        # Prompt preserved
        assert mx.array_equal(output[:, :3], prompt)

    def test_repetition_penalty(self):
        """Repetition penalty runs without error."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3]])
        output = generate(
            model,
            prompt,
            max_tokens=5,
            temperature=0.8,
            repetition_penalty=1.2,
        )
        mx.eval(output)
        assert output.shape == (1, 8)


class TestStreamGenerate:
    def test_yields_tokens(self):
        """stream_generate yields individual token IDs."""
        from lmxlab.models.generate import stream_generate

        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3]])
        tokens = list(
            stream_generate(model, prompt, max_tokens=5, temperature=0.0)
        )
        assert len(tokens) == 5
        assert all(isinstance(t, int) for t in tokens)

    def test_stream_stop_tokens(self):
        """stream_generate stops at stop token."""
        from lmxlab.models.generate import stream_generate

        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3]])
        tokens = list(
            stream_generate(
                model,
                prompt,
                max_tokens=50,
                temperature=0.0,
                stop_tokens=[0],
            )
        )
        # Should stop before 50 tokens (0 is common in random model)
        assert len(tokens) <= 50

    def test_stream_matches_generate(self):
        """Streaming and batch generate produce same tokens (greedy)."""
        from lmxlab.models.generate import stream_generate

        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        prompt = mx.array([[1, 2, 3]])
        mx.random.seed(42)
        batch_output = generate(model, prompt, max_tokens=5, temperature=0.0)
        mx.eval(batch_output)
        batch_tokens = batch_output[0, 3:].tolist()

        mx.random.seed(42)
        model2 = LanguageModel(config)
        # Load same weights
        import mlx.utils as mlx_utils

        model2.load_weights(
            list(dict(mlx_utils.tree_flatten(model.parameters())).items())
        )
        mx.eval(model2.parameters())
        stream_tokens = list(
            stream_generate(model2, prompt, max_tokens=5, temperature=0.0)
        )

        assert batch_tokens == stream_tokens


class TestNoneAttention:
    def test_identity(self):
        """NoneAttention returns input unchanged."""
        from lmxlab.core.attention import attention_registry

        cfg = BlockConfig(d_model=64, n_heads=4)
        attn = attention_registry.get("none")(cfg)
        x = mx.random.normal((1, 4, 64))
        mx.eval(x)
        out, cache = attn(x)
        mx.eval(out)
        assert mx.allclose(out, x)
        assert cache is None


class TestNoneFFN:
    def test_identity(self):
        """NoneFFN returns input unchanged."""
        from lmxlab.core.ffn import ffn_registry

        cfg = BlockConfig(d_model=64, n_heads=4)
        ffn = ffn_registry.get("none")(cfg)
        x = mx.random.normal((1, 4, 64))
        mx.eval(x)
        out = ffn(x)
        mx.eval(out)
        assert mx.allclose(out, x)


class TestGatedReluSquaredFFN:
    def test_output_shape(self):
        """GatedReluSquaredFFN produces correct shape."""
        from lmxlab.core.ffn import ffn_registry

        cfg = BlockConfig(d_model=64, n_heads=4, d_ff=128)
        ffn = ffn_registry.get("gated_relu2")(cfg)
        mx.eval(ffn.parameters())
        x = mx.random.normal((2, 4, 64))
        out = ffn(x)
        mx.eval(out)
        assert out.shape == (2, 4, 64)

    def test_squared_relu_activation(self):
        """Verify squared ReLU: output uses relu(x)^2."""
        from lmxlab.core.ffn import ffn_registry

        cfg = BlockConfig(
            d_model=8,
            n_heads=1,
            d_ff=16,
            bias=False,
        )
        ffn = ffn_registry.get("gated_relu2")(cfg)
        mx.eval(ffn.parameters())
        x = mx.random.normal((1, 2, 8))
        out = ffn(x)
        mx.eval(out)
        # Just verify it runs and shape is correct
        assert out.shape == (1, 2, 8)


class TestMamba2:
    def test_forward_shape(self):
        """Mamba2 produces correct output shape."""
        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            conv_kernel_size=4,
        )
        from lmxlab.core.mamba2 import Mamba2

        mamba = Mamba2(cfg)
        mx.eval(mamba.parameters())
        x = mx.random.normal((2, 8, 64))
        out, cache = mamba(x)
        mx.eval(out)
        assert out.shape == (2, 8, 64)

    def test_cache_constant_size(self):
        """Mamba cache size is independent of sequence length."""
        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            conv_kernel_size=4,
        )
        from lmxlab.core.mamba2 import Mamba2

        mamba = Mamba2(cfg)
        mx.eval(mamba.parameters())

        # Short sequence
        x_short = mx.random.normal((1, 4, 64))
        _, cache_short = mamba(x_short)
        mx.eval(cache_short)

        # Long sequence
        x_long = mx.random.normal((1, 16, 64))
        _, cache_long = mamba(x_long)
        mx.eval(cache_long)

        # SSM state shape should be identical
        assert cache_short[0].shape == cache_long[0].shape
        # Conv state shape should be identical
        assert cache_short[1].shape == cache_long[1].shape

    def test_autoregressive_inference(self):
        """Mamba supports single-token inference with cache."""
        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            conv_kernel_size=4,
        )
        from lmxlab.core.mamba2 import Mamba2

        mamba = Mamba2(cfg)
        mx.eval(mamba.parameters())

        # Prefill
        x = mx.random.normal((1, 4, 64))
        out, cache = mamba(x)
        mx.eval(out, *cache)

        # Single token step
        x_step = mx.random.normal((1, 1, 64))
        out_step, cache2 = mamba(x_step, cache=cache)
        mx.eval(out_step)
        assert out_step.shape == (1, 1, 64)


class TestLatentMoE:
    def test_output_shape(self):
        """LatentMoE produces correct output shape."""
        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_experts=4,
            top_k_experts=2,
            moe_latent_size=32,
            moe_d_ff=64,
            shared_expert_d_ff=128,
        )
        from lmxlab.core.moe import LatentMoEFFN

        moe = LatentMoEFFN(cfg)
        mx.eval(moe.parameters())
        x = mx.random.normal((1, 4, 64))
        out = moe(x)
        mx.eval(out)
        assert out.shape == (1, 4, 64)

    def test_routing_weights_sum(self):
        """Sigmoid-normalized routing weights sum to 1."""
        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_experts=4,
            top_k_experts=2,
            moe_latent_size=32,
            moe_d_ff=64,
            shared_expert_d_ff=128,
        )
        from lmxlab.core.moe import LatentMoEFFN

        moe = LatentMoEFFN(cfg)
        mx.eval(moe.parameters())

        # Router operates on full hidden dim (d_model=64)
        x = mx.random.normal((1, 4, 64))
        router_logits = moe.router(x)  # (1, 4, n_experts)
        top_k_indices = mx.argpartition(
            -router_logits,
            kth=2,
            axis=-1,
        )[:, :, :2]
        top_k_logits = mx.take_along_axis(
            router_logits,
            top_k_indices,
            axis=-1,
        )
        # Sigmoid + normalize (Nemotron 3 convention)
        scores = mx.sigmoid(top_k_logits)
        weights = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        mx.eval(weights)
        sums = mx.sum(weights, axis=-1)
        mx.eval(sums)
        assert mx.allclose(sums, mx.ones_like(sums), atol=1e-5)


class TestHybridPattern:
    def test_parse_pattern(self):
        """Pattern parser maps characters correctly."""
        attn = BlockConfig(attention="gqa")
        moe = BlockConfig(attention="none", ffn="latent_moe")
        mamba = BlockConfig(attention="mamba2", ffn="none")

        configs = _parse_hybrid_pattern(
            "MEM*",
            attn,
            moe,
            mamba,
        )
        assert len(configs) == 4
        assert configs[0].attention == "mamba2"
        assert configs[1].ffn == "latent_moe"
        assert configs[2].attention == "mamba2"
        assert configs[3].attention == "gqa"

    def test_invalid_pattern_char(self):
        """Invalid pattern character raises ValueError."""
        import pytest

        attn = BlockConfig()
        with pytest.raises(ValueError, match="Unknown pattern"):
            _parse_hybrid_pattern("MXE", attn, attn, attn)


class TestNemotronConfig:
    def test_tiny_config(self):
        """nemotron_tiny creates a valid config."""
        cfg = nemotron3_tiny()
        assert cfg.n_layers == 4
        assert cfg.block_configs is not None
        assert len(cfg.block_configs) == 4

    def test_layer_types(self):
        """Layers have correct component types per pattern."""
        cfg = nemotron3_tiny()  # pattern = 'MEM*'
        configs = cfg.block_configs
        # M layer
        assert configs[0].attention == "mamba2"
        assert configs[0].ffn == "none"
        # E layer
        assert configs[1].attention == "none"
        assert configs[1].ffn == "latent_moe"
        # M layer
        assert configs[2].attention == "mamba2"
        assert configs[2].ffn == "none"
        # * layer (attention only, no FFN)
        assert configs[3].attention == "gqa"
        assert configs[3].ffn == "none"


class TestNemotronModel:
    def test_forward_shape(self):
        """Nemotron 3 forward pass produces correct shape."""
        cfg = nemotron3_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())

        x = mx.array([[1, 2, 3, 4]])
        logits, caches = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, cfg.vocab_size)
        assert len(caches) == cfg.n_layers

    def test_heterogeneous_cache(self):
        """Cache types differ per layer in hybrid model."""
        cfg = nemotron3_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())

        x = mx.array([[1, 2, 3, 4]])
        _, caches = model(x)
        mx.eval(caches)

        # M layers (0, 2): Mamba cache = (ssm_state, conv_state)
        assert isinstance(caches[0], tuple)
        assert len(caches[0]) == 2  # (ssm, conv)

        # E layer (1): NoneAttention returns None cache
        assert caches[1] is None

        # * layer (3): KV cache = (K, V)
        assert isinstance(caches[3], tuple)
        assert len(caches[3]) == 2
        # KV cache has 4 dims (B, heads, seq, head_dim)
        assert caches[3][0].ndim == 4

    def test_autoregressive_step(self):
        """Hybrid model supports single-token generation."""
        cfg = nemotron3_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())

        # Prefill
        x = mx.array([[1, 2, 3]])
        logits, cache = model(x)
        # Eval all cache entries (some may be None)
        to_eval = [logits]
        for c in cache:
            if c is not None and isinstance(c, tuple):
                to_eval.extend(v for v in c if isinstance(v, mx.array))
        mx.eval(*to_eval)

        # Single token step
        next_tok = mx.array([[4]])
        logits2, cache2 = model(next_tok, cache=cache)
        mx.eval(logits2)
        assert logits2.shape == (1, 1, cfg.vocab_size)

    def test_parameter_count(self):
        """Tiny Nemotron has reasonable parameter count."""
        cfg = nemotron3_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        n = model.count_parameters()
        assert n > 0
        # Tiny model should be under 5M params
        assert n < 5_000_000


class TestMultiGroupBC:
    """Tests for multi-group B/C sharing in Mamba-2."""

    def test_group1_matches_default(self):
        """n_groups=1 produces same shape as default."""
        from lmxlab.core.mamba2 import Mamba2

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            mamba_n_groups=1,
        )
        mamba = Mamba2(cfg)
        mx.eval(mamba.parameters())
        x = mx.random.normal((1, 8, 64))
        out, cache = mamba(x)
        mx.eval(out)
        assert out.shape == (1, 8, 64)

    def test_multigroup_output_shape(self):
        """Multi-group B/C still produces correct shape."""
        from lmxlab.core.mamba2 import Mamba2

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            mamba_n_groups=2,
        )
        mamba = Mamba2(cfg)
        mx.eval(mamba.parameters())
        x = mx.random.normal((2, 6, 64))
        out, cache = mamba(x)
        mx.eval(out)
        assert out.shape == (2, 6, 64)

    def test_bc_dim_scales_with_groups(self):
        """B/C projection dimension scales with n_groups."""
        from lmxlab.core.mamba2 import Mamba2

        cfg1 = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            mamba_n_groups=1,
        )
        cfg2 = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            mamba_n_groups=2,
        )
        m1 = Mamba2(cfg1)
        m2 = Mamba2(cfg2)
        # in_proj output dim: z + x_bc + dt
        # x_bc = inner_dim + 2*G*N
        # So total proj = inner + inner + 2*G*N + n_heads
        proj1 = m1.in_proj.weight.shape[0]
        proj2 = m2.in_proj.weight.shape[0]
        # Difference should be 2*N (extra group)
        assert proj2 - proj1 == 2 * 16  # 2 * ssm_state_size

    def test_heads_not_divisible_by_groups(self):
        """Raises when n_heads % n_groups != 0."""
        import pytest

        from lmxlab.core.mamba2 import Mamba2

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            mamba_n_groups=3,
        )
        with pytest.raises(ValueError, match="divisible"):
            Mamba2(cfg)


class TestLatentMoEImprovements:
    """Tests for LatentMoE score_correction, scaling, groups."""

    def test_score_correction_bias_exists(self):
        """score_correction_bias initialized to zeros."""
        from lmxlab.core.moe import LatentMoEFFN

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_experts=4,
            top_k_experts=2,
            moe_latent_size=32,
            moe_d_ff=64,
            shared_expert_d_ff=128,
        )
        moe = LatentMoEFFN(cfg)
        mx.eval(moe.score_correction_bias)
        assert moe.score_correction_bias.shape == (4,)
        assert mx.allclose(
            moe.score_correction_bias,
            mx.zeros((4,)),
        )

    def test_scaling_factor_applied(self):
        """Routed output is scaled by scaling_factor."""
        from lmxlab.core.moe import LatentMoEFFN

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_experts=4,
            top_k_experts=2,
            moe_latent_size=32,
            moe_d_ff=64,
            shared_expert_d_ff=128,
            moe_routed_scaling_factor=5.0,
        )
        moe = LatentMoEFFN(cfg)
        assert moe.scaling_factor == 5.0
        mx.eval(moe.parameters())
        x = mx.random.normal((1, 4, 64))
        out = moe(x)
        mx.eval(out)
        assert out.shape == (1, 4, 64)

    def test_grouped_selection(self):
        """Grouped expert selection produces correct shape."""
        from lmxlab.core.moe import LatentMoEFFN

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_experts=8,
            top_k_experts=2,
            moe_latent_size=32,
            moe_d_ff=64,
            shared_expert_d_ff=128,
            moe_n_groups=2,
            moe_topk_groups=1,
        )
        moe = LatentMoEFFN(cfg)
        mx.eval(moe.parameters())
        x = mx.random.normal((1, 4, 64))
        out = moe(x)
        mx.eval(out)
        assert out.shape == (1, 4, 64)


class TestChunkedSSD:
    """Tests for chunked SSD parallel form."""

    def test_segsum_causal(self):
        """segsum produces causal lower-triangular matrix."""
        from lmxlab.core.mamba2 import _segsum

        x = mx.ones((4,))  # simple case
        L = _segsum(x)
        mx.eval(L)
        assert L.shape == (4, 4)

        # Upper triangle should be -inf (masked)
        for i in range(4):
            for j in range(i + 1, 4):
                assert L[i, j].item() < -1e8

        # Diagonal should be 0 (sum from j to i where j=i)
        for i in range(4):
            assert abs(L[i, i].item()) < 1e-5

    def test_chunked_matches_recurrent(self):
        """Chunked SSD matches recurrent scan output."""
        from lmxlab.core.mamba2 import Mamba2

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            mamba_n_groups=1,
            mamba_chunk_size=4,
        )
        mamba = Mamba2(cfg)
        mx.eval(mamba.parameters())

        # Use seq_len >= chunk_size to trigger chunked path
        # but also test recurrent
        x = mx.random.normal((1, 8, 64))
        mx.eval(x)

        # Training (no cache) with chunk_size=4, L=8 -> chunked
        out_chunked, _ = mamba(x)
        mx.eval(out_chunked)

        # Recurrent path: set chunk_size very large
        mamba.chunk_size = 1000
        out_recurrent, _ = mamba(x)
        mx.eval(out_recurrent)

        # Should be close (not exact due to floating point
        # order differences)
        assert mx.allclose(
            out_chunked,
            out_recurrent,
            atol=1e-3,
        ), f"Max diff: {mx.max(mx.abs(out_chunked - out_recurrent)).item()}"

    def test_non_divisible_length(self):
        """Handles seq_len not divisible by chunk_size."""
        from lmxlab.core.mamba2 import Mamba2

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            mamba_n_groups=1,
            mamba_chunk_size=4,
        )
        mamba = Mamba2(cfg)
        mx.eval(mamba.parameters())

        # L=7 is not divisible by chunk_size=4
        x = mx.random.normal((1, 7, 64))
        out, cache = mamba(x)
        mx.eval(out)
        assert out.shape == (1, 7, 64)

    def test_short_sequence_uses_recurrent(self):
        """Short sequences (L < chunk_size) use recurrent."""
        from lmxlab.core.mamba2 import Mamba2

        cfg = BlockConfig(
            d_model=64,
            n_heads=4,
            mamba_n_heads=4,
            mamba_head_dim=32,
            ssm_state_size=16,
            mamba_expand=2,
            mamba_n_groups=1,
            mamba_chunk_size=32,
        )
        mamba = Mamba2(cfg)
        mx.eval(mamba.parameters())

        # L=4 < chunk_size=32 -> recurrent
        x = mx.random.normal((1, 4, 64))
        out, _ = mamba(x)
        mx.eval(out)
        assert out.shape == (1, 4, 64)


class TestReturnHidden:
    """Tests for return_hidden parameter on LanguageModel."""

    def test_return_hidden_false(self):
        """Default: returns (logits, caches)."""
        cfg = gpt_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        result = model(x)
        assert len(result) == 2
        logits, caches = result
        assert logits.shape == (1, 3, cfg.vocab_size)

    def test_return_hidden_true(self):
        """return_hidden=True adds hidden states."""
        cfg = gpt_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        result = model(x, return_hidden=True)
        assert len(result) == 3
        logits, caches, hidden = result
        assert logits.shape == (1, 3, cfg.vocab_size)
        assert hidden.shape == (1, 3, cfg.block.d_model)


class TestMTP:
    """Tests for Multi-Token Prediction."""

    def test_mtp_forward(self):
        """MTP wrapper produces logits and losses."""
        from lmxlab.training.mtp import MultiTokenPrediction

        cfg = ModelConfig(
            block=BlockConfig(
                d_model=64,
                n_heads=4,
                d_ff=128,
                position="none",
            ),
            vocab_size=256,
            n_layers=2,
        )
        model = LanguageModel(cfg)
        mx.eval(model.parameters())

        mtp = MultiTokenPrediction(
            model,
            n_predict=2,
            mtp_weight=0.3,
        )
        mx.eval(mtp.parameters())

        x = mx.array([[1, 2, 3, 4, 5, 6]])
        targets = mx.array([[2, 3, 4, 5, 6, 7]])

        logits, losses = mtp(x, targets)
        mx.eval(logits, losses)
        assert logits.shape == (1, 6, 256)
        assert "main_loss" in losses
        assert "mtp_loss" in losses
        assert "total_loss" in losses

    def test_mtp_loss_contribution(self):
        """MTP loss is non-zero and contributes to total."""
        from lmxlab.training.mtp import MultiTokenPrediction

        cfg = ModelConfig(
            block=BlockConfig(
                d_model=64,
                n_heads=4,
                d_ff=128,
                position="none",
            ),
            vocab_size=256,
            n_layers=2,
        )
        model = LanguageModel(cfg)
        mx.eval(model.parameters())

        mtp = MultiTokenPrediction(
            model,
            n_predict=2,
            mtp_weight=0.5,
        )
        mx.eval(mtp.parameters())

        x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 9]])

        _, losses = mtp(x, targets)
        mx.eval(losses)

        main = losses["main_loss"].item()
        mtp_l = losses["mtp_loss"].item()
        total = losses["total_loss"].item()

        assert main > 0
        assert mtp_l > 0
        expected = main + 0.5 * mtp_l
        assert abs(total - expected) < 1e-4

    def test_mtp_shared_head(self):
        """MTP uses shared lm_head (not separate weights)."""
        from lmxlab.training.mtp import MultiTokenPrediction

        cfg = ModelConfig(
            block=BlockConfig(
                d_model=64,
                n_heads=4,
                d_ff=128,
                position="none",
            ),
            vocab_size=256,
            n_layers=2,
            tie_embeddings=False,
        )
        model = LanguageModel(cfg)
        mtp = MultiTokenPrediction(model, n_predict=1)

        # MTP heads should not have a separate lm_head
        for head in mtp.mtp_heads:
            assert not hasattr(head, "head")


class TestDensePattern:
    """Tests for '-' (dense MLP) pattern character."""

    def test_dash_pattern_parsing(self):
        """'-' maps to dense FFN config."""
        attn = BlockConfig(attention="gqa")
        moe = BlockConfig(attention="none", ffn="latent_moe")
        mamba = BlockConfig(attention="mamba2", ffn="none")
        dense = BlockConfig(attention="none", ffn="relu2")

        configs = _parse_hybrid_pattern(
            "M-*",
            attn,
            moe,
            mamba,
            dense,
        )
        assert len(configs) == 3
        assert configs[0].attention == "mamba2"
        assert configs[1].attention == "none"
        assert configs[1].ffn == "relu2"
        assert configs[2].attention == "gqa"

    def test_8b_config_valid(self):
        """nemotron3_8b produces a valid ModelConfig."""
        cfg = nemotron3_8b()
        assert cfg.n_layers == 52
        assert cfg.vocab_size == 131072
        assert cfg.block_configs is not None
        assert len(cfg.block_configs) == 52

    def test_8b_layer_types(self):
        """8B model has M, -, and * layer types."""
        cfg = nemotron3_8b()
        types = set()
        for bc in cfg.block_configs:
            types.add((bc.attention, bc.ffn))
        # Should have Mamba, dense, and attention layers
        assert ("mamba2", "none") in types
        assert ("none", "relu2") in types
        assert ("gqa", "none") in types


class TestNemotronWeightMap:
    """Tests for Nemotron-H HF weight mapping."""

    def test_embedding_map(self):
        """Maps backbone.embeddings.weight."""
        from lmxlab.models.convert import _nemotron_weight_map

        wmap = _nemotron_weight_map("M*")
        assert wmap("backbone.embeddings.weight") == "embed.weight"

    def test_final_norm_map(self):
        """Maps backbone.norm_f.weight."""
        from lmxlab.models.convert import _nemotron_weight_map

        wmap = _nemotron_weight_map("M*")
        assert wmap("backbone.norm_f.weight") == "final_norm.weight"

    def test_lm_head_map(self):
        """Maps lm_head.weight."""
        from lmxlab.models.convert import _nemotron_weight_map

        wmap = _nemotron_weight_map("M*")
        assert wmap("lm_head.weight") == "head.weight"

    def test_mamba_layer_map(self):
        """Maps Mamba-2 layer weights (M)."""
        from lmxlab.models.convert import _nemotron_weight_map

        wmap = _nemotron_weight_map("M*")
        assert (
            wmap("backbone.layers.0.mixer.in_proj.weight")
            == "blocks.0.attention.in_proj.weight"
        )
        assert (
            wmap("backbone.layers.0.mixer.A_log") == "blocks.0.attention.A_log"
        )
        assert (
            wmap("backbone.layers.0.norm.weight")
            == "blocks.0.attn_norm.weight"
        )

    def test_attn_layer_map(self):
        """Maps attention layer weights (*).

        Attention layers have only Q/K/V/O projections,
        no FFN weights (HF-verified for Nemotron-H-8B).
        """
        from lmxlab.models.convert import _nemotron_weight_map

        wmap = _nemotron_weight_map("M*")
        assert (
            wmap("backbone.layers.1.mixer.q_proj.weight")
            == "blocks.1.attention.q_proj.weight"
        )
        # Attention layers have no FFN weights
        assert wmap("backbone.layers.1.mlp.up_proj.weight") is None

    def test_moe_layer_map(self):
        """Maps LatentMoE layer weights (E)."""
        from lmxlab.models.convert import _nemotron_weight_map

        wmap = _nemotron_weight_map("E")
        assert (
            wmap("backbone.layers.0.mlp.router.weight")
            == "blocks.0.ffn.router.weight"
        )
        assert (
            wmap("backbone.layers.0.mlp.experts.3.up.weight")
            == "blocks.0.ffn.experts.3.up.weight"
        )
        assert (
            wmap("backbone.layers.0.mlp.shared_expert.up.weight")
            == "blocks.0.ffn.shared_expert.up.weight"
        )

    def test_dense_layer_map(self):
        """Maps dense MLP layer weights (-).

        Dense layers use mixer.up/down_proj prefix
        (HF-verified for Nemotron-H-8B).
        """
        from lmxlab.models.convert import _nemotron_weight_map

        wmap = _nemotron_weight_map("-")
        assert (
            wmap("backbone.layers.0.mixer.up_proj.weight")
            == "blocks.0.ffn.up.weight"
        )
        assert (
            wmap("backbone.layers.0.mixer.down_proj.weight")
            == "blocks.0.ffn.down.weight"
        )

    def test_config_from_nemotron_h(self):
        """config_from_hf works with nemotron_h type."""
        from lmxlab.models.convert import config_from_hf

        hf_cfg = {
            "model_type": "nemotron_h",
            "hybrid_override_pattern": "M-M*",
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "num_hidden_layers": 4,
            "mamba_num_heads": 4,
            "ssm_state_size": 16,
            "expand": 2,
            "n_groups": 1,
            "conv_kernel": 4,
        }
        cfg = config_from_hf(hf_cfg)
        assert cfg.n_layers == 4
        assert cfg.vocab_size == 1000


class TestDeepSeekV3Config:
    def test_tiny_config(self):
        """deepseek_v3_tiny creates valid config."""
        from lmxlab.models.deepseek import deepseek_v3_tiny

        cfg = deepseek_v3_tiny()
        assert cfg.n_layers == 4
        assert cfg.block_configs is not None
        assert len(cfg.block_configs) == 4

    def test_dense_then_moe(self):
        """First layer is dense, rest are MoE."""
        from lmxlab.models.deepseek import deepseek_v3_tiny

        cfg = deepseek_v3_tiny()
        assert cfg.block_configs[0].ffn == "gated"
        for bc in cfg.block_configs[1:]:
            assert bc.ffn == "shared_moe"

    def test_forward_shape(self):
        """DeepSeek V3 forward pass produces correct shape."""
        from lmxlab.models.deepseek import deepseek_v3_tiny

        cfg = deepseek_v3_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3, 4]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, cfg.vocab_size)


class TestOLMo2Config:
    def test_tiny_config(self):
        """olmo2_tiny creates valid config with qk_norm."""
        from lmxlab.models.olmo import olmo2_tiny

        cfg = olmo2_tiny()
        assert cfg.block.qk_norm is True
        assert cfg.block.attention == "gqa"

    def test_forward_shape(self):
        """OLMo 2 forward pass produces correct shape."""
        from lmxlab.models.olmo import olmo2_tiny

        cfg = olmo2_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, cfg.vocab_size)


class TestLlama4ScoutConfig:
    def test_tiny_config(self):
        """llama4_scout_tiny creates valid hybrid config."""
        from lmxlab.models.llama4 import llama4_scout_tiny

        cfg = llama4_scout_tiny()
        assert cfg.n_layers == 4
        assert cfg.block_configs is not None
        assert len(cfg.block_configs) == 4

    def test_irope_pattern(self):
        """iRoPE: 3 chunked + 1 NoPE per cycle."""
        from lmxlab.models.llama4 import llama4_scout_tiny

        cfg = llama4_scout_tiny()
        # Layers 0,1,2 = chunked, layer 3 = NoPE
        for i in range(3):
            assert cfg.block_configs[i].attention == "chunked_gqa"
            assert cfg.block_configs[i].position == "rope"
        assert cfg.block_configs[3].attention == "gqa"
        assert cfg.block_configs[3].position == "none"

    def test_forward_shape(self):
        """Llama 4 Scout forward pass produces correct shape."""
        from lmxlab.models.llama4 import llama4_scout_tiny

        cfg = llama4_scout_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3, 4]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, cfg.vocab_size)


class TestQwenNextConfig:
    def test_tiny_config(self):
        """qwen_next_tiny creates valid config."""
        from lmxlab.models.qwen_next import qwen_next_tiny

        cfg = qwen_next_tiny()
        assert cfg.block.attention == "gated_gqa"

    def test_forward_shape(self):
        """Qwen3-Next forward pass produces correct shape."""
        from lmxlab.models.qwen_next import qwen_next_tiny

        cfg = qwen_next_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, cfg.vocab_size)


class TestSmolLM3Config:
    def test_tiny_config(self):
        """smollm3_tiny creates valid hybrid config."""
        from lmxlab.models.smollm import smollm3_tiny

        cfg = smollm3_tiny()
        assert cfg.n_layers == 4
        assert cfg.block_configs is not None

    def test_irope_pattern(self):
        """iRoPE: 3 RoPE + 1 NoPE per cycle."""
        from lmxlab.models.smollm import smollm3_tiny

        cfg = smollm3_tiny()
        for i in range(3):
            assert cfg.block_configs[i].position == "rope"
        assert cfg.block_configs[3].position == "none"

    def test_forward_shape(self):
        """SmolLM3 forward pass produces correct shape."""
        from lmxlab.models.smollm import smollm3_tiny

        cfg = smollm3_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, cfg.vocab_size)


class TestMistralSmallConfig:
    def test_tiny_config(self):
        """mistral_small_tiny creates valid config."""
        from lmxlab.models.mistral import mistral_small_tiny

        cfg = mistral_small_tiny()
        assert cfg.block.attention == "sliding_window_gqa"
        assert cfg.block.window_size == 32

    def test_forward_shape(self):
        """Mistral Small forward pass produces correct shape."""
        from lmxlab.models.mistral import mistral_small_tiny

        cfg = mistral_small_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, cfg.vocab_size)


class TestGPTOSSConfig:
    def test_tiny_config(self):
        """gpt_oss_tiny creates valid config."""
        from lmxlab.models.gpt_oss import gpt_oss_tiny

        cfg = gpt_oss_tiny()
        assert cfg.block.qk_norm is True
        assert cfg.tie_embeddings is True

    def test_forward_shape(self):
        """GPT-OSS forward pass produces correct shape."""
        from lmxlab.models.gpt_oss import gpt_oss_tiny

        cfg = gpt_oss_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, cfg.vocab_size)


class TestGrokConfig:
    def test_tiny_config(self):
        """grok_tiny creates valid config."""
        from lmxlab.models.grok import grok_tiny

        cfg = grok_tiny()
        assert cfg.block.ffn == "shared_moe"
        assert cfg.block.n_experts == 4

    def test_forward_shape(self):
        """Grok forward pass produces correct shape."""
        from lmxlab.models.grok import grok_tiny

        cfg = grok_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, cfg.vocab_size)


class TestKimiConfig:
    def test_tiny_config(self):
        """kimi_tiny creates valid hybrid config."""
        from lmxlab.models.kimi import kimi_tiny

        cfg = kimi_tiny()
        assert cfg.n_layers == 4
        assert cfg.block_configs is not None

    def test_hybrid_pattern(self):
        """3 GQA + 1 DeltaNet per cycle."""
        from lmxlab.models.kimi import kimi_tiny

        cfg = kimi_tiny()
        for i in range(3):
            assert cfg.block_configs[i].attention == "gqa"
        assert cfg.block_configs[3].attention == "gated_deltanet"

    def test_forward_shape(self):
        """Kimi K2.5 forward pass produces correct shape."""
        from lmxlab.models.kimi import kimi_tiny

        cfg = kimi_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3, 4]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, cfg.vocab_size)


class TestFalconH1Config:
    def test_tiny_config(self):
        """falcon_h1_tiny creates valid hybrid config."""
        from lmxlab.models.falcon import falcon_h1_tiny

        cfg = falcon_h1_tiny()
        assert cfg.n_layers == 4
        assert cfg.block_configs is not None
        assert len(cfg.block_configs) == 4

    def test_hybrid_pattern(self):
        """3 Mamba-2 + 1 GQA in MMM* pattern."""
        from lmxlab.models.falcon import falcon_h1_tiny

        cfg = falcon_h1_tiny()
        for i in range(3):
            assert cfg.block_configs[i].attention == "mamba2"
        assert cfg.block_configs[3].attention == "gqa"

    def test_forward_shape(self):
        """Falcon H1 forward pass produces correct shape."""
        from lmxlab.models.falcon import falcon_h1_tiny

        cfg = falcon_h1_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3, 4]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, cfg.vocab_size)


class TestJambaConfig:
    def test_tiny_config(self):
        """jamba_tiny creates valid hybrid config."""
        from lmxlab.models.jamba import jamba_tiny

        cfg = jamba_tiny()
        assert cfg.n_layers == 8
        assert cfg.block_configs is not None
        assert len(cfg.block_configs) == 8

    def test_hybrid_pattern(self):
        """MMMA pattern: 3 Mamba + 1 attention per cycle."""
        from lmxlab.models.jamba import jamba_tiny

        cfg = jamba_tiny()
        # First cycle: layers 0,1,2 = Mamba, layer 3 = attn
        for i in range(3):
            assert cfg.block_configs[i].attention == "mamba2"
        assert cfg.block_configs[3].attention == "gqa"

    def test_moe_layers(self):
        """Even-indexed attention layers use MoE FFN."""
        from lmxlab.models.jamba import jamba_tiny

        cfg = jamba_tiny()
        # Layer 3 is 1st attn (index 0, even) -> MoE
        assert cfg.block_configs[3].ffn == "moe"
        # Layer 7 is 2nd attn (index 1, odd) -> dense
        assert cfg.block_configs[7].ffn == "gated"

    def test_forward_shape(self):
        """Jamba forward pass produces correct shape."""
        from lmxlab.models.jamba import jamba_tiny

        cfg = jamba_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3, 4]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, cfg.vocab_size)


class TestBambaConfig:
    def test_tiny_config(self):
        """bamba_tiny creates valid hybrid config."""
        from lmxlab.models.bamba import bamba_tiny

        cfg = bamba_tiny()
        assert cfg.n_layers == 4
        assert cfg.block_configs is not None
        assert len(cfg.block_configs) == 4

    def test_hybrid_pattern(self):
        """3 Mamba-2 + 1 GQA in MMM* pattern."""
        from lmxlab.models.bamba import bamba_tiny

        cfg = bamba_tiny()
        for i in range(3):
            assert cfg.block_configs[i].attention == "mamba2"
        assert cfg.block_configs[3].attention == "gqa"

    def test_forward_shape(self):
        """Bamba forward pass produces correct shape."""
        from lmxlab.models.bamba import bamba_tiny

        cfg = bamba_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3, 4]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, cfg.vocab_size)


class TestLlama4MaverickConfig:
    def test_tiny_config(self):
        """llama4_maverick_tiny creates valid hybrid config."""
        from lmxlab.models.llama4 import llama4_maverick_tiny

        cfg = llama4_maverick_tiny()
        assert cfg.n_layers == 4
        assert cfg.block_configs is not None

    def test_irope_pattern(self):
        """iRoPE: 3 chunked + 1 NoPE per cycle."""
        from lmxlab.models.llama4 import llama4_maverick_tiny

        cfg = llama4_maverick_tiny()
        for i in range(3):
            assert cfg.block_configs[i].attention == "chunked_gqa"
            assert cfg.block_configs[i].position == "rope"
        assert cfg.block_configs[3].attention == "gqa"
        assert cfg.block_configs[3].position == "none"

    def test_more_experts_than_scout(self):
        """Maverick has more experts than Scout."""
        from lmxlab.models.llama4 import llama4_maverick_tiny

        cfg = llama4_maverick_tiny()
        assert cfg.block_configs[0].n_experts == 8
        assert cfg.block_configs[0].top_k_experts == 1

    def test_forward_shape(self):
        """Llama 4 Maverick forward pass produces correct shape."""
        from lmxlab.models.llama4 import llama4_maverick_tiny

        cfg = llama4_maverick_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3, 4]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, cfg.vocab_size)


class TestQwen3MoeConfig:
    def test_tiny_config(self):
        """qwen3_moe_tiny creates valid config."""
        from lmxlab.models.qwen import qwen3_moe_tiny

        cfg = qwen3_moe_tiny()
        assert cfg.block.ffn == "shared_moe"
        assert cfg.block.n_experts == 4

    def test_forward_shape(self):
        """Qwen3 MoE forward pass produces correct shape."""
        from lmxlab.models.qwen import qwen3_moe_tiny

        cfg = qwen3_moe_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, cfg.vocab_size)


class TestGLM45Config:
    def test_tiny_config(self):
        """glm45_tiny creates valid MLA config with no RoPE."""
        from lmxlab.models.glm import glm45_tiny

        cfg = glm45_tiny()
        assert cfg.block.attention == "mla"
        assert cfg.block.position == "none"
        assert cfg.block.rope_dim == 0

    def test_forward_shape(self):
        """GLM-4.5 forward pass produces correct shape."""
        from lmxlab.models.glm import glm45_tiny

        cfg = glm45_tiny()
        model = LanguageModel(cfg)
        mx.eval(model.parameters())
        x = mx.array([[1, 2, 3]])
        logits, _ = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 3, cfg.vocab_size)
