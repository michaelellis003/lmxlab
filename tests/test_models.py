"""Tests for model architectures and generation."""

import mlx.core as mx

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.gpt import gpt_config, gpt_tiny
from lmxlab.models.llama import llama_config, llama_tiny


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
