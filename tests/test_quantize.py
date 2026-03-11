"""Tests for model quantization."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from lmxlab.core.quantize import (
    dequantize_model,
    quantize_model,
)
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.llama import llama_tiny


class TestQuantizeModel:
    """Test post-training quantization."""

    def test_quantize_reduces_weight_size(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        original_size = _model_size(model)
        quantize_model(model, bits=4, group_size=32)
        quantized_size = _model_size(model)

        # 4-bit should be roughly 8x smaller than float32
        # (with some overhead from scales/biases)
        assert quantized_size < original_size * 0.5

    def test_quantize_preserves_output_shape(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        tokens = mx.array([[1, 2, 3, 4]])
        logits_before, _ = model(tokens)
        mx.eval(logits_before)
        shape_before = logits_before.shape

        quantize_model(model, bits=4, group_size=32)
        logits_after, _ = model(tokens)
        mx.eval(logits_after)

        assert logits_after.shape == shape_before

    def test_quantize_4bit_default(self):
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4)

        # Check that linear layers are now QuantizedLinear
        block = model.blocks[0]
        assert isinstance(block.attention.q_proj, nn.QuantizedLinear)
        assert isinstance(block.ffn.up, nn.QuantizedLinear)

    def test_quantize_8bit(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=8, group_size=32)

        block = model.blocks[0]
        assert isinstance(block.attention.q_proj, nn.QuantizedLinear)

    def test_quantize_custom_group_size(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=64)

        block = model.blocks[0]
        assert isinstance(block.attention.q_proj, nn.QuantizedLinear)

    def test_quantize_skip_norm(self):
        """Norm layers should not be quantized."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)

        # Norms should stay as-is
        assert isinstance(model.blocks[0].attn_norm, nn.RMSNorm)
        assert isinstance(model.final_norm, nn.RMSNorm)

    def test_quantize_forward_runs(self):
        """Full forward pass works after quantization."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)

        tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, caches = model(tokens)
        mx.eval(logits)

        assert logits.shape == (1, 8, config.vocab_size)
        assert len(caches) == config.n_layers

    def test_8bit_reduces_size(self):
        """8-bit quantization also reduces model size."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        original = _model_size(model)
        quantize_model(model, bits=8, group_size=32)
        quantized = _model_size(model)

        # 8-bit should be smaller than float32 but larger than 4-bit
        assert quantized < original * 0.7

    def test_8bit_forward_runs(self):
        """Forward pass works with 8-bit quantization."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=8, group_size=32)

        tokens = mx.array([[1, 2, 3]])
        logits, caches = model(tokens)
        mx.eval(logits)
        assert logits.shape == (1, 3, config.vocab_size)

    def test_4bit_smaller_than_8bit(self):
        """4-bit model should be smaller than 8-bit."""
        config = llama_tiny()

        model4 = LanguageModel(config)
        mx.eval(model4.parameters())
        quantize_model(model4, bits=4, group_size=32)
        size4 = _model_size(model4)

        model8 = LanguageModel(config)
        mx.eval(model8.parameters())
        quantize_model(model8, bits=8, group_size=32)
        size8 = _model_size(model8)

        assert size4 < size8

    def test_quantize_gpt_no_tied_embeddings(self):
        """Quantization works on GPT with untied embeddings."""
        from lmxlab.models.gpt import gpt_config

        config = gpt_config(
            vocab_size=256,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            max_seq_len=128,
            tie_embeddings=False,
        )
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)

        tokens = mx.array([[1, 2, 3, 4]])
        logits, _ = model(tokens)
        mx.eval(logits)
        assert logits.shape == (1, 4, config.vocab_size)

    def test_quantize_generation(self):
        """Can generate tokens after quantization."""
        from lmxlab.models.generate import generate

        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)

        prompt = mx.array([[1, 2, 3]])
        result = generate(model, prompt, max_tokens=5, temperature=0.0)
        mx.eval(result)
        assert result.shape == (1, 8)

    def test_quantize_skip_embedding(self):
        """Embedding layer is quantized to QuantizedEmbedding."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)

        # MLX quantizes embeddings too
        assert isinstance(model.embed, nn.QuantizedEmbedding)

    def test_quantize_all_linear_layers(self):
        """All linear layers in blocks are quantized."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)

        for block in model.blocks:
            attn = block.attention
            assert isinstance(attn.q_proj, nn.QuantizedLinear)
            assert isinstance(attn.k_proj, nn.QuantizedLinear)
            assert isinstance(attn.v_proj, nn.QuantizedLinear)
            assert isinstance(attn.o_proj, nn.QuantizedLinear)


class TestDequantizeModel:
    """Test dequantization back to float."""

    def test_dequantize_restores_linear(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)
        assert isinstance(
            model.blocks[0].attention.q_proj,
            nn.QuantizedLinear,
        )

        dequantize_model(model)
        assert isinstance(model.blocks[0].attention.q_proj, nn.Linear)

    def test_dequantize_forward_runs(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)
        dequantize_model(model)

        tokens = mx.array([[1, 2, 3]])
        logits, _ = model(tokens)
        mx.eval(logits)
        assert logits.shape == (1, 3, config.vocab_size)

    def test_dequantize_restores_embedding(self):
        """Dequantize restores QuantizedEmbedding to Embedding."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)
        assert isinstance(model.embed, nn.QuantizedEmbedding)

        dequantize_model(model)
        assert isinstance(model.embed, nn.Embedding)

    def test_dequantize_all_layers(self):
        """All layers restored after dequantize."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)
        dequantize_model(model)

        for block in model.blocks:
            attn = block.attention
            assert isinstance(attn.q_proj, nn.Linear)
            assert isinstance(attn.k_proj, nn.Linear)
            assert isinstance(attn.v_proj, nn.Linear)
            assert isinstance(attn.o_proj, nn.Linear)

    def test_roundtrip_output_similar(self):
        """Quantize→dequantize output is close to original."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        tokens = mx.array([[1, 2, 3, 4]])
        logits_orig, _ = model(tokens)
        mx.eval(logits_orig)

        quantize_model(model, bits=8, group_size=32)
        dequantize_model(model)

        logits_roundtrip, _ = model(tokens)
        mx.eval(logits_roundtrip)

        # 8-bit roundtrip should be fairly close
        diff = mx.abs(logits_orig - logits_roundtrip)
        mx.eval(diff)
        assert mx.mean(diff).item() < 1.0

    def test_dequantize_generation(self):
        """Can generate after quantize→dequantize roundtrip."""
        from lmxlab.models.generate import generate

        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4, group_size=32)
        dequantize_model(model)

        prompt = mx.array([[1, 2]])
        result = generate(model, prompt, max_tokens=3, temperature=0.0)
        mx.eval(result)
        assert result.shape == (1, 5)


def _model_size(model) -> int:
    """Total bytes of all model parameters."""
    leaves = mlx.utils.tree_flatten(model.parameters())
    total = 0
    for _, p in leaves:
        total += p.nbytes
    return total
