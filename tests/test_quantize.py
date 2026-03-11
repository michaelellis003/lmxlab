"""Tests for model quantization."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from lmt_metal.core.quantize import (
    dequantize_model,
    quantize_model,
)
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.models.llama import llama_tiny


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


def _model_size(model) -> int:
    """Total bytes of all model parameters."""
    leaves = mlx.utils.tree_flatten(model.parameters())
    total = 0
    for _, p in leaves:
        total += p.nbytes
    return total
