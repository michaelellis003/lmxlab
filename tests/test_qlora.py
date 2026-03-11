"""Tests for QLoRA (Quantized LoRA)."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from lmxlab.core.qlora import LoRAQuantizedLinear, apply_qlora
from lmxlab.core.quantize import quantize_model
from lmxlab.models.base import LanguageModel
from lmxlab.models.llama import llama_tiny


class TestLoRAQuantizedLinear:
    """Test LoRAQuantizedLinear module."""

    def test_output_shape(self):
        base = nn.Linear(64, 32, bias=False)
        mx.eval(base.parameters())
        ql = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)
        lora_ql = LoRAQuantizedLinear.from_quantized(ql, rank=4)

        x = mx.random.normal((2, 8, 64))
        y = lora_ql(x)
        mx.eval(y)
        assert y.shape == (2, 8, 32)

    def test_with_bias(self):
        base = nn.Linear(64, 32, bias=True)
        mx.eval(base.parameters())
        ql = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)
        lora_ql = LoRAQuantizedLinear.from_quantized(ql, rank=4)

        x = mx.random.normal((2, 8, 64))
        y = lora_ql(x)
        mx.eval(y)
        assert y.shape == (2, 8, 32)

    def test_lora_shapes(self):
        base = nn.Linear(128, 64, bias=False)
        mx.eval(base.parameters())
        ql = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)
        lora_ql = LoRAQuantizedLinear.from_quantized(ql, rank=8)

        assert lora_ql.lora_A.shape == (128, 8)
        assert lora_ql.lora_B.shape == (8, 64)

    def test_initial_output_matches_quantized(self):
        """B is zero-initialized, so initial output = quantized output."""
        base = nn.Linear(64, 64, bias=False)
        mx.eval(base.parameters())
        ql = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)

        lora_ql = LoRAQuantizedLinear.from_quantized(ql, rank=4)

        x = mx.random.normal((1, 4, 64))
        mx.eval(x)

        q_out = ql(x)
        lora_out = lora_ql(x)
        mx.eval(q_out, lora_out)

        assert mx.allclose(q_out, lora_out, atol=1e-5).item()

    def test_only_lora_trainable(self):
        base = nn.Linear(64, 64, bias=False)
        mx.eval(base.parameters())
        ql = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)
        lora_ql = LoRAQuantizedLinear.from_quantized(ql, rank=4)

        trainable = lora_ql.trainable_parameters()
        flat = dict(mlx.utils.tree_flatten(trainable))

        assert "lora_A" in flat
        assert "lora_B" in flat
        assert "weight" not in flat
        assert "scales" not in flat

    def test_scaling(self):
        base = nn.Linear(64, 64, bias=False)
        mx.eval(base.parameters())
        ql = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)
        lora_ql = LoRAQuantizedLinear.from_quantized(ql, rank=4, alpha=8.0)
        assert lora_ql.scaling == 2.0

    def test_quantized_weight_preserved(self):
        """Base weight stays quantized (uint32), not dequantized."""
        base = nn.Linear(64, 64, bias=False)
        mx.eval(base.parameters())
        ql = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)
        lora_ql = LoRAQuantizedLinear.from_quantized(ql, rank=4)

        assert lora_ql.weight.dtype == mx.uint32


class TestApplyQLoRA:
    """Test applying QLoRA to a quantized model."""

    def test_apply_to_quantized_model(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4)
        apply_qlora(model, rank=4, targets=["attention"])

        block = model.blocks[0]
        assert isinstance(block.attention.q_proj, LoRAQuantizedLinear)
        assert isinstance(block.attention.k_proj, LoRAQuantizedLinear)

    def test_ffn_untouched_when_not_targeted(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4)
        apply_qlora(model, rank=4, targets=["attention"])

        block = model.blocks[0]
        assert isinstance(block.ffn.gate, nn.QuantizedLinear)

    def test_forward_after_qlora(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4)
        apply_qlora(model, rank=4, targets=["attention"])

        tokens = mx.array([[1, 2, 3, 4]])
        logits, caches = model(tokens)
        mx.eval(logits)

        assert logits.shape == (1, 4, config.vocab_size)

    def test_trainable_count_small(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        total_before = _count_params(model.parameters())

        quantize_model(model, bits=4)
        apply_qlora(model, rank=4, targets=["attention"])

        trainable = _count_params(model.trainable_parameters())

        # QLoRA trainable params should be << total params
        assert trainable < total_before * 0.2

    def test_lora_params_only_lora(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4)
        apply_qlora(model, rank=4, targets=["attention"])

        trainable = model.trainable_parameters()
        flat = dict(mlx.utils.tree_flatten(trainable))

        for key in flat:
            assert "lora_A" in key or "lora_B" in key

    def test_apply_to_ffn(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        quantize_model(model, bits=4)
        apply_qlora(model, rank=4, targets=["ffn"])

        block = model.blocks[0]
        assert isinstance(block.ffn.gate, LoRAQuantizedLinear)
        assert isinstance(block.attention.q_proj, nn.QuantizedLinear)


def _count_params(params) -> int:
    flat = mlx.utils.tree_flatten(params)
    return sum(p.size for _, p in flat)
