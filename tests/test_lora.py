"""Tests for LoRA (Low-Rank Adaptation)."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from lmt_metal.core.lora import (
    LoRALinear,
    apply_lora,
    lora_parameters,
    merge_lora,
)
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.llama import llama_tiny


class TestLoRALinear:
    """Test LoRALinear module."""

    def test_output_shape(self):
        linear = LoRALinear.from_linear(nn.Linear(64, 64, bias=False), rank=4)
        x = mx.random.normal((2, 8, 64))
        y = linear(x)
        assert y.shape == (2, 8, 64)

    def test_with_bias(self):
        linear = LoRALinear.from_linear(nn.Linear(64, 64, bias=True), rank=4)
        x = mx.random.normal((2, 8, 64))
        y = linear(x)
        assert y.shape == (2, 8, 64)

    def test_rank_constraint(self):
        linear = LoRALinear.from_linear(nn.Linear(64, 32, bias=False), rank=8)
        # lora_A: (64, 8), lora_B: (8, 32)
        assert linear.lora_A.shape == (64, 8)
        assert linear.lora_B.shape == (8, 32)

    def test_initial_output_matches_base(self):
        """LoRA B is zero-initialized, so initial output = base output."""
        base = nn.Linear(64, 64, bias=False)
        mx.eval(base.parameters())
        lora = LoRALinear.from_linear(base, rank=4)

        x = mx.random.normal((1, 4, 64))
        mx.eval(x)

        base_out = base(x)
        lora_out = lora(x)
        mx.eval(base_out, lora_out)

        assert mx.allclose(base_out, lora_out, atol=1e-5).item()

    def test_scaling(self):
        linear = LoRALinear.from_linear(
            nn.Linear(64, 64, bias=False),
            rank=4,
            alpha=8.0,
        )
        # scaling = alpha / rank = 8 / 4 = 2.0
        assert linear.scaling == 2.0

    def test_only_lora_params_trainable(self):
        linear = LoRALinear.from_linear(nn.Linear(64, 64, bias=False), rank=4)
        trainable = linear.trainable_parameters()
        flat = dict(mlx.utils.tree_flatten(trainable))
        # Only lora_A and lora_B should be trainable
        assert "lora_A" in flat
        assert "lora_B" in flat
        assert "weight" not in flat


class TestApplyLora:
    """Test applying LoRA to a model."""

    def test_apply_to_attention(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        apply_lora(model, rank=4, targets=["attention"])

        # Attention projections should be LoRALinear
        block = model.blocks[0]
        assert isinstance(block.attention.q_proj, LoRALinear)
        assert isinstance(block.attention.k_proj, LoRALinear)
        assert isinstance(block.attention.v_proj, LoRALinear)
        assert isinstance(block.attention.o_proj, LoRALinear)

        # FFN should remain nn.Linear
        assert isinstance(block.ffn.gate, nn.Linear)

    def test_apply_to_ffn(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        apply_lora(model, rank=4, targets=["ffn"])

        block = model.blocks[0]
        assert isinstance(block.ffn.gate, LoRALinear)
        assert isinstance(block.ffn.up, LoRALinear)
        assert isinstance(block.ffn.down, LoRALinear)

        # Attention should remain nn.Linear
        assert isinstance(block.attention.q_proj, nn.Linear)

    def test_apply_to_all(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        apply_lora(model, rank=4, targets=["attention", "ffn"])

        block = model.blocks[0]
        assert isinstance(block.attention.q_proj, LoRALinear)
        assert isinstance(block.ffn.gate, LoRALinear)

    def test_forward_after_lora(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        apply_lora(model, rank=4, targets=["attention"])

        tokens = mx.array([[1, 2, 3, 4]])
        logits, caches = model(tokens)
        mx.eval(logits)

        assert logits.shape == (1, 4, config.vocab_size)

    def test_trainable_count_much_smaller(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        total_before = _count_params(model.parameters())

        apply_lora(model, rank=4, targets=["attention"])

        trainable = _count_params(model.trainable_parameters())

        # LoRA trainable params should be << total params
        assert trainable < total_before * 0.2


class TestLoraParameters:
    """Test extracting LoRA-only parameters."""

    def test_lora_parameters_only(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        apply_lora(model, rank=4, targets=["attention"])

        lora_params = lora_parameters(model)
        flat = dict(mlx.utils.tree_flatten(lora_params))

        # Should only have lora_A and lora_B keys
        for key in flat:
            assert "lora_A" in key or "lora_B" in key


class TestMergeLora:
    """Test merging LoRA weights back into base model."""

    def test_merge_removes_lora(self):
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        apply_lora(model, rank=4, targets=["attention"])
        assert isinstance(model.blocks[0].attention.q_proj, LoRALinear)

        merge_lora(model)
        assert isinstance(model.blocks[0].attention.q_proj, nn.Linear)

    def test_merge_preserves_output(self):
        """Output should be identical before and after merge."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        apply_lora(model, rank=4, targets=["attention"])

        # Set non-zero LoRA weights for meaningful test
        for block in model.blocks:
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                layer = getattr(block.attention, name)
                layer.lora_A = mx.random.normal(layer.lora_A.shape) * 0.01
                layer.lora_B = mx.random.normal(layer.lora_B.shape) * 0.01
        mx.eval(model.parameters())

        tokens = mx.array([[1, 2, 3, 4, 5]])
        logits_before, _ = model(tokens)
        mx.eval(logits_before)

        merge_lora(model)
        logits_after, _ = model(tokens)
        mx.eval(logits_after)

        assert mx.allclose(logits_before, logits_after, atol=1e-4).item()


def _count_params(params) -> int:
    flat = mlx.utils.tree_flatten(params)
    return sum(p.size for _, p in flat)
