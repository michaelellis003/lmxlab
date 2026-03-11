"""Tests for HuggingFace weight conversion."""

import mlx.core as mx
import pytest

from lmt_metal.models.base import LanguageModel
from lmt_metal.models.convert import (
    WEIGHT_MAPS,
    config_from_hf,
    convert_weights,
)
from lmt_metal.models.llama import llama_config


class TestWeightMapLlama:
    """Test LLaMA HF-to-lmt-metal weight name mapping."""

    def test_embed_mapping(self):
        wmap = WEIGHT_MAPS["llama"]
        assert wmap("model.embed_tokens.weight") == "embed.weight"

    def test_attention_projections(self):
        wmap = WEIGHT_MAPS["llama"]
        assert (
            wmap("model.layers.0.self_attn.q_proj.weight")
            == "blocks.0.attention.q_proj.weight"
        )
        assert (
            wmap("model.layers.5.self_attn.k_proj.weight")
            == "blocks.5.attention.k_proj.weight"
        )
        assert (
            wmap("model.layers.31.self_attn.v_proj.weight")
            == "blocks.31.attention.v_proj.weight"
        )
        assert (
            wmap("model.layers.0.self_attn.o_proj.weight")
            == "blocks.0.attention.o_proj.weight"
        )

    def test_ffn_projections(self):
        wmap = WEIGHT_MAPS["llama"]
        assert (
            wmap("model.layers.0.mlp.gate_proj.weight")
            == "blocks.0.ffn.gate.weight"
        )
        assert (
            wmap("model.layers.0.mlp.up_proj.weight")
            == "blocks.0.ffn.up.weight"
        )
        assert (
            wmap("model.layers.0.mlp.down_proj.weight")
            == "blocks.0.ffn.down.weight"
        )

    def test_norm_mapping(self):
        wmap = WEIGHT_MAPS["llama"]
        assert (
            wmap("model.layers.0.input_layernorm.weight")
            == "blocks.0.attn_norm.weight"
        )
        assert (
            wmap("model.layers.0.post_attention_layernorm.weight")
            == "blocks.0.ffn_norm.weight"
        )

    def test_final_norm(self):
        wmap = WEIGHT_MAPS["llama"]
        assert wmap("model.norm.weight") == "final_norm.weight"

    def test_lm_head(self):
        wmap = WEIGHT_MAPS["llama"]
        assert wmap("lm_head.weight") == "head.weight"

    def test_unknown_key_returns_none(self):
        wmap = WEIGHT_MAPS["llama"]
        assert wmap("model.layers.0.self_attn.rotary_emb.inv_freq") is None

    def test_multiple_layer_indices(self):
        wmap = WEIGHT_MAPS["llama"]
        for i in [0, 7, 15, 31]:
            result = wmap(f"model.layers.{i}.self_attn.q_proj.weight")
            assert result == f"blocks.{i}.attention.q_proj.weight"


class TestConvertWeights:
    """Test bulk weight conversion."""

    def test_converts_known_keys(self):
        hf_weights = {
            "model.embed_tokens.weight": mx.ones((10, 4)),
            "model.norm.weight": mx.ones((4,)),
        }
        converted = convert_weights(hf_weights, "llama")
        assert "embed.weight" in converted
        assert "final_norm.weight" in converted
        assert len(converted) == 2

    def test_skips_unknown_keys(self):
        hf_weights = {
            "model.embed_tokens.weight": mx.ones((10, 4)),
            "model.layers.0.self_attn.rotary_emb.inv_freq": mx.ones((8,)),
        }
        converted = convert_weights(hf_weights, "llama")
        assert len(converted) == 1
        assert "embed.weight" in converted

    def test_unknown_arch_raises(self):
        with pytest.raises(KeyError, match="nosuch"):
            convert_weights({}, "nosuch")

    def test_preserves_array_values(self):
        arr = mx.array([1.0, 2.0, 3.0])
        hf_weights = {"model.norm.weight": arr}
        converted = convert_weights(hf_weights, "llama")
        assert mx.array_equal(converted["final_norm.weight"], arr)


class TestConfigFromHf:
    """Test ModelConfig extraction from HF config dict."""

    def test_llama_config_basic(self):
        hf_config = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
        config = config_from_hf(hf_config)
        assert config.vocab_size == 32000
        assert config.n_layers == 32
        assert config.block.d_model == 4096
        assert config.block.n_heads == 32
        assert config.block.n_kv_heads == 8
        assert config.block.d_ff == 11008
        assert config.block.attention == "gqa"
        assert config.block.ffn == "gated"
        assert config.block.norm == "rms_norm"
        assert config.block.position == "rope"
        assert config.tie_embeddings is False

    def test_llama_config_defaults(self):
        """Minimal config with defaults."""
        hf_config = {
            "model_type": "llama",
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
        }
        config = config_from_hf(hf_config)
        assert config.vocab_size == 1000
        assert config.block.n_kv_heads == 4  # defaults to n_heads
        assert config.block.rope_theta == 10000.0

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            config_from_hf({"model_type": "bert"})

    def test_missing_required_keys_raises(self):
        """Missing required keys give a clear ValueError."""
        hf_config = {
            "model_type": "llama",
            "vocab_size": 1000,
            # missing: hidden_size, num_attention_heads,
            #          intermediate_size, num_hidden_layers
        }
        with pytest.raises(ValueError, match="missing required"):
            config_from_hf(hf_config)

    def test_gemma_config(self):
        hf_config = {
            "model_type": "gemma",
            "vocab_size": 256000,
            "hidden_size": 2048,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_hidden_layers": 18,
            "intermediate_size": 16384,
        }
        config = config_from_hf(hf_config)
        assert config.block.attention == "gqa"
        assert config.block.norm == "rms_norm"


class TestLoadIntoModel:
    """Test loading converted weights into a LanguageModel."""

    def test_round_trip_tiny_llama(self):
        """Save model weights, rename to HF format, convert back."""
        import mlx.utils

        config = llama_config(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            d_ff=64,
            max_seq_len=32,
            tie_embeddings=False,
        )
        model = LanguageModel(config)
        mx.eval(model.parameters())

        # Get lmt-metal weights
        lmt_weights = dict(mlx.utils.tree_flatten(model.parameters()))

        # Create fake HF weights by reversing the mapping
        hf_weights = {}
        reverse_map = {}
        for lmt_name, arr in lmt_weights.items():
            # Find the HF name that maps to this lmt name
            hf_name = _reverse_llama_name(lmt_name)
            if hf_name:
                hf_weights[hf_name] = arr
                reverse_map[hf_name] = lmt_name

        # Convert back
        converted = convert_weights(hf_weights, "llama")

        # All original weights should be present
        for name in lmt_weights:
            assert name in converted, f"Missing: {name}"
            assert mx.array_equal(converted[name], lmt_weights[name]), (
                f"Mismatch: {name}"
            )


def _reverse_llama_name(lmt_name: str) -> str | None:
    """Reverse map lmt-metal name to HF LLaMA name (test helper)."""
    mappings = {
        "embed.weight": "model.embed_tokens.weight",
        "final_norm.weight": "model.norm.weight",
        "head.weight": "lm_head.weight",
    }
    if lmt_name in mappings:
        return mappings[lmt_name]

    # blocks.{i}.attention.{proj}.weight ->
    #   model.layers.{i}.self_attn.{proj}.weight
    if lmt_name.startswith("blocks."):
        parts = lmt_name.split(".")
        idx = parts[1]
        if parts[2] == "attention":
            proj = parts[3]  # q_proj, k_proj, etc.
            param = parts[4]  # weight
            return f"model.layers.{idx}.self_attn.{proj}.{param}"
        elif parts[2] == "ffn":
            proj = parts[3]  # gate, up, down
            param = parts[4]  # weight
            ffn_map = {
                "gate": "gate_proj",
                "up": "up_proj",
                "down": "down_proj",
            }
            return f"model.layers.{idx}.mlp.{ffn_map[proj]}.{param}"
        elif parts[2] == "attn_norm":
            return f"model.layers.{idx}.input_layernorm.{parts[3]}"
        elif parts[2] == "ffn_norm":
            return f"model.layers.{idx}.post_attention_layernorm.{parts[3]}"
    return None
