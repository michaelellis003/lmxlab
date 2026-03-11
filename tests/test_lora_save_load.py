"""Tests for LoRA adapter save/load."""

import json

import mlx.core as mx
import pytest

from lmt_metal.core.lora import (
    apply_lora,
    load_lora_adapters,
    save_lora_adapters,
)
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.llama import llama_config


@pytest.fixture()
def tiny_model():
    """Build a tiny LLaMA model with LoRA applied."""
    config = llama_config(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        n_layers=2,
        d_ff=32,
        max_seq_len=16,
    )
    model = LanguageModel(config)
    mx.eval(model.parameters())
    apply_lora(model, rank=4, alpha=1.0, targets=["attention"])
    mx.eval(model.parameters())
    return model


@pytest.fixture()
def save_dir(tmp_path):
    """Provide a temporary save directory."""
    return tmp_path / "lora_adapters"


class TestSaveLoraAdapters:
    """Tests for save_lora_adapters."""

    def test_creates_safetensors_file(self, tiny_model, save_dir):
        save_lora_adapters(save_dir, tiny_model)
        assert (save_dir / "adapter.safetensors").exists()

    def test_creates_metadata_file(self, tiny_model, save_dir):
        save_lora_adapters(save_dir, tiny_model)
        assert (save_dir / "adapter_config.json").exists()

    def test_metadata_contains_rank(self, tiny_model, save_dir):
        save_lora_adapters(save_dir, tiny_model, rank=4, alpha=1.0)
        meta = json.loads((save_dir / "adapter_config.json").read_text())
        assert meta["rank"] == 4
        assert meta["alpha"] == 1.0

    def test_metadata_custom_fields(self, tiny_model, save_dir):
        save_lora_adapters(
            save_dir, tiny_model, metadata={"base_model": "test"}
        )
        meta = json.loads((save_dir / "adapter_config.json").read_text())
        assert meta["base_model"] == "test"

    def test_saves_only_lora_weights(self, tiny_model, save_dir):
        save_lora_adapters(save_dir, tiny_model)
        weights = mx.load(str(save_dir / "adapter.safetensors"))
        # Every key should contain lora_A or lora_B
        for key in weights:
            assert "lora_A" in key or "lora_B" in key, (
                f"Non-LoRA key saved: {key}"
            )

    def test_saved_file_is_small(self, tiny_model, save_dir):
        save_lora_adapters(save_dir, tiny_model)
        adapter_size = (save_dir / "adapter.safetensors").stat().st_size
        # LoRA adapters should be small (< 100KB for tiny model)
        assert adapter_size < 100_000

    def test_creates_directory(self, tiny_model, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        save_lora_adapters(nested, tiny_model)
        assert (nested / "adapter.safetensors").exists()


class TestLoadLoraAdapters:
    """Tests for load_lora_adapters."""

    def test_roundtrip(self, tiny_model, save_dir):
        """Save then load produces identical LoRA weights."""
        # Get original LoRA weights
        original_params = {}
        for k, v in mx.utils.tree_flatten(tiny_model.parameters()):
            if "lora_A" in k or "lora_B" in k:
                original_params[k] = v

        save_lora_adapters(save_dir, tiny_model)

        # Build a fresh model with LoRA
        config = llama_config(
            vocab_size=32,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            n_layers=2,
            d_ff=32,
            max_seq_len=16,
        )
        new_model = LanguageModel(config)
        mx.eval(new_model.parameters())
        apply_lora(new_model, rank=4, alpha=1.0, targets=["attention"])
        mx.eval(new_model.parameters())

        load_lora_adapters(save_dir, new_model)

        # Verify LoRA weights match
        for k, v in mx.utils.tree_flatten(new_model.parameters()):
            if k in original_params:
                assert mx.allclose(v, original_params[k]), f"Mismatch at {k}"

    def test_output_matches_after_load(self, tiny_model, save_dir):
        """Model output is identical after save/load cycle."""
        tokens = mx.array([[1, 2, 3, 4]])
        original_logits, _ = tiny_model(tokens)
        mx.eval(original_logits)

        save_lora_adapters(save_dir, tiny_model)

        # Fresh model with LoRA
        config = llama_config(
            vocab_size=32,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            n_layers=2,
            d_ff=32,
            max_seq_len=16,
        )
        new_model = LanguageModel(config)
        # Load base weights from original
        base_weights = dict(mx.utils.tree_flatten(tiny_model.parameters()))
        # Filter out LoRA keys for base
        base_only = {
            k: v
            for k, v in base_weights.items()
            if "lora_A" not in k and "lora_B" not in k
        }
        new_model.load_weights(list(base_only.items()))
        mx.eval(new_model.parameters())

        apply_lora(new_model, rank=4, alpha=1.0, targets=["attention"])
        mx.eval(new_model.parameters())

        load_lora_adapters(save_dir, new_model)

        new_logits, _ = new_model(tokens)
        mx.eval(new_logits)
        assert mx.allclose(original_logits, new_logits, atol=1e-5)

    def test_returns_metadata(self, tiny_model, save_dir):
        save_lora_adapters(
            save_dir,
            tiny_model,
            rank=4,
            alpha=1.0,
            metadata={"base_model": "test"},
        )
        meta = load_lora_adapters(save_dir, tiny_model)
        assert meta["rank"] == 4
        assert meta["base_model"] == "test"

    def test_missing_dir_raises(self, tiny_model, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lora_adapters(tmp_path / "nonexistent", tiny_model)
