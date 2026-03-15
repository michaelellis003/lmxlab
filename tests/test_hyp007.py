"""Recipe infrastructure tests for HYP-007."""

import sys
from pathlib import Path

import mlx.core as mx

# Add recipes to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "recipes"))

from hyp007_test_time_compute import (
    DROPOUT_RATES,
    K_VALUES,
    SEEDS,
    build_grid,
    evaluate_pass_at_k_modular,
    make_config,
)

from lmxlab.data.batching import batch_iterator
from lmxlab.data.modular_arithmetic import ModularArithmeticDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.models.base import LanguageModel


class TestHYP007Grid:
    """Test grid definition and config construction."""

    def test_grid_size(self):
        """9 entries (3 dropout x 3 seeds)."""
        grid = build_grid()
        assert len(grid) == len(DROPOUT_RATES) * len(SEEDS)
        assert len(grid) == 9

    def test_make_config_dropout(self):
        """Dropout applied correctly to block config."""
        for rate in [0.0, 0.1, 0.2]:
            config = make_config(rate)
            assert config.block.dropout == rate

    def test_make_config_vocab_size(self):
        """GPT-2 BPE vocab (50257)."""
        config = make_config(0.0)
        assert config.vocab_size == 50257

    def test_make_config_preserves_llama(self):
        """Other llama-10m params unchanged."""
        config = make_config(0.1)
        assert config.block.d_model == 128
        assert config.n_layers == 14
        assert config.block.n_heads == 4
        assert config.block.n_kv_heads == 2
        assert config.block.d_ff == 512
        assert config.tie_embeddings is True

    def test_data_compatible_with_batch_iterator(self):
        """Token stream from dataset works with batch_iterator."""
        ds = ModularArithmeticDataset(p=97, split="train", seed=42)
        tokens = ds.get_tokens()
        # Should produce at least one batch
        batches = list(
            batch_iterator(
                tokens,
                batch_size=8,
                seq_len=256,
                shuffle=False,
            )
        )
        assert len(batches) > 0
        x, y = batches[0]
        assert x.shape == (8, 256)
        assert y.shape == (8, 256)

    def test_evaluate_function_returns_all_k(self):
        """Evaluation returns pass@k for all K_VALUES."""
        # Use a tiny model for speed
        config = make_config(0.0)
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokenizer = TiktokenTokenizer("gpt2")

        # Small test set
        ds = ModularArithmeticDataset(p=5, split="test", seed=42)

        results = evaluate_pass_at_k_modular(
            model=model,
            dataset=ds,
            tokenizer=tokenizer,
            k_values=K_VALUES,
            n_samples=64,
            temperature=0.8,
        )

        for k in K_VALUES:
            key = f"pass_at_{k}"
            assert key in results, f"Missing {key}"
            assert 0.0 <= results[key] <= 1.0
