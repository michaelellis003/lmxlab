"""Tests for advanced inference."""

import mlx.core as mx
import pytest

from lmxlab.inference.sampling import best_of_n, majority_vote
from lmxlab.inference.speculative import speculative_decode
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_tiny


@pytest.fixture
def tiny_model() -> LanguageModel:
    config = gpt_tiny()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


class TestBestOfN:
    def test_output_shape(self, tiny_model):
        """Returns single best completion."""
        prompt = mx.array([[1, 2, 3]])
        result = best_of_n(
            tiny_model,
            prompt,
            n=3,
            max_tokens=5,
            temperature=0.8,
        )
        mx.eval(result)
        assert result.shape[0] == 1
        assert result.shape[1] == 8  # 3 + 5

    def test_preserves_prompt(self, tiny_model):
        """Prompt prefix is preserved."""
        prompt = mx.array([[10, 20, 30]])
        result = best_of_n(
            tiny_model,
            prompt,
            n=2,
            max_tokens=3,
            temperature=0.8,
        )
        mx.eval(result)
        assert mx.array_equal(result[:, :3], prompt).item()


class TestMajorityVote:
    def test_returns_nonempty(self, tiny_model):
        """At least one group is returned."""
        prompt = mx.array([[1, 2]])
        results = majority_vote(tiny_model, prompt, n=3, max_tokens=5)
        assert len(results) > 0

    def test_counts_sum_to_n(self, tiny_model):
        """Total counts should equal n."""
        n = 5
        prompt = mx.array([[1, 2]])
        results = majority_vote(tiny_model, prompt, n=n, max_tokens=3)
        total = sum(count for _, count in results)
        assert total == n

    def test_sorted_by_count(self, tiny_model):
        """Results sorted by frequency descending."""
        prompt = mx.array([[1, 2]])
        results = majority_vote(
            tiny_model,
            prompt,
            n=10,
            max_tokens=3,
            temperature=0.5,
        )
        counts = [count for _, count in results]
        assert counts == sorted(counts, reverse=True)


class TestSpeculativeDecode:
    def test_output_length(self, tiny_model):
        """Speculative decode produces correct length."""
        prompt = mx.array([[1, 2, 3]])
        result, stats = speculative_decode(
            tiny_model,
            tiny_model,  # same model as draft
            prompt,
            max_tokens=5,
        )
        mx.eval(result)
        assert result.shape == (1, 8)  # 3 + 5

    def test_same_model_high_acceptance(self, tiny_model):
        """Using same model as draft should have high acceptance."""
        prompt = mx.array([[1, 2, 3]])
        _, stats = speculative_decode(
            tiny_model,
            tiny_model,
            prompt,
            max_tokens=10,
            draft_tokens=4,
        )
        # Same model should accept everything
        assert stats["acceptance_rate"] >= 0.9

    def test_preserves_prompt(self, tiny_model):
        """Prompt tokens are preserved."""
        prompt = mx.array([[10, 20, 30]])
        result, _ = speculative_decode(
            tiny_model,
            tiny_model,
            prompt,
            max_tokens=5,
        )
        mx.eval(result)
        assert mx.array_equal(result[:, :3], prompt).item()

    def test_stats_populated(self, tiny_model):
        """Stats dict has expected keys."""
        prompt = mx.array([[1, 2]])
        _, stats = speculative_decode(
            tiny_model,
            tiny_model,
            prompt,
            max_tokens=5,
        )
        assert "acceptance_rate" in stats
        assert "total_drafted" in stats
        assert "total_accepted" in stats
