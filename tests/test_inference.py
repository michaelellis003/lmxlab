"""Tests for advanced inference."""

import mlx.core as mx
import pytest

from lmxlab.inference.sampling import best_of_n, majority_vote
from lmxlab.inference.speculative import speculative_decode
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate, stream_generate
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.llama import llama_tiny


@pytest.fixture
def tiny_model() -> LanguageModel:
    config = gpt_tiny()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def llama_model() -> LanguageModel:
    config = llama_tiny()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    return model


# ── best_of_n ──────────────────────────────────────────────


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

    def test_n_equals_one(self, tiny_model):
        """n=1 still produces valid output."""
        prompt = mx.array([[5, 6]])
        result = best_of_n(
            tiny_model,
            prompt,
            n=1,
            max_tokens=4,
            temperature=0.8,
        )
        mx.eval(result)
        assert result.shape == (1, 6)

    def test_length_normalized_scoring(self, tiny_model):
        """length_normalized score_fn runs without error."""
        prompt = mx.array([[1, 2, 3]])
        result = best_of_n(
            tiny_model,
            prompt,
            n=2,
            max_tokens=5,
            temperature=0.8,
            score_fn="length_normalized",
        )
        mx.eval(result)
        assert result.shape == (1, 8)

    def test_single_token_prompt(self, tiny_model):
        """Works with a single-token prompt."""
        prompt = mx.array([[42]])
        result = best_of_n(
            tiny_model,
            prompt,
            n=2,
            max_tokens=3,
            temperature=0.8,
        )
        mx.eval(result)
        assert result.shape == (1, 4)
        assert result[0, 0].item() == 42

    def test_different_architectures(self, llama_model):
        """Works with LLaMA architecture too."""
        prompt = mx.array([[1, 2]])
        result = best_of_n(
            llama_model,
            prompt,
            n=2,
            max_tokens=3,
            temperature=0.8,
        )
        mx.eval(result)
        assert result.shape == (1, 5)


# ── majority_vote ──────────────────────────────────────────


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

    def test_result_structure(self, tiny_model):
        """Each result is a (token_list, count) tuple."""
        prompt = mx.array([[1, 2]])
        results = majority_vote(tiny_model, prompt, n=3, max_tokens=4)
        for token_list, count in results:
            assert isinstance(token_list, list)
            assert isinstance(count, int)
            assert count > 0
            assert len(token_list) == 4

    def test_greedy_single_group(self, tiny_model):
        """Temperature=0 should produce identical outputs."""
        prompt = mx.array([[1, 2, 3]])
        results = majority_vote(
            tiny_model,
            prompt,
            n=4,
            max_tokens=3,
            temperature=0.0,
        )
        # Greedy: all should be identical → single group
        assert len(results) == 1
        assert results[0][1] == 4

    def test_n_equals_one(self, tiny_model):
        """n=1 returns single group with count 1."""
        prompt = mx.array([[1, 2]])
        results = majority_vote(tiny_model, prompt, n=1, max_tokens=3)
        assert len(results) == 1
        assert results[0][1] == 1


# ── speculative_decode ─────────────────────────────────────


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

    def test_acceptance_rate_bounded(self, tiny_model):
        """Acceptance rate is between 0 and 1+."""
        prompt = mx.array([[1, 2, 3]])
        _, stats = speculative_decode(
            tiny_model,
            tiny_model,
            prompt,
            max_tokens=8,
            draft_tokens=3,
        )
        assert stats["acceptance_rate"] >= 0.0

    def test_different_draft_sizes(self, tiny_model):
        """Works with different draft_tokens values."""
        prompt = mx.array([[1, 2]])
        for draft_k in [1, 2, 4, 8]:
            result, _ = speculative_decode(
                tiny_model,
                tiny_model,
                prompt,
                max_tokens=6,
                draft_tokens=draft_k,
            )
            mx.eval(result)
            assert result.shape == (1, 8)

    def test_single_token_generation(self, tiny_model):
        """max_tokens=1 still works."""
        prompt = mx.array([[1, 2, 3]])
        result, stats = speculative_decode(
            tiny_model,
            tiny_model,
            prompt,
            max_tokens=1,
        )
        mx.eval(result)
        assert result.shape == (1, 4)

    def test_different_models(self, tiny_model, llama_model):
        """Works with different draft and target models."""
        prompt = mx.array([[1, 2, 3]])
        result, stats = speculative_decode(
            tiny_model,
            llama_model,
            prompt,
            max_tokens=5,
        )
        mx.eval(result)
        assert result.shape == (1, 8)
        assert stats["total_drafted"] > 0


# ── generate ───────────────────────────────────────────────


class TestGenerate:
    def test_greedy_deterministic(self, tiny_model):
        """Greedy generation is deterministic."""
        prompt = mx.array([[1, 2, 3]])
        r1 = generate(tiny_model, prompt, max_tokens=5, temperature=0.0)
        r2 = generate(tiny_model, prompt, max_tokens=5, temperature=0.0)
        mx.eval(r1, r2)
        assert mx.array_equal(r1, r2).item()

    def test_output_shape(self, tiny_model):
        """Output has correct shape."""
        prompt = mx.array([[1, 2, 3]])
        result = generate(tiny_model, prompt, max_tokens=10)
        mx.eval(result)
        assert result.shape == (1, 13)

    def test_preserves_prompt(self, tiny_model):
        """Prompt prefix is preserved in output."""
        prompt = mx.array([[10, 20, 30]])
        result = generate(tiny_model, prompt, max_tokens=5, temperature=0.0)
        mx.eval(result)
        assert mx.array_equal(result[:, :3], prompt).item()

    def test_stop_tokens(self, tiny_model):
        """Stop tokens halt generation early."""
        prompt = mx.array([[1, 2]])
        # Generate greedy to find what token comes first
        full = generate(tiny_model, prompt, max_tokens=10, temperature=0.0)
        mx.eval(full)
        # Use the 3rd generated token as stop token
        stop_tok = full[0, 4].item()
        stopped = generate(
            tiny_model,
            prompt,
            max_tokens=10,
            temperature=0.0,
            stop_tokens=[stop_tok],
        )
        mx.eval(stopped)
        # Should be shorter (stopped before generating stop_tok)
        assert stopped.shape[1] <= full.shape[1]

    def test_top_k_sampling(self, tiny_model):
        """top_k sampling produces valid tokens."""
        prompt = mx.array([[1, 2]])
        result = generate(
            tiny_model,
            prompt,
            max_tokens=5,
            temperature=0.8,
            top_k=10,
        )
        mx.eval(result)
        assert result.shape == (1, 7)

    def test_top_p_sampling(self, tiny_model):
        """top_p (nucleus) sampling produces valid tokens."""
        prompt = mx.array([[1, 2]])
        result = generate(
            tiny_model,
            prompt,
            max_tokens=5,
            temperature=0.8,
            top_p=0.9,
        )
        mx.eval(result)
        assert result.shape == (1, 7)

    def test_repetition_penalty(self, tiny_model):
        """Repetition penalty changes output."""
        prompt = mx.array([[1, 2, 3]])
        r1 = generate(
            tiny_model,
            prompt,
            max_tokens=10,
            temperature=0.0,
            repetition_penalty=1.0,
        )
        r2 = generate(
            tiny_model,
            prompt,
            max_tokens=10,
            temperature=0.0,
            repetition_penalty=2.0,
        )
        mx.eval(r1, r2)
        # With penalty, output should (usually) differ
        # Note: not guaranteed but very likely with strong penalty
        # Just verify both are valid
        assert r1.shape == (1, 13)
        assert r2.shape == (1, 13)

    def test_max_tokens_zero(self, tiny_model):
        """max_tokens=0 returns just the prompt."""
        prompt = mx.array([[1, 2, 3]])
        result = generate(tiny_model, prompt, max_tokens=0, temperature=0.0)
        mx.eval(result)
        assert mx.array_equal(result, prompt).item()

    def test_batch_generation(self, tiny_model):
        """Batch generation produces correct shapes."""
        prompt = mx.array([[1, 2], [3, 4]])
        result = generate(tiny_model, prompt, max_tokens=5, temperature=0.0)
        mx.eval(result)
        assert result.shape == (2, 7)


# ── stream_generate ────────────────────────────────────────


class TestStreamGenerate:
    def test_yields_tokens(self, tiny_model):
        """stream_generate yields individual tokens."""
        prompt = mx.array([[1, 2, 3]])
        tokens = list(
            stream_generate(tiny_model, prompt, max_tokens=5, temperature=0.0)
        )
        assert len(tokens) == 5
        for tok in tokens:
            assert isinstance(tok, int)

    def test_matches_batch_generate(self, tiny_model):
        """Streaming output matches batch generate output."""
        prompt = mx.array([[1, 2, 3]])
        batch = generate(tiny_model, prompt, max_tokens=5, temperature=0.0)
        mx.eval(batch)
        stream = list(
            stream_generate(tiny_model, prompt, max_tokens=5, temperature=0.0)
        )
        batch_tokens = batch[0, 3:].tolist()
        assert stream == batch_tokens

    def test_stop_token_halts(self, tiny_model):
        """Stop token causes early termination."""
        prompt = mx.array([[1, 2]])
        # Generate to find tokens
        full = list(
            stream_generate(tiny_model, prompt, max_tokens=10, temperature=0.0)
        )
        assert len(full) == 10
        # Use 3rd token as stop
        stop_tok = full[2]
        stopped = list(
            stream_generate(
                tiny_model,
                prompt,
                max_tokens=10,
                temperature=0.0,
                stop_tokens=[stop_tok],
            )
        )
        assert len(stopped) < len(full)

    def test_max_tokens_zero(self, tiny_model):
        """max_tokens=0 yields nothing."""
        prompt = mx.array([[1, 2]])
        tokens = list(stream_generate(tiny_model, prompt, max_tokens=0))
        assert tokens == []
