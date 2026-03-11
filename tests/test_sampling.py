"""Unit tests for sampling helper functions in generate.py."""

import mlx.core as mx

from lmxlab.models.generate import (
    _apply_repetition_penalty,
    _sample_next,
    _sample_top_k,
    _sample_top_p,
)


class TestSampleTopP:
    """Tests for nucleus (top-p) sampling."""

    def test_output_shape(self):
        """Output shape is (batch, 1)."""
        logits = mx.random.normal(shape=(2, 100))
        result = _sample_top_p(logits, top_p=0.9)
        mx.eval(result)
        assert result.shape == (2, 1)

    def test_samples_within_vocab(self):
        """Sampled tokens are valid vocab indices."""
        vocab = 50
        logits = mx.random.normal(shape=(1, vocab))
        result = _sample_top_p(logits, top_p=0.9)
        mx.eval(result)
        assert 0 <= result[0, 0].item() < vocab

    def test_top_p_1_allows_all(self):
        """top_p=1.0 allows sampling from full distribution."""
        logits = mx.random.normal(shape=(1, 10))
        result = _sample_top_p(logits, top_p=1.0)
        mx.eval(result)
        assert 0 <= result[0, 0].item() < 10

    def test_top_p_small_concentrates(self):
        """Small top_p concentrates on high-probability tokens."""
        # One very dominant token
        logits = mx.zeros((1, 10))
        logits = logits.at[0, 3].add(100.0)
        result = _sample_top_p(logits, top_p=0.1)
        mx.eval(result)
        # Should always pick the dominant token
        assert result[0, 0].item() == 3

    def test_batch_independent(self):
        """Each batch element samples independently."""
        logits = mx.zeros((2, 5))
        logits = logits.at[0, 0].add(100.0)
        logits = logits.at[1, 4].add(100.0)
        result = _sample_top_p(logits, top_p=0.1)
        mx.eval(result)
        assert result[0, 0].item() == 0
        assert result[1, 0].item() == 4


class TestSampleTopK:
    """Tests for top-k sampling."""

    def test_output_shape(self):
        """Output shape is (batch, 1)."""
        logits = mx.random.normal(shape=(2, 100))
        result = _sample_top_k(logits, top_k=10)
        mx.eval(result)
        assert result.shape == (2, 1)

    def test_samples_from_top_k(self):
        """Sampled token must be among top-k by logit value."""
        mx.random.seed(42)
        vocab = 20
        logits = mx.random.normal(shape=(1, vocab))
        mx.eval(logits)
        top_k = 3

        # Find actual top-k indices
        top_indices = set(mx.argsort(-logits, axis=-1)[0, :top_k].tolist())

        # Sample many times and verify all are in top-k
        for _ in range(20):
            result = _sample_top_k(logits, top_k=top_k)
            mx.eval(result)
            assert result[0, 0].item() in top_indices

    def test_k_equals_one_is_greedy(self):
        """top_k=1 should be equivalent to greedy."""
        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
        result = _sample_top_k(logits, top_k=1)
        mx.eval(result)
        assert result[0, 0].item() == 1  # index of max (5.0)

    def test_samples_within_vocab(self):
        """Sampled tokens are valid vocab indices."""
        vocab = 30
        logits = mx.random.normal(shape=(1, vocab))
        result = _sample_top_k(logits, top_k=5)
        mx.eval(result)
        assert 0 <= result[0, 0].item() < vocab


class TestApplyRepetitionPenalty:
    """Tests for repetition penalty logic."""

    def test_no_penalty_unchanged(self):
        """penalty=1.0 returns logits unchanged."""
        logits = mx.array([[1.0, 2.0, 3.0]])
        generated = [mx.array([[0]])]
        result = _apply_repetition_penalty(logits, generated, 1.0)
        mx.eval(result)
        assert mx.array_equal(result, logits).item()

    def test_empty_generated_unchanged(self):
        """Empty generated list returns logits unchanged."""
        logits = mx.array([[1.0, 2.0, 3.0]])
        result = _apply_repetition_penalty(logits, [], 2.0)
        mx.eval(result)
        assert mx.array_equal(result, logits).item()

    def test_positive_logit_divided(self):
        """Positive logits for repeated tokens are divided."""
        logits = mx.array([[4.0, 2.0, 1.0]])
        generated = [mx.array([[0]])]  # token 0 was generated
        result = _apply_repetition_penalty(logits, generated, 2.0)
        mx.eval(result)
        # Token 0: 4.0 / 2.0 = 2.0; others unchanged
        assert abs(result[0, 0].item() - 2.0) < 1e-5
        assert abs(result[0, 1].item() - 2.0) < 1e-5
        assert abs(result[0, 2].item() - 1.0) < 1e-5

    def test_negative_logit_multiplied(self):
        """Negative logits for repeated tokens are multiplied."""
        logits = mx.array([[-2.0, 1.0, 3.0]])
        generated = [mx.array([[0]])]  # token 0 was generated
        result = _apply_repetition_penalty(logits, generated, 2.0)
        mx.eval(result)
        # Token 0: -2.0 * 2.0 = -4.0 (more negative = lower prob)
        assert abs(result[0, 0].item() - (-4.0)) < 1e-5
        assert abs(result[0, 1].item() - 1.0) < 1e-5

    def test_multiple_generated_tokens(self):
        """Penalty applies to all previously generated tokens."""
        logits = mx.array([[4.0, 6.0, 1.0]])
        generated = [mx.array([[0]]), mx.array([[1]])]
        result = _apply_repetition_penalty(logits, generated, 2.0)
        mx.eval(result)
        # Token 0: 4.0 / 2.0 = 2.0
        # Token 1: 6.0 / 2.0 = 3.0
        # Token 2: unchanged
        assert abs(result[0, 0].item() - 2.0) < 1e-5
        assert abs(result[0, 1].item() - 3.0) < 1e-5
        assert abs(result[0, 2].item() - 1.0) < 1e-5

    def test_ungenerated_tokens_unchanged(self):
        """Tokens not in generated list are unaffected."""
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        generated = [mx.array([[1]])]  # only token 1
        result = _apply_repetition_penalty(logits, generated, 5.0)
        mx.eval(result)
        assert abs(result[0, 0].item() - 1.0) < 1e-5
        assert abs(result[0, 2].item() - 3.0) < 1e-5
        assert abs(result[0, 3].item() - 4.0) < 1e-5


class TestSampleNext:
    """Tests for the _sample_next dispatch function."""

    def test_greedy_picks_argmax(self):
        """temperature=0 returns argmax."""
        logits = mx.array([[1.0, 5.0, 3.0, 2.0]])
        result = _sample_next(logits, temperature=0.0, top_k=0, top_p=1.0)
        mx.eval(result)
        assert result[0, 0].item() == 1

    def test_output_shape(self):
        """Output is (batch, 1) for all modes."""
        logits = mx.random.normal(shape=(3, 50))
        for temp, k, p in [
            (0.0, 0, 1.0),
            (1.0, 0, 1.0),
            (1.0, 5, 1.0),
            (1.0, 0, 0.9),
        ]:
            result = _sample_next(logits, temp, k, p)
            mx.eval(result)
            assert result.shape == (3, 1), f"Failed for t={temp} k={k} p={p}"

    def test_top_k_dispatch(self):
        """top_k > 0 uses top-k sampling path."""
        logits = mx.array([[0.0, 100.0, 0.0, 0.0]])
        result = _sample_next(logits, temperature=1.0, top_k=1, top_p=1.0)
        mx.eval(result)
        assert result[0, 0].item() == 1

    def test_top_p_dispatch(self):
        """top_p < 1.0 uses nucleus sampling path."""
        logits = mx.array([[0.0, 100.0, 0.0, 0.0]])
        result = _sample_next(logits, temperature=1.0, top_k=0, top_p=0.1)
        mx.eval(result)
        assert result[0, 0].item() == 1

    def test_temperature_scaling(self):
        """Higher temperature produces more uniform distribution."""
        # With very peaked logits, low temp should be deterministic
        logits = mx.array([[0.0, 10.0, 0.0]])
        result = _sample_next(logits, temperature=0.01, top_k=0, top_p=1.0)
        mx.eval(result)
        assert result[0, 0].item() == 1
