"""Tests for ModularArithmeticDataset and pass@k evaluation."""

import re

import pytest

pytest.importorskip("tiktoken")

import mlx.core as mx

from lmxlab.data.modular_arithmetic import ModularArithmeticDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.eval.metrics import pass_at_k


class TestModularArithmeticDataset:
    """Comprehensive tests for the dataset."""

    def test_split_sizes(self):
        """80/20 split produces correct sizes for p=97."""
        train = ModularArithmeticDataset(p=97, split="train")
        test = ModularArithmeticDataset(p=97, split="test")
        total = train.num_examples + test.num_examples
        assert total == 97 * 97
        # ~80% train
        assert 7000 < train.num_examples < 8000
        # ~20% test
        assert 1400 < test.num_examples < 2500

    def test_no_overlap(self):
        """Train/test pairs are fully disjoint."""
        train = ModularArithmeticDataset(p=97, split="train")
        test = ModularArithmeticDataset(p=97, split="test")
        train_set = {(a, b) for a, b, _ in train._pairs}
        test_set = {(a, b) for a, b, _ in test._pairs}
        assert train_set & test_set == set()

    def test_complete_coverage(self):
        """Train + test covers all p*p pairs."""
        train = ModularArithmeticDataset(p=97, split="train")
        test = ModularArithmeticDataset(p=97, split="test")
        train_set = {(a, b) for a, b, _ in train._pairs}
        test_set = {(a, b) for a, b, _ in test._pairs}
        all_pairs = {(a, b) for a in range(97) for b in range(97)}
        assert train_set | test_set == all_pairs

    def test_answers_correct(self):
        """Every (a+b) mod p is verified."""
        ds = ModularArithmeticDataset(p=97, split="train")
        for a, b, c in ds._pairs:
            assert c == (a + b) % 97

        ds_test = ModularArithmeticDataset(p=97, split="test")
        for a, b, c in ds_test._pairs:
            assert c == (a + b) % 97

    def test_deterministic_split(self):
        """Same seed produces same split."""
        ds1 = ModularArithmeticDataset(p=97, split="train", seed=42)
        ds2 = ModularArithmeticDataset(p=97, split="train", seed=42)
        pairs1 = {(a, b) for a, b, _ in ds1._pairs}
        pairs2 = {(a, b) for a, b, _ in ds2._pairs}
        assert pairs1 == pairs2

    def test_different_seeds_differ(self):
        """Different seeds produce different splits."""
        ds1 = ModularArithmeticDataset(p=97, split="train", seed=0)
        ds2 = ModularArithmeticDataset(p=97, split="train", seed=1)
        pairs1 = {(a, b) for a, b, _ in ds1._pairs}
        pairs2 = {(a, b) for a, b, _ in ds2._pairs}
        assert pairs1 != pairs2

    def test_prompts_shape(self):
        """get_prompts() returns list of (mx.array, int)."""
        ds = ModularArithmeticDataset(p=97, split="test")
        prompts = ds.get_prompts()
        assert len(prompts) == ds.num_examples
        for prompt_tokens, answer in prompts:
            assert isinstance(prompt_tokens, mx.array)
            assert isinstance(answer, int)
            assert 0 <= answer < 97

    def test_prompt_answer_correct(self):
        """Each prompt's answer matches (a+b) mod p."""
        ds = ModularArithmeticDataset(p=97, split="test")
        tokenizer = TiktokenTokenizer("gpt2")
        for prompt_tokens, answer in ds.get_prompts():
            text = tokenizer.decode(prompt_tokens.tolist())
            # text like "42 + 55 ="
            match = re.match(r"(\d+) \+ (\d+) =", text)
            assert match is not None, f"Bad prompt: {text!r}"
            a, b = int(match.group(1)), int(match.group(2))
            assert answer == (a + b) % 97

    def test_small_modulus(self):
        """p=5: 25 total, ~20 train, ~5 test."""
        train = ModularArithmeticDataset(p=5, split="train")
        test = ModularArithmeticDataset(p=5, split="test")
        total = train.num_examples + test.num_examples
        assert total == 25
        assert train.num_examples > 0
        assert test.num_examples > 0

    def test_token_stream_length(self):
        """Token count matches n_examples * tokens_per_example."""
        ds = ModularArithmeticDataset(p=5, split="train")
        tokens = ds.get_tokens()
        expected = ds.num_examples * ds._tokens_per_example
        assert len(tokens) == expected

    def test_tokenization_roundtrip(self):
        """decode(encode(example)) recovers original text."""
        tokenizer = TiktokenTokenizer("gpt2")
        ds = ModularArithmeticDataset(p=11, split="train")
        for a, b, c in ds._pairs:
            text = f"{a} + {b} = {c}\n"
            toks = tokenizer.encode(text)
            recovered = tokenizer.decode(toks)
            assert recovered == text

    def test_invalid_split_raises(self):
        """Invalid split name raises ValueError."""
        try:
            ModularArithmeticDataset(p=5, split="val")
            raise AssertionError("Should have raised")
        except ValueError:
            pass


class TestTokenFormatCrossReference:
    """Cross-reference GPT-2 BPE tokenization assumptions."""

    def test_numbers_are_single_tokens(self):
        """GPT-2 BPE encodes " 0" through " 96" as single tokens."""
        tokenizer = TiktokenTokenizer("gpt2")
        for n in range(97):
            toks = tokenizer.encode(f" {n}")
            assert len(toks) == 1, (
                f"' {n}' encoded to {len(toks)} tokens: {toks}"
            )

    def test_answer_token_ids_unique(self):
        """All 97 answer token IDs are distinct."""
        ds = ModularArithmeticDataset(p=97, split="train")
        ids = ds.answer_token_ids
        assert len(set(ids)) == 97

    def test_prompt_ends_with_equals(self):
        """Every prompt's last token is '=' or ' ='."""
        tokenizer = TiktokenTokenizer("gpt2")
        ds = ModularArithmeticDataset(p=11, split="test")
        for prompt_tokens, _ in ds.get_prompts():
            last_tok = prompt_tokens[-1].item()
            decoded = tokenizer.decode([last_tok])
            assert "=" in decoded, (
                f"Last token decodes to {decoded!r}, expected '='"
            )

    def test_example_format(self):
        """Each example matches regex r'\\d+ \\+ \\d+ = \\d+\\n'."""
        tokenizer = TiktokenTokenizer("gpt2")
        ds = ModularArithmeticDataset(p=11, split="train")
        pattern = re.compile(r"\d+ \+ \d+ = \d+\n")
        for a, b, c in ds._pairs:
            text = f"{a} + {b} = {c}\n"
            assert pattern.fullmatch(text), f"Bad format: {text!r}"
            # Also verify roundtrip
            toks = tokenizer.encode(text)
            decoded = tokenizer.decode(toks)
            assert pattern.fullmatch(decoded)


class TestPassAtKEvaluation:
    """Test pass@k metric properties."""

    def test_k_monotonicity(self):
        """pass@k is non-decreasing in k."""
        n, c = 64, 10
        prev = 0.0
        for k in [1, 2, 4, 8, 16, 32, 64]:
            score = pass_at_k(n, c, k)
            assert score >= prev - 1e-10, f"pass@{k}={score} < pass@{prev}"
            prev = score

    def test_perfect_model_score(self):
        """When c=n, pass@k=1.0 for all k."""
        n = 64
        for k in [1, 4, 16, 64]:
            assert pass_at_k(n, n, k) == 1.0

    def test_zero_model_score(self):
        """When c=0, pass@k=0.0 for all k."""
        n = 64
        for k in [1, 4, 16, 64]:
            assert pass_at_k(n, 0, k) == 0.0

    def test_known_accuracy(self):
        """With c=3, n=10, verify against hand-computed values.

        pass@1 = 1 - C(7,1)/C(10,1) = 1 - 7/10 = 0.3
        pass@5 = 1 - C(7,5)/C(10,5) = 1 - 21/252 ≈ 0.9167
        """
        assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-6
        assert abs(pass_at_k(10, 3, 5) - (1 - 21 / 252)) < 1e-6
