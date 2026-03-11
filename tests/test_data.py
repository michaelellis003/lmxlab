"""Tests for data pipeline."""

import mlx.core as mx
import pytest

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.dataset import TextDataset, TokenDataset
from lmt_metal.data.tokenizer import CharTokenizer


class TestCharTokenizer:
    def test_roundtrip(self):
        """Encode then decode returns original text."""
        text = "hello world"
        tok = CharTokenizer(text)
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_vocab_size(self):
        """Vocab size matches unique characters."""
        tok = CharTokenizer("aabbcc")
        assert tok.vocab_size == 3  # a, b, c

    def test_default_vocab(self):
        """Default tokenizer uses ASCII printable."""
        tok = CharTokenizer()
        assert tok.vocab_size == 95  # ASCII 32-126

    def test_unknown_char_raises(self):
        """Encoding unknown character raises KeyError."""
        tok = CharTokenizer("abc")
        with pytest.raises(KeyError):
            tok.encode("xyz")

    def test_deterministic(self):
        """Same text always produces same encoding."""
        tok = CharTokenizer("hello")
        assert tok.encode("hello") == tok.encode("hello")


class TestTextDataset:
    def test_length(self):
        text = "abcdefghij"
        tok = CharTokenizer(text)
        ds = TextDataset(text, tok, seq_len=4)
        # 10 tokens, seq_len=4 -> 6 windows
        assert len(ds) == 6

    def test_shapes(self):
        text = "abcdefghij"
        tok = CharTokenizer(text)
        ds = TextDataset(text, tok, seq_len=4)
        x, y = ds[0]
        assert x.shape == (4,)
        assert y.shape == (4,)

    def test_target_is_shifted(self):
        text = "abcde"
        tok = CharTokenizer(text)
        ds = TextDataset(text, tok, seq_len=3)
        x, y = ds[0]
        mx.eval(x, y)
        # y should be x shifted by 1
        tokens = tok.encode(text)
        assert list(x.tolist()) == tokens[:3]
        assert list(y.tolist()) == tokens[1:4]


class TestTokenDataset:
    def test_shapes(self):
        tokens = mx.arange(20, dtype=mx.int32)
        ds = TokenDataset(tokens, seq_len=5)
        assert len(ds) == 15
        x, y = ds[0]
        assert x.shape == (5,)


class TestBatchIterator:
    def test_basic(self):
        tokens = mx.arange(100, dtype=mx.int32)
        batches = list(
            batch_iterator(tokens, batch_size=4, seq_len=10, shuffle=False)
        )
        # 99 usable tokens / 10 = 9 sequences, 9 // 4 = 2 batches
        assert len(batches) == 2
        x, y = batches[0]
        assert x.shape == (4, 10)
        assert y.shape == (4, 10)

    def test_target_shifted(self):
        tokens = mx.arange(50, dtype=mx.int32)
        batches = list(
            batch_iterator(tokens, batch_size=2, seq_len=5, shuffle=False)
        )
        x, y = batches[0]
        mx.eval(x, y)
        # First sequence: x=[0,1,2,3,4], y=[1,2,3,4,5]
        assert x[0, 0].item() == 0
        assert y[0, 0].item() == 1

    def test_too_little_data_raises(self):
        tokens = mx.arange(10, dtype=mx.int32)
        with pytest.raises(ValueError, match="Not enough data"):
            list(batch_iterator(tokens, batch_size=100, seq_len=5))

    def test_shuffle(self):
        """Shuffled batches should differ from unshuffled."""
        mx.random.seed(42)
        tokens = mx.arange(200, dtype=mx.int32)
        b1 = list(
            batch_iterator(tokens, batch_size=4, seq_len=10, shuffle=False)
        )
        mx.random.seed(0)
        b2 = list(
            batch_iterator(tokens, batch_size=4, seq_len=10, shuffle=True)
        )
        # At least one batch should differ
        mx.eval(b1[0][0], b2[0][0])
        differs = not mx.array_equal(b1[0][0], b2[0][0]).item()
        assert differs


class TestHFTokenizer:
    def test_protocol_compliance(self):
        """HFTokenizer has required Tokenizer protocol methods."""
        from lmt_metal.data.tokenizer import HFTokenizer

        assert hasattr(HFTokenizer, "encode")
        assert hasattr(HFTokenizer, "decode")
        assert hasattr(HFTokenizer, "vocab_size")

    def test_has_special_token_properties(self):
        """HFTokenizer exposes eos and bos token IDs."""
        from lmt_metal.data.tokenizer import HFTokenizer

        assert hasattr(HFTokenizer, "eos_token_id")
        assert hasattr(HFTokenizer, "bos_token_id")
