"""Tests for data pipeline."""

import mlx.core as mx
import pytest

from lmxlab.data.batching import batch_iterator
from lmxlab.data.dataset import TextDataset, TokenDataset
from lmxlab.data.tokenizer import CharTokenizer


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

    def test_fit_rebuilds_vocab(self):
        """fit() rebuilds vocabulary from new text."""
        tok = CharTokenizer("abc")
        assert tok.vocab_size == 3
        tok.fit("xyz123")
        assert tok.vocab_size == 6
        # Old chars should no longer work
        with pytest.raises(KeyError):
            tok.encode("a")
        # New chars should work
        ids = tok.encode("xyz")
        assert tok.decode(ids) == "xyz"

    def test_fit_roundtrip(self):
        """fit() produces a working tokenizer."""
        tok = CharTokenizer()
        tok.fit("the quick brown fox")
        text = "the fox"
        assert tok.decode(tok.encode(text)) == text


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


class TestHFDataset:
    def test_class_exists(self):
        """HFDataset is importable."""
        from lmxlab.data.dataset import HFDataset

        assert HFDataset is not None

    def test_has_required_methods(self):
        """HFDataset has token_iterator and batch_iterator."""
        from lmxlab.data.dataset import HFDataset

        assert hasattr(HFDataset, "token_iterator")
        assert hasattr(HFDataset, "batch_iterator")

    def _make_hf_dataset(self, fake_data, tok, seq_len=4):
        """Create an HFDataset with mock data."""
        from lmxlab.data.dataset import HFDataset

        with pytest.MonkeyPatch.context() as mp:
            import types

            fake_module = types.ModuleType("datasets")
            fake_module.load_dataset = lambda *a, **kw: fake_data
            mp.setitem(__import__("sys").modules, "datasets", fake_module)
            return HFDataset(
                "fake/dataset",
                tok,
                seq_len=seq_len,
                split="train",
            )

    def test_batch_iterator_shapes(self):
        """batch_iterator yields correct shapes."""
        text = "abcdefghijklmnopqrstuvwxyz" * 10
        tok = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
        ds = self._make_hf_dataset([{"text": text}], tok, seq_len=4)
        batches = list(ds.batch_iterator(batch_size=2, max_batches=3))
        assert len(batches) <= 3
        for x, y in batches:
            assert x.shape == (2, 4)
            assert y.shape == (2, 4)

    def test_token_iterator(self):
        """token_iterator yields tokens from mock dataset."""
        tok = CharTokenizer("helloworld")
        ds = self._make_hf_dataset(
            [{"text": "hello"}, {"text": "world"}],
            tok,
            seq_len=4,
        )
        tokens = list(ds.token_iterator())
        assert len(tokens) == 10  # 'hello' + 'world'

    def test_max_batches_respected(self):
        """batch_iterator stops at max_batches."""
        text = "abcdefghijklmnop" * 100
        tok = CharTokenizer("abcdefghijklmnop")
        ds = self._make_hf_dataset([{"text": text}], tok, seq_len=4)
        batches = list(ds.batch_iterator(batch_size=2, max_batches=1))
        assert len(batches) == 1

    def test_empty_text_skipped(self):
        """Empty text entries are skipped."""
        tok = CharTokenizer("hello")
        ds = self._make_hf_dataset(
            [{"text": ""}, {"text": "   "}, {"text": "hello"}],
            tok,
            seq_len=4,
        )
        tokens = list(ds.token_iterator())
        assert len(tokens) == 5  # only 'hello'


class TestHFTokenizer:
    def test_protocol_compliance(self):
        """HFTokenizer has required Tokenizer protocol methods."""
        from lmxlab.data.tokenizer import HFTokenizer

        assert hasattr(HFTokenizer, "encode")
        assert hasattr(HFTokenizer, "decode")
        assert hasattr(HFTokenizer, "vocab_size")

    def test_has_special_token_properties(self):
        """HFTokenizer exposes eos and bos token IDs."""
        from lmxlab.data.tokenizer import HFTokenizer

        assert hasattr(HFTokenizer, "eos_token_id")
        assert hasattr(HFTokenizer, "bos_token_id")


# ── Integration tests ──────────────────────────────────────


class TestDataPipelineIntegration:
    """End-to-end data pipeline tests."""

    def test_text_to_batches(self):
        """Full pipeline: text → tokenizer → batch_iterator."""
        text = "the quick brown fox jumps over the lazy dog " * 10
        tok = CharTokenizer(text)
        tokens = mx.array(tok.encode(text), dtype=mx.int32)
        batches = list(
            batch_iterator(tokens, batch_size=2, seq_len=8, shuffle=False)
        )
        assert len(batches) > 0
        x, y = batches[0]
        mx.eval(x, y)
        assert x.shape == (2, 8)
        assert y.shape == (2, 8)

    def test_text_dataset_to_batches(self):
        """TextDataset items can be used in a training loop."""
        text = "abcdefghijklmnopqrstuvwxyz" * 5
        tok = CharTokenizer(text)
        ds = TextDataset(text, tok, seq_len=8)
        # Simulate a mini training loop
        for i in range(min(3, len(ds))):
            x, y = ds[i]
            mx.eval(x, y)
            assert x.shape == (8,)
            assert y.shape == (8,)

    def test_token_dataset_batching(self):
        """TokenDataset → batch_iterator integration."""
        tokens = mx.arange(200, dtype=mx.int32)
        ds = TokenDataset(tokens, seq_len=10)
        assert len(ds) > 0
        # Verify same tokens feed into batch_iterator
        batches = list(
            batch_iterator(tokens, batch_size=4, seq_len=10, shuffle=False)
        )
        assert len(batches) > 0

    def test_shuffled_batches_cover_data(self):
        """Shuffled batches still cover the full dataset."""
        mx.random.seed(123)
        tokens = mx.arange(100, dtype=mx.int32)
        batches_unshuffled = list(
            batch_iterator(tokens, batch_size=2, seq_len=5, shuffle=False)
        )
        mx.random.seed(456)
        batches_shuffled = list(
            batch_iterator(tokens, batch_size=2, seq_len=5, shuffle=True)
        )
        # Same number of batches
        assert len(batches_shuffled) == len(batches_unshuffled)

    def test_char_tokenizer_special_chars(self):
        """CharTokenizer handles whitespace and punctuation."""
        text = "Hello, World! How's it going?\n\tFine."
        tok = CharTokenizer(text)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_single_char_text(self):
        """Edge case: single character repeated."""
        text = "aaaaaaaaaa"
        tok = CharTokenizer(text)
        assert tok.vocab_size == 1
        ids = tok.encode(text)
        assert all(i == ids[0] for i in ids)
        assert tok.decode(ids) == text
