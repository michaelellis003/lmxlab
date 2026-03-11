"""Dataset classes for language model training."""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx

from lmxlab.data.tokenizer import Tokenizer


class TextDataset:
    """Dataset that tokenizes raw text.

    Tokenizes text and stores as a flat array of token IDs.
    Yields overlapping windows of (input, target) pairs.

    Args:
        text: Raw text to tokenize.
        tokenizer: Tokenizer to use.
        seq_len: Sequence length for training windows.
    """

    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        seq_len: int = 128,
    ) -> None:
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        tokens = tokenizer.encode(text)
        self.tokens = mx.array(tokens, dtype=mx.int32)

    def __len__(self) -> int:
        """Number of training windows available."""
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[mx.array, mx.array]:
        """Get a (input, target) pair at the given index.

        Args:
            idx: Starting position in the token array.

        Returns:
            Tuple of (input_tokens, target_tokens), each
            of shape (seq_len,).
        """
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


class TokenDataset:
    """Dataset from pre-tokenized data.

    Wraps an existing array of token IDs.

    Args:
        tokens: Array of token IDs.
        seq_len: Sequence length for training windows.
    """

    def __init__(
        self,
        tokens: mx.array,
        seq_len: int = 128,
    ) -> None:
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Number of training windows available."""
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[mx.array, mx.array]:
        """Get a (input, target) pair.

        Args:
            idx: Starting position.

        Returns:
            Tuple of (input_tokens, target_tokens).
        """
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


class HFDataset:
    """Dataset backed by a HuggingFace dataset.

    Streams or loads a HuggingFace dataset, tokenizes on-the-fly,
    and yields batches of (input, target) pairs.

    Requires the ``datasets`` package (``pip install datasets``).

    Args:
        name: HuggingFace dataset name (e.g. ``'wikitext'``).
        tokenizer: Tokenizer implementing the Tokenizer protocol.
        seq_len: Sequence length for training windows.
        split: Dataset split to use.
        text_field: Name of the text column in the dataset.
        config_name: Optional dataset configuration name.
        streaming: Whether to stream the dataset.
    """

    def __init__(
        self,
        name: str,
        tokenizer: Tokenizer,
        seq_len: int = 128,
        split: str = "train",
        text_field: str = "text",
        config_name: str | None = None,
        streaming: bool = False,
    ) -> None:
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self._streaming = streaming
        self._dataset = load_dataset(
            name, config_name, split=split, streaming=streaming
        )

    def token_iterator(self) -> Iterator[int]:
        """Yield token IDs one at a time from the dataset."""
        for example in self._dataset:
            text = example[self.text_field]
            if text and text.strip():
                yield from self.tokenizer.encode(text)

    def batch_iterator(
        self,
        batch_size: int = 8,
        max_batches: int | None = None,
    ) -> Iterator[tuple[mx.array, mx.array]]:
        """Yield (input, target) batches from the dataset.

        Accumulates tokens into a buffer and yields batches
        of shape ``(batch_size, seq_len)``.

        Args:
            batch_size: Number of sequences per batch.
            max_batches: Stop after this many batches.

        Yields:
            Tuple of (inputs, targets), each of shape
            ``(batch_size, seq_len)``.
        """
        buffer: list[int] = []
        tokens_needed = batch_size * self.seq_len + 1
        n_batches = 0
        for token_id in self.token_iterator():
            buffer.append(token_id)
            if len(buffer) >= tokens_needed:
                arr = mx.array(buffer[:tokens_needed], dtype=mx.int32)
                inputs = arr[:-1].reshape(batch_size, self.seq_len)
                targets = arr[1:].reshape(batch_size, self.seq_len)
                yield inputs, targets
                n_batches += 1
                if max_batches and n_batches >= max_batches:
                    return
                buffer = buffer[tokens_needed - 1 :]
