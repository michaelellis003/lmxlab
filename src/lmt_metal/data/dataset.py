"""Dataset classes for language model training."""

import mlx.core as mx

from lmt_metal.data.tokenizer import Tokenizer


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
