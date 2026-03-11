"""Tokenizer protocol and implementations."""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    """Protocol for tokenizers.

    All tokenizers must implement encode/decode and expose
    their vocabulary size.
    """

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input string.

        Returns:
            List of token IDs.
        """
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded string.
        """
        ...


class CharTokenizer:
    """Character-level tokenizer.

    Simple tokenizer that maps each unique character to an ID.
    Useful for testing and small-scale experiments.

    Can be initialized with text directly, or created empty
    and fitted later via ``fit()``.

    Args:
        text: Text to build vocabulary from. If None, call
            ``fit()`` before encoding.
    """

    def __init__(self, text: str | None = None) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        if text is not None:
            self.fit(text)

    def fit(self, text: str) -> None:
        """Build vocabulary from text.

        Args:
            text: Text to extract characters from.
        """
        chars = sorted(set(text))
        self._char_to_id = {c: i for i, c in enumerate(chars)}
        self._id_to_char = {i: c for c, i in self._char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return len(self._char_to_id)

    def encode(self, text: str) -> list[int]:
        """Encode text to character-level token IDs.

        Args:
            text: Input string.

        Returns:
            List of token IDs.

        Raises:
            KeyError: If text contains unknown characters.
        """
        return [self._char_to_id[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded string.
        """
        return "".join(self._id_to_char[t] for t in tokens)


class TiktokenTokenizer:
    """BPE tokenizer using OpenAI's tiktoken.

    Wraps a tiktoken encoding for use with lmt-metal.
    Supports any tiktoken encoding name (e.g. 'gpt2',
    'cl100k_base', 'o200k_base').

    Requires ``tiktoken`` to be installed::

        pip install tiktoken

    Args:
        encoding_name: Name of the tiktoken encoding.
            Defaults to 'gpt2' (50257 tokens).

    Example:
        >>> tok = TiktokenTokenizer('gpt2')
        >>> tok.encode('hello world')
        [31373, 995]
        >>> tok.decode([31373, 995])
        'hello world'
    """

    def __init__(self, encoding_name: str = "gpt2") -> None:
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. "
                "Install it with: pip install tiktoken"
            ) from e

        self._enc = tiktoken.get_encoding(encoding_name)
        self._encoding_name = encoding_name

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        """Encode text to BPE token IDs.

        Args:
            text: Input string.

        Returns:
            List of token IDs.
        """
        return self._enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode BPE token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded string.
        """
        return self._enc.decode(tokens)
