"""Tokenizer protocol and implementations."""

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

    Args:
        text: Text to build vocabulary from. If None, uses
            ASCII printable characters.
    """

    def __init__(self, text: str | None = None) -> None:
        if text is not None:
            chars = sorted(set(text))
        else:
            # ASCII printable characters
            chars = [chr(i) for i in range(32, 127)]

        self._char_to_id: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self._id_to_char: dict[int, str] = {
            i: c for c, i in self._char_to_id.items()
        }

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
