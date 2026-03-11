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

    Can be initialized with text directly, or created with
    default ASCII printable characters (no args). Use
    ``fit()`` to rebuild the vocabulary from new text.

    Args:
        text: Text to build vocabulary from. If None, uses
            ASCII printable characters (32-126).
    """

    def __init__(self, text: str | None = None) -> None:
        if text is not None:
            chars = sorted(set(text))
        else:
            chars = [chr(i) for i in range(32, 127)]
        self._char_to_id: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self._id_to_char: dict[int, str] = {
            i: c for c, i in self._char_to_id.items()
        }

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

    Wraps a tiktoken encoding for use with lmxlab.
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
        return int(self._enc.n_vocab)

    def encode(self, text: str) -> list[int]:
        """Encode text to BPE token IDs.

        Args:
            text: Input string.

        Returns:
            List of token IDs.
        """
        return list(self._enc.encode(text))

    def decode(self, tokens: list[int]) -> str:
        """Decode BPE token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded string.
        """
        return str(self._enc.decode(tokens))


class HFTokenizer:
    """HuggingFace tokenizer wrapper.

    Wraps a HuggingFace ``AutoTokenizer`` for use with lmxlab.
    Use this when working with pretrained models loaded via
    ``load_from_hf``.

    Requires ``transformers`` to be installed::

        pip install transformers

    Args:
        repo_id: HuggingFace model repo ID or local path
            (e.g., 'meta-llama/Llama-3.2-1B').

    Example:
        >>> tok = HFTokenizer('meta-llama/Llama-3.2-1B')
        >>> tok.encode('hello world')
        [15339, 1917]
        >>> tok.decode([15339, 1917])
        'hello world'
    """

    def __init__(self, repo_id: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for HFTokenizer. "
                "Install with: pip install transformers"
            ) from e

        self._tok = AutoTokenizer.from_pretrained(repo_id)
        self._repo_id = repo_id

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return len(self._tok)

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID, if available."""
        eos = self._tok.eos_token_id
        return int(eos) if eos is not None else None

    @property
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence token ID, if available."""
        bos = self._tok.bos_token_id
        return int(bos) if bos is not None else None

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Does not add special tokens (BOS/EOS) by default,
        so the output matches what the model expects for
        continuation.

        Args:
            text: Input string.

        Returns:
            List of token IDs.
        """
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded string.
        """
        return str(self._tok.decode(tokens))
