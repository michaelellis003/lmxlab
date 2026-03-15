"""Modular arithmetic dataset for pass@k evaluation."""

from __future__ import annotations

import mlx.core as mx

from lmxlab.data.tokenizer import TiktokenTokenizer


class ModularArithmeticDataset:
    """Dataset for (a + b) mod p.

    Generates all p*p pairs, splits deterministically into
    train/test, tokenizes as ``"a + b = c\\n"`` using BPE.

    Each number 0-96 is a single GPT-2 BPE token when preceded
    by a space, enabling fast single-token evaluation.

    Args:
        p: Prime modulus (default 97).
        split: ``'train'`` or ``'test'``.
        train_fraction: Fraction for training (0.8).
        seed: Random seed for deterministic split.
    """

    def __init__(
        self,
        p: int = 97,
        split: str = "train",
        train_fraction: float = 0.8,
        seed: int = 0,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.p = p
        self.split = split
        self.train_fraction = train_fraction
        self.seed = seed

        tokenizer = TiktokenTokenizer("gpt2")
        self._tokenizer = tokenizer

        # Pre-compute answer token IDs: " 0", " 1", ..., " p-1"
        self._answer_token_ids: list[int] = []
        for c in range(p):
            toks = tokenizer.encode(f" {c}")
            if len(toks) != 1:
                raise ValueError(
                    f"Number {c} is not a single BPE token: {toks}"
                )
            self._answer_token_ids.append(toks[0])

        # Generate pairs and split
        threshold = int(train_fraction * 10000)
        train_pairs: list[tuple[int, int, int]] = []
        test_pairs: list[tuple[int, int, int]] = []

        for a in range(p):
            for b in range(p):
                c = (a + b) % p
                h = hash((a, b, seed)) % 10000
                if h < threshold:
                    train_pairs.append((a, b, c))
                else:
                    test_pairs.append((a, b, c))

        pairs = train_pairs if split == "train" else test_pairs
        self._pairs = pairs

        # Tokenize all examples into a flat token stream
        all_tokens: list[int] = []
        self._prompts: list[tuple[mx.array, int]] = []
        self._tokens_per_example: int = 0

        for a, b, c in pairs:
            text = f"{a} + {b} = {c}\n"
            toks = tokenizer.encode(text)
            if self._tokens_per_example == 0:
                self._tokens_per_example = len(toks)
            all_tokens.extend(toks)

            # Build prompt: everything up to and including " ="
            prompt_text = f"{a} + {b} ="
            prompt_toks = tokenizer.encode(prompt_text)
            self._prompts.append((mx.array(prompt_toks), c))

        self._tokens = mx.array(all_tokens)

    @property
    def num_examples(self) -> int:
        """Number of examples in this split."""
        return len(self._pairs)

    @property
    def answer_token_ids(self) -> list[int]:
        """Token ID for each answer value 0..p-1."""
        return self._answer_token_ids

    def get_tokens(self) -> mx.array:
        """Flat token stream for next-token prediction.

        Returns:
            1-D array of token IDs.
        """
        return self._tokens

    def get_prompts(self) -> list[tuple[mx.array, int]]:
        """Prompts for pass@k evaluation.

        Returns:
            List of ``(prompt_tokens, answer)`` where
            ``prompt_tokens`` is a 1-D mx.array and
            ``answer`` is the integer c = (a+b) mod p.
        """
        return self._prompts
