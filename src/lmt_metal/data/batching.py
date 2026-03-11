"""Batch iterator for MLX training."""

from collections.abc import Iterator

import mlx.core as mx


def batch_iterator(
    tokens: mx.array,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
) -> Iterator[tuple[mx.array, mx.array]]:
    """Yield batches of (input, target) pairs from a token array.

    Creates non-overlapping windows from the token array,
    optionally shuffles, and yields batches.

    Args:
        tokens: Flat array of token IDs.
        batch_size: Number of sequences per batch.
        seq_len: Length of each sequence.
        shuffle: Whether to shuffle windows each epoch.

    Yields:
        Tuples of (input_batch, target_batch), each of
        shape (batch_size, seq_len).
    """
    # Calculate number of complete sequences
    n_tokens = len(tokens)
    n_sequences = (n_tokens - 1) // seq_len

    if n_sequences < batch_size:
        raise ValueError(
            f"Not enough data for batch_size={batch_size}: "
            f"only {n_sequences} sequences available"
        )

    # Truncate to fit evenly
    usable = n_sequences * seq_len
    data = tokens[: usable + 1]

    # Create input/target arrays
    # Shape: (n_sequences, seq_len)
    inputs = mx.stack(
        [data[i * seq_len : (i + 1) * seq_len] for i in range(n_sequences)]
    )
    targets = mx.stack(
        [
            data[i * seq_len + 1 : (i + 1) * seq_len + 1]
            for i in range(n_sequences)
        ]
    )

    # Shuffle
    if shuffle:
        indices = mx.random.permutation(n_sequences)
        inputs = inputs[indices]
        targets = targets[indices]

    # Yield batches
    n_batches = n_sequences // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        yield inputs[start:end], targets[start:end]
