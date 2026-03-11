"""Curriculum learning utilities."""

from collections.abc import Iterator

import mlx.core as mx


def length_curriculum(
    tokens: mx.array,
    batch_size: int,
    min_seq_len: int = 32,
    max_seq_len: int = 512,
    n_stages: int = 4,
    batches_per_stage: int = 100,
) -> Iterator[tuple[mx.array, mx.array]]:
    """Generate batches with increasing sequence length.

    Starts with short sequences and gradually increases,
    following curriculum learning principles.

    Args:
        tokens: Flat array of token IDs.
        batch_size: Sequences per batch.
        min_seq_len: Starting sequence length.
        max_seq_len: Final sequence length.
        n_stages: Number of curriculum stages.
        batches_per_stage: Batches per stage.

    Yields:
        (input, target) tuples with progressively longer sequences.
    """
    for stage in range(n_stages):
        # Linear interpolation of sequence length
        progress = stage / max(n_stages - 1, 1)
        seq_len = int(min_seq_len + progress * (max_seq_len - min_seq_len))

        n_tokens = len(tokens)
        n_sequences = (n_tokens - 1) // seq_len

        if n_sequences < batch_size:
            continue

        for _ in range(batches_per_stage):
            # Random starting positions
            starts = mx.random.randint(
                0, n_tokens - seq_len - 1, shape=(batch_size,)
            )
            mx.eval(starts)

            inputs_list = []
            targets_list = []
            for s in starts.tolist():
                s = int(s)
                inputs_list.append(tokens[s : s + seq_len])
                targets_list.append(tokens[s + 1 : s + seq_len + 1])

            yield mx.stack(inputs_list), mx.stack(targets_list)


def difficulty_curriculum(
    easy_data: mx.array,
    hard_data: mx.array,
    batch_size: int,
    seq_len: int,
    n_batches: int = 200,
    warmup_fraction: float = 0.5,
) -> Iterator[tuple[mx.array, mx.array]]:
    """Mix easy and hard data with increasing difficulty.

    Starts with mostly easy data and transitions to hard data.

    Args:
        easy_data: Token array of easier text.
        hard_data: Token array of harder text.
        batch_size: Sequences per batch.
        seq_len: Sequence length.
        n_batches: Total number of batches.
        warmup_fraction: Fraction of training spent warming up.

    Yields:
        (input, target) tuples with mixed difficulty.
    """
    for i in range(n_batches):
        # Hard data fraction increases linearly
        progress = i / max(n_batches - 1, 1)
        hard_fraction = min(progress / warmup_fraction, 1.0)
        n_hard = int(batch_size * hard_fraction)
        n_easy = batch_size - n_hard

        inputs_list = []
        targets_list = []

        # Sample from easy data
        if n_easy > 0:
            starts = mx.random.randint(
                0, len(easy_data) - seq_len - 1, shape=(n_easy,)
            )
            mx.eval(starts)
            for s in starts.tolist():
                s = int(s)
                inputs_list.append(easy_data[s : s + seq_len])
                targets_list.append(easy_data[s + 1 : s + seq_len + 1])

        # Sample from hard data
        if n_hard > 0:
            starts = mx.random.randint(
                0, len(hard_data) - seq_len - 1, shape=(n_hard,)
            )
            mx.eval(starts)
            for s in starts.tolist():
                s = int(s)
                inputs_list.append(hard_data[s : s + seq_len])
                targets_list.append(hard_data[s + 1 : s + seq_len + 1])

        yield mx.stack(inputs_list), mx.stack(targets_list)
