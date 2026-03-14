"""Reward model for scoring completions.

Wraps a LanguageModel with a scalar head that projects
the last-token hidden state to a reward score.
"""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.models.base import LanguageModel


class RewardModel(nn.Module):
    """Reward model: language model + scalar head.

    Takes token sequences and returns a scalar reward score
    based on the last-token hidden state.

    Args:
        model: Base language model.

    Example:
        >>> rm = RewardModel(model)
        >>> scores = rm(token_ids)  # (batch, 1)
    """

    def __init__(self, model: LanguageModel) -> None:
        super().__init__()
        self.model = model
        d_model = model.config.block.d_model
        self.score_head = nn.Linear(d_model, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Score sequences.

        Args:
            x: Token IDs (batch, seq_len).

        Returns:
            Scalar reward scores (batch, 1).
        """
        logits, _, hidden = self.model(x, return_hidden=True)
        # Take last-token hidden state
        last_hidden = hidden[:, -1, :]  # (batch, d_model)
        return self.score_head(last_hidden)  # (batch, 1)
