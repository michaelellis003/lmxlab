"""Multi-Token Prediction (MTP) training module.

MTP trains the model to predict multiple future tokens at each
position, not just the next token. This provides richer training
signal and can enable speculative decoding at inference time.

Architecture:
    For each prediction depth k (1..n_predict):
    - Shared transformer backbone produces hidden states h
    - MTP head k projects [h_t; embed(y_{t+k-1})] -> h'
    - h' is projected to logits for token t+k

The backbone is the standard LanguageModel. MTP heads are
lightweight projection layers trained alongside the backbone.

Loss = main_loss + sum(mtp_weight * mtp_loss_k) for k=1..n_predict

Reference: DeepSeek-V3 (arxiv.org/abs/2501.12948), Meta (2024)
"""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.models.base import LanguageModel


class MTPHead(nn.Module):
    """Single multi-token prediction head.

    Takes hidden states and previous target embeddings,
    produces logits for a future token position.

    Args:
        d_model: Hidden dimension.
        vocab_size: Vocabulary size.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # Project concatenated [hidden; embedding] -> d_model
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, h: mx.array, prev_embed: mx.array) -> mx.array:
        """Predict future token logits.

        Args:
            h: Hidden states (batch, seq_len, d_model).
            prev_embed: Embeddings of previous target tokens
                (batch, seq_len, d_model).

        Returns:
            Logits (batch, seq_len, vocab_size).
        """
        combined = mx.concatenate([h, prev_embed], axis=-1)
        projected = self.norm(self.proj(combined))
        return self.head(projected)


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction wrapper around a LanguageModel.

    Adds n_predict auxiliary prediction heads that predict
    future tokens at each position.

    Args:
        model: Base language model.
        n_predict: Number of future tokens to predict (1-4 typical).
        mtp_weight: Weight for auxiliary MTP losses.
    """

    def __init__(
        self,
        model: LanguageModel,
        n_predict: int = 2,
        mtp_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_predict = n_predict
        self.mtp_weight = mtp_weight

        d_model = model.config.block.d_model
        vocab_size = model.config.vocab_size

        self.mtp_heads = [
            MTPHead(d_model, vocab_size) for _ in range(n_predict)
        ]

    def __call__(
        self,
        x: mx.array,
        targets: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Forward pass with multi-token prediction.

        Args:
            x: Input token IDs (batch, seq_len).
            targets: Target token IDs (batch, seq_len).
                Must have length >= seq_len + n_predict.

        Returns:
            Tuple of (main_logits, loss_dict) where loss_dict
            contains 'main_loss', 'mtp_loss', and 'total_loss'.
        """
        # Standard forward pass
        logits, _ = self.model(x)

        # Main loss (next token prediction)
        main_logits = logits[:, :-1, :]
        main_targets = targets[:, : main_logits.shape[1]]
        main_loss = nn.losses.cross_entropy(
            main_logits.reshape(-1, main_logits.shape[-1]),
            main_targets.reshape(-1),
            reduction="mean",
        )

        # Get hidden states from the final norm
        # (before the output head projection)
        h = self.model.embed(x)
        for block in self.model.blocks:
            h, _ = block(h)
        h = self.model.final_norm(h)

        # MTP losses for each prediction depth
        mtp_losses = []
        for k, head in enumerate(self.mtp_heads, start=1):
            # For depth k, predict token at position t+k
            # using hidden state at position t and
            # embedding of token at position t+k-1
            if targets.shape[1] <= k:
                continue

            # Hidden states for positions that have
            # enough future context
            h_slice = h[:, :-k, :]

            # Previous target embeddings (tokens at t+k-1)
            prev_toks = targets[:, k - 1 : -1]
            if prev_toks.shape[1] > h_slice.shape[1]:
                prev_toks = prev_toks[:, : h_slice.shape[1]]
            prev_embed = self.model.embed(prev_toks)

            # Predict
            mtp_logits = head(h_slice, prev_embed)
            mtp_targets = targets[:, k : k + mtp_logits.shape[1]]

            if mtp_targets.shape[1] > 0:
                mtp_loss = nn.losses.cross_entropy(
                    mtp_logits.reshape(-1, mtp_logits.shape[-1]),
                    mtp_targets.reshape(-1),
                    reduction="mean",
                )
                mtp_losses.append(mtp_loss)

        # Combine losses
        if mtp_losses:
            avg_mtp_loss = sum(mtp_losses) / len(mtp_losses)
        else:
            avg_mtp_loss = mx.array(0.0)

        total_loss = main_loss + self.mtp_weight * avg_mtp_loss

        losses = {
            "main_loss": main_loss,
            "mtp_loss": avg_mtp_loss,
            "total_loss": total_loss,
        }

        return logits, losses
