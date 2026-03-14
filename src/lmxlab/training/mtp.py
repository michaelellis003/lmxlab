"""Multi-Token Prediction (MTP) training module.

MTP trains the model to predict multiple future tokens at each
position, not just the next token. This provides richer training
signal and can enable speculative decoding at inference time.

Architecture (DeepSeek-V3 sequential MTP):
    For each prediction depth k (1..n_predict):
    - MTPHead takes hidden_t and embed(target_{t+k-1})
    - Concatenates [norm(h), norm(embed)], projects to d_model
    - Passes through a transformer block
    - Shares lm_head with base model for logit projection

The backbone is the standard LanguageModel. MTP heads are
trained alongside the backbone. During inference, only the
base model is used.

Loss = main_loss + mtp_lambda * mean(mtp_loss_k)

Reference: DeepSeek-V3 (arxiv.org/abs/2412.19437),
    Meta (arxiv.org/abs/2404.19737)
"""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.block import ConfigurableBlock
from lmxlab.core.config import BlockConfig
from lmxlab.models.base import LanguageModel


class MTPHead(nn.Module):
    """Single multi-token prediction head.

    Takes hidden states and previous target embeddings,
    normalizes both, concatenates, projects back to d_model,
    then runs through a transformer block.

    Args:
        d_model: Hidden dimension.
        block_config: Block configuration for the MTP block.
    """

    def __init__(
        self,
        d_model: int,
        block_config: BlockConfig,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.RMSNorm(d_model)
        self.embed_norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(
            2 * d_model,
            d_model,
            bias=False,
        )
        self.block = ConfigurableBlock(block_config)

    def __call__(
        self,
        h: mx.array,
        prev_embed: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Produce hidden states for future token prediction.

        Args:
            h: Hidden states (batch, seq_len, d_model).
            prev_embed: Embeddings of previous target tokens
                (batch, seq_len, d_model).
            mask: Optional causal mask.

        Returns:
            Hidden states (batch, seq_len, d_model).
        """
        combined = mx.concatenate(
            [
                self.hidden_norm(h),
                self.embed_norm(prev_embed),
            ],
            axis=-1,
        )
        projected = self.proj(combined)
        out, _ = self.block(projected, mask=mask)
        return out


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction wrapper around a LanguageModel.

    Adds n_predict auxiliary prediction heads that predict
    future tokens at each position. Shares the base model's
    lm_head for logit projection.

    Training-only module. At inference time, use the base
    model directly.

    Args:
        model: Base language model.
        n_predict: Number of future tokens to predict.
        mtp_weight: Weight for auxiliary MTP losses.
        block_config: Block config for MTP heads. If None,
            uses the base model's block config.
    """

    def __init__(
        self,
        model: LanguageModel,
        n_predict: int = 2,
        mtp_weight: float = 0.3,
        block_config: BlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_predict = n_predict
        self.mtp_weight = mtp_weight

        d_model = model.config.block.d_model

        # Default MTP block: lightweight attention block
        if block_config is None:
            block_config = BlockConfig(
                attention="mha",
                ffn="standard",
                norm="rms_norm",
                position="none",
                d_model=d_model,
                n_heads=max(1, model.config.block.n_heads // 2),
                d_ff=d_model * 2,
                bias=False,
                pre_norm=True,
            )

        self.mtp_heads = [
            MTPHead(d_model, block_config) for _ in range(n_predict)
        ]

    def _project_logits(self, h: mx.array) -> mx.array:
        """Project hidden states to logits using shared head.

        Args:
            h: Hidden states (batch, seq_len, d_model).

        Returns:
            Logits (batch, seq_len, vocab_size).
        """
        if self.model.config.tie_embeddings:
            return h @ self.model.embed.weight.T
        return self.model.head(h)

    def __call__(
        self,
        x: mx.array,
        targets: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Forward pass with multi-token prediction.

        Args:
            x: Input token IDs (batch, seq_len).
            targets: Target token IDs (batch, seq_len).
                For MTP depth k, predicts target at t+k.

        Returns:
            Tuple of (main_logits, loss_dict) where loss_dict
            contains 'main_loss', 'mtp_loss', 'total_loss'.
        """
        # Forward with hidden states
        logits, _, hidden = self.model(
            x,
            return_hidden=True,
        )

        # Main loss (next token prediction)
        main_logits = logits[:, :-1, :]
        main_targets = targets[:, : main_logits.shape[1]]
        main_loss = nn.losses.cross_entropy(
            main_logits.reshape(-1, main_logits.shape[-1]),
            main_targets.reshape(-1),
            reduction="mean",
        )

        # MTP losses for each prediction depth
        mtp_losses = []
        h = hidden
        for k, head in enumerate(self.mtp_heads, start=1):
            if targets.shape[1] <= k:
                continue

            # Hidden states for positions with enough future
            h_slice = h[:, :-k, :]

            # Embeddings of tokens at position t+k-1
            prev_toks = targets[:, k - 1 : -1]
            if prev_toks.shape[1] > h_slice.shape[1]:
                prev_toks = prev_toks[:, : h_slice.shape[1]]
            prev_embed = self.model.embed(prev_toks)

            # MTP head produces new hidden states
            h_mtp = head(h_slice, prev_embed)

            # Project to logits using shared lm_head
            mtp_logits = self._project_logits(h_mtp)
            mtp_targets = targets[:, k : k + mtp_logits.shape[1]]

            if mtp_targets.shape[1] > 0:
                mtp_loss = nn.losses.cross_entropy(
                    mtp_logits.reshape(
                        -1,
                        mtp_logits.shape[-1],
                    ),
                    mtp_targets.reshape(-1),
                    reduction="mean",
                )
                mtp_losses.append(mtp_loss)

            # Chain: next head uses this head's output
            h = h_mtp

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
