"""Mixture of Experts feed-forward module."""

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.core.config import BlockConfig
from lmt_metal.core.ffn import GatedFFN


class MoEFFN(nn.Module):
    """Mixture of Experts feed-forward network.

    Routes each token to the top-k experts via a learned
    router. Each expert is a GatedFFN (SwiGLU).

    Args:
        config: Block configuration.
        n_experts: Total number of experts.
        top_k: Number of experts per token.
    """

    def __init__(
        self,
        config: BlockConfig,
        n_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: projects hidden states to expert logits
        self.router = nn.Linear(config.d_model, n_experts, bias=False)

        # Expert FFNs
        self.experts = [GatedFFN(config) for _ in range(n_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to top-k experts and combine outputs.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        # Compute routing weights
        router_logits = self.router(x)  # (B, T, n_experts)
        router_probs = mx.softmax(router_logits, axis=-1)

        # Select top-k experts
        top_k_indices = mx.argpartition(
            -router_logits, kth=self.top_k, axis=-1
        )[:, :, : self.top_k]  # (B, T, top_k)
        top_k_weights = mx.take_along_axis(
            router_probs, top_k_indices, axis=-1
        )

        # Normalize weights
        top_k_weights = top_k_weights / mx.sum(
            top_k_weights, axis=-1, keepdims=True
        )

        # Compute expert outputs and combine
        output = mx.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, :, k]  # (B, T)
            weights = top_k_weights[:, :, k : k + 1]  # (B, T, 1)

            # Process each expert
            for e in range(self.n_experts):
                mask = expert_indices == e  # (B, T)
                if not mx.any(mask).item():
                    continue
                expert_out = self.experts[e](x)  # (B, T, D)
                mask_expanded = mask[:, :, None]  # (B, T, 1)
                output = output + expert_out * weights * mask_expanded

        return output
