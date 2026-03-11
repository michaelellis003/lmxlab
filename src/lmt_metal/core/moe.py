"""Mixture of Experts feed-forward module."""

import mlx.core as mx
import mlx.nn as nn

from lmt_metal.core.config import BlockConfig
from lmt_metal.core.ffn import GatedFFN, ffn_registry


@ffn_registry.register("moe")
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
        n_experts: int | None = None,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts or config.n_experts or 8
        self.top_k = top_k or config.top_k_experts

        # Router: projects hidden states to expert logits
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)

        # Expert FFNs
        self.experts = [GatedFFN(config) for _ in range(self.n_experts)]

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


@ffn_registry.register("shared_moe")
class SharedExpertMoEFFN(nn.Module):
    """MoE with shared experts and bias-based load balancing.

    Combines a set of always-active shared experts with top-k
    routed experts. Uses aux-loss-free load balancing: a
    learnable bias is added to router logits for expert
    selection, but the original un-biased scores are used
    for gating weights.

    Output = shared_experts(x) + routed_experts_output(x)

    Args:
        config: Block configuration.
        n_experts: Number of routed experts.
        top_k: Number of routed experts per token.
        n_shared: Number of shared (always-active) experts.
    """

    def __init__(
        self,
        config: BlockConfig,
        n_experts: int | None = None,
        top_k: int | None = None,
        n_shared: int | None = None,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts or config.n_experts or 8
        self.top_k = top_k or config.top_k_experts
        self.n_shared = n_shared or config.n_shared_experts or 1

        # Router: projects hidden states to expert logits
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)

        # Learnable bias for aux-loss-free load balancing.
        # Added to logits for selection only, not for weights.
        self.expert_bias = mx.zeros((self.n_experts,))

        # Routed experts
        self.experts = [GatedFFN(config) for _ in range(self.n_experts)]

        # Shared experts (always active, not gated)
        self.shared_experts = [GatedFFN(config) for _ in range(self.n_shared)]

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens and combine with shared expert output.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        # --- Shared expert path (always active) ---
        shared_out = self.shared_experts[0](x)
        for i in range(1, self.n_shared):
            shared_out = shared_out + self.shared_experts[i](x)

        # --- Routed expert path ---
        router_logits = self.router(x)  # (B, T, E)

        # Bias-based selection: add bias for top-k picking
        biased_logits = router_logits + self.expert_bias

        # Use un-biased logits for gating weights
        router_probs = mx.softmax(router_logits, axis=-1)

        # Select top-k using biased logits
        top_k_indices = mx.argpartition(
            -biased_logits, kth=self.top_k, axis=-1
        )[:, :, : self.top_k]  # (B, T, top_k)
        top_k_weights = mx.take_along_axis(
            router_probs, top_k_indices, axis=-1
        )

        # Normalize weights
        top_k_weights = top_k_weights / mx.sum(
            top_k_weights, axis=-1, keepdims=True
        )

        # Compute routed expert outputs and combine
        routed_out = mx.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, :, k]
            weights = top_k_weights[:, :, k : k + 1]

            for e in range(self.n_experts):
                mask = expert_indices == e
                if not mx.any(mask).item():
                    continue
                expert_out = self.experts[e](x)
                mask_expanded = mask[:, :, None]
                routed_out = routed_out + expert_out * weights * mask_expanded

        return shared_out + routed_out
