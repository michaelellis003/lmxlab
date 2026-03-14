"""Mixture of Experts feed-forward modules."""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.config import BlockConfig
from lmxlab.core.ffn import (
    FFNBase,
    GatedFFN,
    ReluSquaredFFN,
    ffn_registry,
)


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

        # Select top-k experts
        top_k_indices = mx.argpartition(
            -router_logits, kth=self.top_k, axis=-1
        )[:, :, : self.top_k]  # (B, T, top_k)

        # Softmax over top-k logits only (Mixtral convention)
        top_k_logits = mx.take_along_axis(
            router_logits, top_k_indices, axis=-1
        )
        top_k_weights = mx.softmax(top_k_logits, axis=-1)

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

        # Select top-k using biased logits
        top_k_indices = mx.argpartition(
            -biased_logits, kth=self.top_k, axis=-1
        )[:, :, : self.top_k]  # (B, T, top_k)

        # Softmax over top-k un-biased logits (DSV3)
        top_k_logits = mx.take_along_axis(
            router_logits, top_k_indices, axis=-1
        )
        top_k_weights = mx.softmax(top_k_logits, axis=-1)

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


@ffn_registry.register("latent_moe")
class LatentMoEFFN(FFNBase):
    """MoE with latent-space expert computation.

    Routes from full hidden dimension, then down-projects to
    latent space for expert computation. This enables many
    more experts (e.g. 512) at manageable compute cost.

    Architecture (per Nemotron 3 / LatentMoE paper)::

        router(x) -> top-k selection + sigmoid + normalize
        latent = down_proj(x)
        for each expert: expert_FFN(latent)  (in latent dim)
        up_proj(weighted_sum) -> d_model
        + shared_expert(x)

    Router operates on full hidden dim (not latent).
    Expert FFNs are non-gated relu2: down(relu2(up(x))).
    Shared expert is also non-gated relu2 at full dimension.

    Cross-references:
    - nvidia/Nemotron-H-8B modeling_nemotron_h.py
    - LatentMoE (Elango et al., 2025, arXiv:2601.18089)

    Args:
        config: Block configuration with moe_latent_size,
            n_experts, top_k_experts, moe_d_ff,
            shared_expert_d_ff.
    """

    def __init__(self, config: BlockConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.n_experts = config.n_experts or 8
        self.top_k = config.top_k_experts
        self.scaling_factor = config.moe_routed_scaling_factor
        self.moe_n_groups = config.moe_n_groups
        self.moe_topk_groups = config.moe_topk_groups
        latent = config.moe_latent_size or config.d_model // 4

        # Down-project to latent space for expert computation
        self.down_proj = nn.Linear(
            config.d_model,
            latent,
            bias=False,
        )

        # Router operates on full hidden dim (not latent).
        # Reference: NemotronHTopkRouter routes from
        # hidden_size, not moe_latent_size.
        self.router = nn.Linear(
            config.d_model,
            self.n_experts,
            bias=False,
        )

        # Score correction bias for aux-loss-free load
        # balancing. Added to sigmoid scores after scoring
        # (distinct from SharedExpertMoE expert_bias which
        # biases logits before selection).
        # Reference: DeepSeek-V3 section 3.1.2
        self.score_correction_bias = mx.zeros(
            (self.n_experts,),
        )

        # Expert FFNs: non-gated relu2 in latent space.
        # Reference: NemotronHMLP with is_expert=True uses
        # input_size=moe_latent_size, activation=relu2,
        # 2-layer FFN (not gated 3-layer).
        expert_d_ff = config.moe_d_ff or config.d_ff
        expert_cfg = BlockConfig(
            d_model=latent,
            d_ff=expert_d_ff,
            bias=False,
        )
        self.experts = [
            ReluSquaredFFN(expert_cfg) for _ in range(self.n_experts)
        ]

        # Up-project expert outputs from latent to d_model
        self.up_proj = nn.Linear(
            latent,
            config.d_model,
            bias=False,
        )

        # Shared expert: full dimension, non-gated relu2.
        # Reference: NemotronHMLP with is_expert=False.
        shared_d_ff = config.shared_expert_d_ff or config.d_ff
        shared_cfg = BlockConfig(
            d_model=config.d_model,
            d_ff=shared_d_ff,
            bias=False,
        )
        self.shared_expert = ReluSquaredFFN(shared_cfg)

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens and combine with shared expert.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        # Shared expert path (always active, on full dim)
        shared_out = self.shared_expert(x)

        # Route from full hidden dim (not latent)
        router_logits = self.router(x)  # (B, T, E)

        # Sigmoid scores + correction bias for load balancing.
        # Reference: DeepSeek-V3 score_correction_bias.
        scores = mx.sigmoid(router_logits)
        scores = scores + self.score_correction_bias

        # Grouped expert selection: pick top groups first,
        # then top-k from selected groups.
        if self.moe_n_groups > 1:
            top_k_indices = self._grouped_topk(scores)
        else:
            top_k_indices = mx.argpartition(
                -scores,
                kth=self.top_k,
                axis=-1,
            )[:, :, : self.top_k]

        # Gather corrected scores for selected experts
        top_k_scores = mx.take_along_axis(
            scores,
            top_k_indices,
            axis=-1,
        )
        # Normalize (Nemotron 3 convention)
        top_k_weights = top_k_scores / (
            mx.sum(top_k_scores, axis=-1, keepdims=True) + 1e-20
        )

        # Down-project to latent space for experts
        latent = self.down_proj(x)  # (B, T, latent)

        # Compute routed expert outputs
        routed_out = mx.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, :, k]
            weights = top_k_weights[:, :, k : k + 1]

            for e in range(self.n_experts):
                mask = expert_indices == e
                if not mx.any(mask).item():
                    continue
                # Expert operates in latent space
                expert_out = self.experts[e](latent)
                # Up-project to d_model
                expert_out = self.up_proj(expert_out)
                mask_expanded = mask[:, :, None]
                routed_out = routed_out + expert_out * weights * mask_expanded

        # Scale routed output
        routed_out = routed_out * self.scaling_factor

        return shared_out + routed_out

    def _grouped_topk(
        self,
        scores: mx.array,
    ) -> mx.array:
        """Select top-k experts using grouped selection.

        Divides experts into groups, picks top groups by sum
        of scores, then selects top-k from those groups.

        Args:
            scores: Sigmoid scores (B, T, n_experts).

        Returns:
            Indices of selected experts (B, T, top_k).
        """
        B, T, E = scores.shape
        G = self.moe_n_groups
        experts_per_group = E // G

        # Reshape to (B, T, G, E//G) and sum per group
        grouped = scores.reshape(B, T, G, experts_per_group)
        group_scores = mx.sum(grouped, axis=-1)  # (B, T, G)

        # Select top groups
        top_group_idx = mx.argpartition(
            -group_scores,
            kth=self.moe_topk_groups,
            axis=-1,
        )[:, :, : self.moe_topk_groups]  # (B, T, topk_g)

        # Build mask for selected groups
        group_mask = mx.zeros((B, T, G))
        for g in range(self.moe_topk_groups):
            gidx = top_group_idx[:, :, g]  # (B, T)
            for gi in range(G):
                group_mask = mx.where(
                    (gidx == gi)[:, :, None]
                    * (mx.arange(G) == gi)[None, None, :],
                    1.0,
                    group_mask,
                )

        # Expand mask to expert level: (B, T, E)
        expert_mask = mx.repeat(
            group_mask,
            experts_per_group,
            axis=-1,
        )

        # Mask out non-selected groups
        masked_scores = mx.where(
            expert_mask > 0,
            scores,
            -1e9,
        )

        # Select top-k from remaining experts
        top_k_indices = mx.argpartition(
            -masked_scores,
            kth=self.top_k,
            axis=-1,
        )[:, :, : self.top_k]

        return top_k_indices
