"""Analytical FLOP estimation for architecture comparison.

Provides pure-analytical FLOP counting (no GPU instrumentation)
for fair architecture comparisons under compute-matched budgets.

Follows the Megatron-LM / Narayanan et al. (2021,
arXiv:2104.04473) methodology:
each multiply-accumulate counts as 2 FLOPs. Intentionally omits
softmax, layer norms, residual adds, dropout, and activation
functions — these are O(d) or O(seq) per layer, dominated by
the O(d^2) and O(d * d_ff) matmul terms.
"""

from lmxlab.core.config import ModelConfig


def _block_flops(
    model_config: ModelConfig,
    layer_idx: int,
) -> int:
    """Compute forward-pass FLOPs for a single block.

    Args:
        model_config: Model configuration.
        layer_idx: Layer index (for per-layer overrides).

    Returns:
        FLOPs for one token through this block.
    """
    block = model_config.get_block_config(layer_idx)
    d = block.d_model
    h = block.n_heads
    hd = block.head_dim
    kv_h = block.effective_n_kv_heads
    kv_dim = kv_h * hd
    seq = block.max_seq_len
    d_ff = block.d_ff

    # Attention projections
    qkvo = (
        2 * d * d  # Q projection
        + 2 * d * kv_dim  # K projection
        + 2 * d * kv_dim  # V projection
        + 2 * d * d  # output projection
    )
    # Attention scores and weighted sum
    attn_ops = (
        2 * h * seq * hd  # Q·K^T
        + 2 * h * seq * hd  # scores·V
    )
    # FFN: gated has 3 projections, standard has 2
    ffn_mult = 3 if block.ffn == "gated" else 2
    ffn_flops = 2 * ffn_mult * d * d_ff

    return qkvo + attn_ops + ffn_flops


def estimate_flops_per_token(model_config: ModelConfig) -> int:
    """Estimate FLOPs for one forward pass per token.

    Counts multiply-accumulate as 2 FLOPs. Covers attention
    projections, attention scores, FFN, and unembedding.
    Embedding lookup is not counted (table index, not matmul).

    Supports per-layer block overrides via ``block_configs``.

    Args:
        model_config: Model configuration.

    Returns:
        Total FLOPs per token (forward pass only).
    """
    # Sum across layers (supports heterogeneous blocks)
    total = sum(
        _block_flops(model_config, i) for i in range(model_config.n_layers)
    )

    # Unembedding (logits projection — matmul even if tied)
    d = model_config.block.d_model
    total += 2 * d * model_config.vocab_size

    return total


def estimate_flops_per_step(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
) -> int:
    """Estimate FLOPs for one training step.

    Multiplies per-token FLOPs by batch_size * seq_len * 3
    (forward = 1x, backward ~ 2x forward).

    Args:
        model_config: Model configuration.
        batch_size: Training batch size.
        seq_len: Sequence length.

    Returns:
        Total FLOPs per training step.
    """
    return estimate_flops_per_token(model_config) * batch_size * seq_len * 3
