"""Unfused attention weight extraction for analysis.

The fused ``mx.fast.scaled_dot_product_attention`` never
exposes attention weights. This module re-computes them
manually using Q, K projections from each attention layer.

Performance note: ~10-20% slower than the fused path.
Use only for analysis, not training.

Example:
    >>> model = LanguageModel(gpt_tiny())
    >>> tokens = mx.array([[1, 2, 3, 4]])
    >>> maps = extract_attention_maps(model, tokens)
    >>> maps["layer_0"].shape  # (batch, heads, seq, seq)
    (1, 2, 4, 4)
"""

from __future__ import annotations

import math

import mlx.core as mx

from lmxlab.models.base import LanguageModel, _create_causal_mask


def extract_attention_maps(
    model: LanguageModel,
    tokens: mx.array,
) -> dict[str, mx.array]:
    """Extract per-layer attention weight matrices.

    Runs Q, K projections manually and computes
    ``softmax(Q @ K^T / sqrt(d_head) + mask)`` for each
    attention layer. Non-attention layers (e.g. Mamba) are
    skipped.

    Args:
        model: Language model (must have attention blocks
            with ``q_proj`` and ``k_proj`` attributes).
        tokens: Input token IDs of shape (batch, seq_len).

    Returns:
        Dict mapping ``"layer_N"`` to attention weights of
        shape (batch, n_heads, seq_len, seq_len).
    """
    config = model.config

    # Embed + positional encoding (same as model forward)
    h = model.embed_dropout(model.embed(tokens))
    if model._sinusoidal:
        h = model.blocks[0].position(h)

    T = h.shape[1]
    mask = _create_causal_mask(T)

    maps: dict[str, mx.array] = {}

    for i, block in enumerate(model.blocks):
        cfg = config.get_block_config(i)
        attn = block.attention

        # Only extract from attention layers with Q/K projections
        has_qk = hasattr(attn, "q_proj") and hasattr(attn, "k_proj")

        if has_qk:
            # Pre-norm: normalize before attention
            h_normed = block.attn_norm(h) if block.config.pre_norm else h

            B, L, _ = h_normed.shape
            n_heads = cfg.n_heads
            head_dim = cfg.head_dim
            n_kv = cfg.effective_n_kv_heads

            q = attn.q_proj(h_normed)
            k = attn.k_proj(h_normed)

            q = q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, n_kv, head_dim).transpose(0, 2, 1, 3)

            # Apply QK-norm if present
            if hasattr(attn, "_apply_qk_norm"):
                q, k = attn._apply_qk_norm(q, k)

            # Apply RoPE if present
            rope = block._rope
            if rope is not None:
                q, k = rope(q, k, offset=0)

            # Expand KV heads for GQA
            if n_kv < n_heads:
                repeats = n_heads // n_kv
                k = mx.repeat(k, repeats, axis=1)

            # Manual attention: Q @ K^T / sqrt(d)
            scale = 1.0 / math.sqrt(head_dim)
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale

            # Apply causal mask
            if mask is not None:
                scores = scores + mask

            weights = mx.softmax(scores, axis=-1)
            maps[f"layer_{i}"] = weights

        # Run the actual block to update h for next layer
        h, _ = block(h, mask=mask, cache=None)

    mx.eval(maps)
    return maps
