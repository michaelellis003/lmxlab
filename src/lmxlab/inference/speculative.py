"""Speculative decoding for faster inference.

Uses a small draft model to propose tokens, verified by the
target model in a single forward pass. Especially interesting
on unified memory where both models share the same memory pool.
"""

import mlx.core as mx

from lmxlab.models.base import LanguageModel


def speculative_decode(
    target_model: LanguageModel,
    draft_model: LanguageModel,
    prompt: mx.array,
    max_tokens: int = 100,
    draft_tokens: int = 4,
    temperature: float = 0.0,
) -> tuple[mx.array, dict[str, float]]:
    """Generate tokens using speculative decoding (greedy).

    Draft model proposes tokens, target model verifies in one
    forward pass. Accepted tokens are kept; on mismatch, use
    the target model's token and restart drafting.

    This is especially efficient on Apple Silicon where both
    models share unified memory -- no data transfer overhead.

    Args:
        target_model: Large target model.
        draft_model: Small draft model.
        prompt: Token IDs (1, prompt_len).
        max_tokens: Maximum new tokens.
        draft_tokens: Tokens to draft per step.
        temperature: Sampling temperature (only 0.0 supported).

    Returns:
        Tuple of (tokens, stats_dict).
    """
    tokens = list(prompt[0].tolist())
    prompt_len = len(tokens)
    n_accepted = 0
    n_drafted = 0

    while len(tokens) - prompt_len < max_tokens:
        remaining = max_tokens - (len(tokens) - prompt_len)
        n_draft = min(draft_tokens, remaining)

        # Draft: generate n_draft tokens with small model
        drafted: list[int] = []
        for _ in range(n_draft):
            d_input = mx.array([tokens + drafted])
            d_logits, _ = draft_model(d_input)
            mx.eval(d_logits)
            next_tok = mx.argmax(d_logits[:, -1, :], axis=-1).item()
            drafted.append(next_tok)
        n_drafted += len(drafted)

        # Verify: run target model on all tokens + drafted
        verify_seq = tokens + drafted
        t_input = mx.array([verify_seq])
        t_logits, _ = target_model(t_input)
        mx.eval(t_logits)

        # Check each drafted token against target
        accepted = 0
        for i, draft_tok in enumerate(drafted):
            # Target's prediction at position before this token
            pos = len(tokens) + i - 1
            target_tok = mx.argmax(t_logits[:, pos, :], axis=-1).item()

            if target_tok == draft_tok:
                accepted += 1
            else:
                # Use target's token and stop
                tokens.append(target_tok)
                n_accepted += accepted + 1
                break
        else:
            # All drafted tokens accepted
            tokens.extend(drafted)
            n_accepted += accepted

            # Also get the next token from target
            if len(tokens) - prompt_len < max_tokens:
                last_pos = len(tokens) - 1
                next_tok = mx.argmax(t_logits[:, last_pos, :], axis=-1).item()
                tokens.append(next_tok)
                n_accepted += 1

    # Truncate to exact length
    tokens = tokens[: prompt_len + max_tokens]
    result = mx.array([tokens])

    stats = {
        "acceptance_rate": (n_accepted / max(n_drafted, 1)),
        "total_drafted": n_drafted,
        "total_accepted": n_accepted,
    }
    return result, stats
