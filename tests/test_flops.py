"""Tests for FLOP estimation and FLOPCounter callback."""

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.experiments.flops import (
    estimate_flops_per_step,
    estimate_flops_per_token,
)
from lmxlab.training.callbacks import FLOPCounter


def _gpt_config():
    """Small GPT-style config for testing."""
    return ModelConfig(
        block=BlockConfig(
            d_model=256,
            n_heads=8,
            d_ff=1024,
            ffn='standard',
            max_seq_len=256,
        ),
        n_layers=6,
        vocab_size=32000,
    )


def _llama_config():
    """Small LLaMA-style config for testing."""
    return ModelConfig(
        block=BlockConfig(
            d_model=256,
            n_heads=8,
            n_kv_heads=4,
            d_ff=683,
            ffn='gated',
            max_seq_len=256,
        ),
        n_layers=6,
        vocab_size=32000,
    )


def test_standard_ffn_flops():
    """Verify analytical formula for standard FFN config."""
    cfg = _gpt_config()
    flops = estimate_flops_per_token(cfg)
    assert flops > 0

    # Manually compute expected
    d, h, hd = 256, 8, 32
    seq, d_ff = 256, 1024
    kv_dim = 8 * 32  # MHA: kv_heads == n_heads

    qkvo = 2 * d * d + 2 * d * kv_dim + 2 * d * kv_dim + 2 * d * d
    attn = 2 * h * seq * hd + 2 * h * seq * hd
    ffn = 2 * 2 * d * d_ff
    block = qkvo + attn + ffn
    expected = block * 6 + 2 * d * 32000

    assert flops == expected


def test_gated_ffn_more_flops():
    """Gated FFN has more FLOPs than standard at same d_ff."""
    standard = ModelConfig(
        block=BlockConfig(
            d_model=256, n_heads=8, d_ff=1024,
            ffn='standard', max_seq_len=256,
        ),
        n_layers=6,
        vocab_size=32000,
    )
    gated = ModelConfig(
        block=BlockConfig(
            d_model=256, n_heads=8, d_ff=1024,
            ffn='gated', max_seq_len=256,
        ),
        n_layers=6,
        vocab_size=32000,
    )
    assert estimate_flops_per_token(gated) > \
        estimate_flops_per_token(standard)


def test_gqa_fewer_flops():
    """GQA (fewer KV heads) has fewer attention FLOPs than MHA."""
    mha = ModelConfig(
        block=BlockConfig(
            d_model=256, n_heads=8, d_ff=1024,
            ffn='standard', max_seq_len=256,
        ),
        n_layers=6,
        vocab_size=32000,
    )
    gqa = ModelConfig(
        block=BlockConfig(
            d_model=256, n_heads=8, n_kv_heads=2, d_ff=1024,
            ffn='standard', max_seq_len=256,
        ),
        n_layers=6,
        vocab_size=32000,
    )
    assert estimate_flops_per_token(gqa) < \
        estimate_flops_per_token(mha)


def test_step_scales_with_batch():
    """Step FLOPs scale linearly with batch_size * seq_len."""
    cfg = _gpt_config()
    f1 = estimate_flops_per_step(cfg, batch_size=16, seq_len=128)
    f2 = estimate_flops_per_step(cfg, batch_size=32, seq_len=128)
    f4 = estimate_flops_per_step(cfg, batch_size=16, seq_len=256)

    assert f2 == 2 * f1  # double batch
    assert f4 == 2 * f1  # double seq_len


def test_6nd_approximation():
    """estimate_flops_per_step within 30% of 6*N*D for GPT."""
    cfg = _gpt_config()
    batch, seq = 32, 256

    # Count parameters (rough: embeddings + layers)
    d, d_ff, vocab = 256, 1024, 32000
    n_layers = 6
    # Embedding
    n_params = vocab * d
    # Per layer: attn (4 * d^2) + ffn (2 * d * d_ff)
    n_params += n_layers * (4 * d * d + 2 * d * d_ff)

    tokens = batch * seq
    approx_6nd = 6 * n_params * tokens

    actual = estimate_flops_per_step(cfg, batch, seq)
    ratio = actual / approx_6nd
    assert 0.7 < ratio < 1.3, f'ratio={ratio:.2f}'


def test_flop_counter_accumulates():
    """FLOPCounter callback accumulates correctly."""
    counter = FLOPCounter(flops_per_step=1_000_000, log_interval=100)
    counter.on_train_begin(None)

    metrics: dict = {}
    for step in range(1, 11):
        counter.on_step_end(step, metrics)

    assert counter.total_flops == 10_000_000
    assert metrics.get('total_flops') == 10_000_000
    assert not counter.should_stop


def test_flop_counter_stops_at_budget():
    """FLOPCounter sets should_stop when budget exceeded."""
    counter = FLOPCounter(
        flops_per_step=1e12,
        flop_budget=5e12,
        log_interval=100,
    )
    counter.on_train_begin(None)

    metrics: dict = {}
    for step in range(1, 10):
        counter.on_step_end(step, metrics)

    # After 5 steps: 5e12, should trigger stop
    assert counter.should_stop
    assert counter.total_flops >= 5e12
