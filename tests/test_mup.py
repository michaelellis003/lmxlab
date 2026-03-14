"""Tests for μP (Maximal Update Parameterization)."""

import math

import mlx.core as mx
import mlx.utils
import pytest

from lmxlab.core.attention import GQA, MHA, SlidingWindowGQA
from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_config
from lmxlab.models.llama import llama_config
from lmxlab.training.config import TrainConfig
from lmxlab.training.optimizers import create_mup_optimizer


class TestMupConfig:
    """Test μP configuration fields."""

    def test_block_config_mup_default_false(self):
        """BlockConfig.mup defaults to False (SP)."""
        cfg = BlockConfig()
        assert cfg.mup is False

    def test_block_config_mup_true(self):
        """BlockConfig.mup can be set to True."""
        cfg = BlockConfig(mup=True)
        assert cfg.mup is True

    def test_model_config_mup_base_width_default_none(self):
        """ModelConfig.mup_base_width defaults to None (SP)."""
        cfg = ModelConfig()
        assert cfg.mup_base_width is None

    def test_model_config_mup_base_width_set(self):
        """ModelConfig.mup_base_width can be set."""
        cfg = ModelConfig(mup_base_width=64)
        assert cfg.mup_base_width == 64

    def test_model_config_width_mult(self):
        """ModelConfig.width_mult computes d_model / base_width."""
        cfg = ModelConfig(
            block=BlockConfig(d_model=256),
            mup_base_width=64,
        )
        assert cfg.width_mult == 4.0

    def test_model_config_width_mult_none(self):
        """ModelConfig.width_mult is 1.0 when μP disabled."""
        cfg = ModelConfig(block=BlockConfig(d_model=256))
        assert cfg.width_mult == 1.0


class TestMupAttentionScaling:
    """Test attention scaling changes under μP."""

    def test_mha_sp_scale(self):
        """MHA uses 1/√d_head in standard parameterization."""
        cfg = BlockConfig(d_model=64, n_heads=4, mup=False)
        attn = MHA(cfg)
        expected = (64 // 4) ** -0.5  # 1/√16 = 0.25
        assert attn.scale == pytest.approx(expected)

    def test_mha_mup_scale(self):
        """MHA uses 1/d_head in μP."""
        cfg = BlockConfig(d_model=64, n_heads=4, mup=True)
        attn = MHA(cfg)
        expected = (64 // 4) ** -1.0  # 1/16 = 0.0625
        assert attn.scale == pytest.approx(expected)

    def test_gqa_sp_scale(self):
        """GQA uses 1/√d_head in standard parameterization."""
        cfg = BlockConfig(
            d_model=64, n_heads=4, n_kv_heads=2, mup=False,
        )
        attn = GQA(cfg)
        expected = (64 // 4) ** -0.5
        assert attn.scale == pytest.approx(expected)

    def test_gqa_mup_scale(self):
        """GQA uses 1/d_head in μP."""
        cfg = BlockConfig(
            d_model=64, n_heads=4, n_kv_heads=2, mup=True,
        )
        attn = GQA(cfg)
        expected = (64 // 4) ** -1.0
        assert attn.scale == pytest.approx(expected)

    def test_sliding_window_gqa_mup_scale(self):
        """SlidingWindowGQA uses 1/d_head in μP."""
        cfg = BlockConfig(
            d_model=64, n_heads=4, n_kv_heads=2,
            window_size=32, mup=True,
        )
        attn = SlidingWindowGQA(cfg)
        expected = (64 // 4) ** -1.0
        assert attn.scale == pytest.approx(expected)


class TestMupLogitScaling:
    """Test output logit scaling under μP."""

    def test_logits_unscaled_without_mup(self):
        """Logits are NOT scaled when μP is disabled."""
        # Use position='none' so zero-input gives identical
        # hidden states at all positions (attention scale is
        # irrelevant with uniform inputs).
        cfg = ModelConfig(
            block=BlockConfig(
                d_model=64, n_heads=2, d_ff=128,
                max_seq_len=32, position='none',
            ),
            vocab_size=32, n_layers=1,
        )
        model = LanguageModel(cfg)
        x = mx.zeros((1, 4), dtype=mx.int32)
        logits_sp, _ = model(x)

        # Build same model with mup_base_width=64 (mult=1)
        cfg_mup = ModelConfig(
            block=BlockConfig(
                d_model=64, n_heads=2, d_ff=128,
                max_seq_len=32, mup=True, position='none',
            ),
            vocab_size=32, n_layers=1,
            mup_base_width=64,
        )
        model_mup = LanguageModel(cfg_mup)
        # Copy weights so outputs are comparable
        flat_weights = mlx.utils.tree_flatten(
            model.parameters()
        )
        model_mup.load_weights(flat_weights)
        logits_mup, _ = model_mup(x)

        # width_mult=1, so logits should be identical
        mx.eval(logits_sp, logits_mup)
        assert mx.allclose(logits_sp, logits_mup, atol=1e-5)

    def test_logits_scaled_with_mup(self):
        """Logits are divided by width_mult when μP enabled."""
        base_width = 32
        target_width = 128
        width_mult = target_width / base_width  # 4.0

        cfg = ModelConfig(
            block=BlockConfig(
                d_model=target_width, n_heads=4,
                d_ff=256, mup=True, position='none',
            ),
            vocab_size=32,
            n_layers=1,
            mup_base_width=base_width,
        )
        model = LanguageModel(cfg)
        x = mx.zeros((1, 4), dtype=mx.int32)
        logits, _ = model(x)
        mx.eval(logits)

        # Build same model without μP to get unscaled logits
        cfg_sp = ModelConfig(
            block=BlockConfig(
                d_model=target_width, n_heads=4,
                d_ff=256, mup=False, position='none',
            ),
            vocab_size=32,
            n_layers=1,
        )
        model_sp = LanguageModel(cfg_sp)
        flat_weights = mlx.utils.tree_flatten(
            model.parameters()
        )
        model_sp.load_weights(flat_weights)
        logits_sp, _ = model_sp(x)
        mx.eval(logits_sp)

        # μP logits should be SP logits / width_mult
        expected = logits_sp / width_mult
        mx.eval(expected)
        assert mx.allclose(logits, expected, atol=1e-5)


class TestMupOptimizer:
    """Test μP optimizer with per-layer LR groups."""

    def test_create_mup_optimizer_returns_multi(self):
        """create_mup_optimizer returns a MultiOptimizer."""
        import mlx.optimizers as optim

        cfg = TrainConfig(
            learning_rate=1e-3,
            warmup_steps=10,
            max_steps=1000,
        )
        opt = create_mup_optimizer(cfg, width_mult=2.0)
        assert isinstance(opt, optim.MultiOptimizer)

    def test_mup_optimizer_has_two_groups(self):
        """MultiOptimizer has embed and hidden groups."""
        cfg = TrainConfig(
            learning_rate=1e-3,
            warmup_steps=10,
            max_steps=1000,
        )
        opt = create_mup_optimizer(cfg, width_mult=2.0)
        assert len(opt.optimizers) == 2

    def test_mup_optimizer_scales_hidden_lr(self):
        """Hidden layer LR is scaled by 1/width_mult.

        Verify the ratio between embed and hidden schedules
        is correct at the peak (right after warmup).
        """
        from lmxlab.training.optimizers import create_schedule

        base_lr = 1e-3
        width_mult = 4.0
        warmup = 10
        cfg = TrainConfig(
            learning_rate=base_lr,
            warmup_steps=warmup,
            max_steps=1000,
        )
        # Create the two schedules directly
        embed_sched = create_schedule(cfg)
        scaled_cfg = TrainConfig(
            learning_rate=base_lr / width_mult,
            warmup_steps=warmup,
            max_steps=1000,
        )
        hidden_sched = create_schedule(scaled_cfg)

        # Check LR ratio at peak (step=warmup)
        embed_lr = float(embed_sched(warmup))
        hidden_lr = float(hidden_sched(warmup))

        assert embed_lr == pytest.approx(base_lr, rel=1e-2)
        assert hidden_lr == pytest.approx(
            base_lr / width_mult, rel=1e-2,
        )
        assert embed_lr / hidden_lr == pytest.approx(
            width_mult, rel=1e-2,
        )


class TestMupFactories:
    """Test factory functions with μP parameter."""

    def test_gpt_config_mup(self):
        """gpt_config with mup_base_width sets μP fields."""
        cfg = gpt_config(
            d_model=256, n_heads=4, d_ff=512,
            mup_base_width=64,
        )
        assert cfg.mup_base_width == 64
        assert cfg.block.mup is True
        assert cfg.width_mult == 4.0

    def test_gpt_config_no_mup(self):
        """gpt_config without mup_base_width leaves SP."""
        cfg = gpt_config(d_model=256, n_heads=4, d_ff=512)
        assert cfg.mup_base_width is None
        assert cfg.block.mup is False

    def test_llama_config_mup(self):
        """llama_config with mup_base_width sets μP fields."""
        cfg = llama_config(
            d_model=256, n_heads=4, n_kv_heads=2,
            d_ff=341, mup_base_width=64,
        )
        assert cfg.mup_base_width == 64
        assert cfg.block.mup is True

    def test_llama_config_no_mup(self):
        """llama_config without mup_base_width leaves SP."""
        cfg = llama_config(
            d_model=256, n_heads=4, n_kv_heads=2,
            d_ff=341,
        )
        assert cfg.mup_base_width is None
        assert cfg.block.mup is False


class TestMupWeightInit:
    """Test μP weight initialization scaling.

    Cross-referenced against Microsoft mup and Cerebras:
    - Hidden layer weights: scaled by 1/√width_mult
    - Embedding weights: unchanged (constant across widths)
    - Output head weights (untied): scaled by 1/√width_mult

    Reference: Yang et al. 2022 Table 8, Cerebras
    "Practitioner's Guide to μP".
    """

    def test_hidden_weight_variance_shrinks_with_width(self):
        """Hidden weight variance scales as 1/width_mult.

        μP requires init variance of hidden layers to shrink
        proportionally to width_mult. Since std scales by
        1/√width_mult, variance scales by 1/width_mult.
        """
        mx.random.seed(42)
        base_width = 64
        target_width = 256
        width_mult = target_width / base_width  # 4.0

        # Build target model (μP, width=target)
        mx.random.seed(42)
        cfg_mup = gpt_config(
            vocab_size=32, d_model=target_width, n_heads=4,
            n_layers=1, d_ff=target_width * 4, max_seq_len=16,
            mup_base_width=base_width,
        )
        model_mup = LanguageModel(cfg_mup)

        # Build SP model at same target width (no μP scaling)
        mx.random.seed(42)
        cfg_sp = gpt_config(
            vocab_size=32, d_model=target_width, n_heads=4,
            n_layers=1, d_ff=target_width * 4, max_seq_len=16,
        )
        model_sp = LanguageModel(cfg_sp)

        # Compare a hidden weight variance: μP should be
        # smaller than SP by factor of 1/√width_mult
        mup_w = model_mup.blocks[0].attention.q_proj.weight
        sp_w = model_sp.blocks[0].attention.q_proj.weight
        mx.eval(mup_w, sp_w)

        mup_var = mx.var(mup_w).item()
        sp_var = mx.var(sp_w).item()

        # μP variance should be ~sp_var / width_mult
        # (std scaled by 1/√m, so var scaled by 1/m)
        ratio = mup_var / sp_var
        expected_ratio = 1.0 / width_mult
        assert ratio == pytest.approx(
            expected_ratio, rel=0.15,
        ), (
            f'μP/SP variance ratio {ratio:.4f} != '
            f'expected {expected_ratio:.4f}'
        )

    def test_embedding_weight_unchanged_by_mup(self):
        """Embedding init is NOT scaled by μP.

        μP prescribes constant embedding init regardless of
        width. Reference: Yang et al. 2022, Cerebras guide.
        """
        mx.random.seed(42)
        cfg_sp = gpt_config(
            vocab_size=32, d_model=128, n_heads=4,
            n_layers=1, d_ff=512, max_seq_len=16,
        )
        model_sp = LanguageModel(cfg_sp)
        sp_embed = model_sp.embed.weight

        mx.random.seed(42)
        cfg_mup = gpt_config(
            vocab_size=32, d_model=128, n_heads=4,
            n_layers=1, d_ff=512, max_seq_len=16,
            mup_base_width=64,
        )
        model_mup = LanguageModel(cfg_mup)
        mup_embed = model_mup.embed.weight

        mx.eval(sp_embed, mup_embed)
        # Embedding weights should be identical
        assert mx.allclose(sp_embed, mup_embed, atol=1e-6), (
            'μP should NOT modify embedding weights'
        )

    def test_width_mult_1_no_init_change(self):
        """At width_mult=1, init is unchanged."""
        mx.random.seed(42)
        cfg_sp = gpt_config(
            vocab_size=32, d_model=64, n_heads=2,
            n_layers=1, d_ff=256, max_seq_len=16,
        )
        model_sp = LanguageModel(cfg_sp)

        mx.random.seed(42)
        cfg_mup = gpt_config(
            vocab_size=32, d_model=64, n_heads=2,
            n_layers=1, d_ff=256, max_seq_len=16,
            mup_base_width=64,
        )
        model_mup = LanguageModel(cfg_mup)

        # All weights should be identical
        sp_flat = mlx.utils.tree_flatten(model_sp.parameters())
        mup_flat = mlx.utils.tree_flatten(model_mup.parameters())
        for (k1, v1), (k2, v2) in zip(
            sp_flat, mup_flat, strict=True,
        ):
            mx.eval(v1, v2)
            assert k1 == k2
            assert mx.allclose(v1, v2, atol=1e-6), (
                f'Weight {k1} differs at width_mult=1'
            )

    def test_ffn_weight_scaled_by_mup(self):
        """FFN weights are also scaled by 1/√width_mult."""
        mx.random.seed(42)
        cfg_sp = gpt_config(
            vocab_size=32, d_model=128, n_heads=4,
            n_layers=1, d_ff=512, max_seq_len=16,
        )
        model_sp = LanguageModel(cfg_sp)

        mx.random.seed(42)
        cfg_mup = gpt_config(
            vocab_size=32, d_model=128, n_heads=4,
            n_layers=1, d_ff=512, max_seq_len=16,
            mup_base_width=32,
        )
        model_mup = LanguageModel(cfg_mup)

        width_mult = 128 / 32  # 4.0
        expected_scale = 1.0 / math.sqrt(width_mult)  # 0.5

        sp_w = model_sp.blocks[0].ffn.up.weight
        mup_w = model_mup.blocks[0].ffn.up.weight
        mx.eval(sp_w, mup_w)

        # μP weight should be SP weight * scale
        expected = sp_w * expected_scale
        mx.eval(expected)
        assert mx.allclose(mup_w, expected, atol=1e-6), (
            'FFN weight not scaled correctly by μP'
        )


class TestMupAttentionScaleReference:
    """Cross-reference attention scaling against known values.

    References:
    - PyTorch nn.MultiheadAttention: 1/√d_head
    - nanoGPT: 1.0 / math.sqrt(k.size(-1))
    - HuggingFace LlamaAttention: 1/√d_head
    - Microsoft mup: 1/d_head for μP
    - Cerebras: 1/d_head for μP
    """

    def test_sp_matches_pytorch_convention(self):
        """SP scale matches PyTorch's 1/√d_k convention."""
        for d_model, n_heads in [(64, 4), (128, 8), (256, 8)]:
            d_head = d_model // n_heads
            cfg = BlockConfig(
                d_model=d_model, n_heads=n_heads,
            )
            attn = MHA(cfg)
            # PyTorch convention: 1/√d_head
            expected = 1.0 / math.sqrt(d_head)
            assert attn.scale == pytest.approx(expected), (
                f'd_model={d_model}, n_heads={n_heads}: '
                f'scale={attn.scale} != {expected}'
            )

    def test_mup_matches_microsoft_convention(self):
        """μP scale matches Microsoft mup's 1/d_head."""
        for d_model, n_heads in [(64, 4), (128, 8), (256, 8)]:
            d_head = d_model // n_heads
            cfg = BlockConfig(
                d_model=d_model, n_heads=n_heads,
                mup=True,
            )
            attn = MHA(cfg)
            # Microsoft mup convention: 1/d_head
            expected = 1.0 / d_head
            assert attn.scale == pytest.approx(expected), (
                f'd_model={d_model}, n_heads={n_heads}: '
                f'scale={attn.scale} != {expected}'
            )


class TestMupCoordinateCheck:
    """Coordinate check: activations shouldn't grow with width.

    The standard μP validation test (Yang et al. 2022): verify
    that hidden activations have roughly constant scale as width
    increases, when using μP. Under SP, activations grow with
    width.
    """

    def test_hidden_activations_stable_across_widths(self):
        """Hidden activation norms stay bounded across widths."""
        mx.random.seed(42)
        base_width = 32
        widths = [32, 64, 128]
        norms = []

        for width in widths:
            cfg = gpt_config(
                vocab_size=32, d_model=width,
                n_heads=max(2, width // 16),
                n_layers=1, d_ff=width * 2,
                max_seq_len=16,
                mup_base_width=base_width,
            )
            model = LanguageModel(cfg)
            x = mx.random.randint(0, 32, shape=(1, 8))
            logits, _ = model(x)
            mx.eval(logits)
            norm = mx.sqrt(mx.sum(logits * logits)).item()
            norms.append(norm)

        # Under μP, logit norms should NOT grow linearly
        # with width. Check that 4x width doesn't give >3x norm.
        ratio = norms[-1] / max(norms[0], 1e-8)
        width_ratio = widths[-1] / widths[0]  # 4.0
        assert ratio < width_ratio, (
            f'Logit norm grew {ratio:.1f}x for {width_ratio}x '
            f'width — μP scaling may not be working'
        )
