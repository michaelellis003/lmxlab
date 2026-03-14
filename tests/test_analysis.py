"""Tests for the analysis and interpretability toolkit."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from lmxlab.analysis.activations import ActivationCapture
from lmxlab.analysis.attention import extract_attention_maps
from lmxlab.analysis.probing import LinearProbe
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.llama import llama_tiny


class TestActivationCapture:
    """Tests for ActivationCapture."""

    def test_returns_correct_keys(self):
        """Captured dict has expected layer keys."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((1, 8), dtype=mx.int32)

        with ActivationCapture(model) as cap:
            model(tokens)

        # GPT tiny has 2 layers
        assert "layer_0/input" in cap.activations
        assert "layer_0/output" in cap.activations
        assert "layer_1/input" in cap.activations
        assert "layer_1/output" in cap.activations

    def test_correct_shapes(self):
        """Activations have shape (batch, seq, d_model)."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((2, 16), dtype=mx.int32)

        with ActivationCapture(model) as cap:
            model(tokens)

        for key, val in cap.activations.items():
            mx.eval(val)
            assert val.shape == (2, 16, 64), f"{key}: {val.shape}"

    def test_context_manager_restores(self):
        """Original blocks are restored after exiting context."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        original_block = model.blocks[0]

        with ActivationCapture(model):
            # During capture, blocks are wrapped
            assert model.blocks[0] is not original_block

        # After exit, originals restored
        assert model.blocks[0] is original_block

    def test_layer_norms(self):
        """layer_norms() returns float values for each key."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((1, 8), dtype=mx.int32)

        with ActivationCapture(model) as cap:
            model(tokens)

        norms = cap.layer_norms()
        assert len(norms) > 0
        for _key, val in norms.items():
            assert isinstance(val, float)
            assert val >= 0

    def test_output_matches_without_capture(self):
        """Model output is unchanged by activation capture."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((1, 8), dtype=mx.int32)

        # Without capture
        logits_ref, _ = model(tokens)
        mx.eval(logits_ref)

        # With capture
        with ActivationCapture(model):
            logits_cap, _ = model(tokens)
        mx.eval(logits_cap)

        assert mx.allclose(logits_ref, logits_cap, atol=1e-5).item()


class TestAttentionMaps:
    """Tests for unfused attention extraction."""

    def test_returns_per_layer(self):
        """Returns one map per attention layer."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((1, 8), dtype=mx.int32)

        maps = extract_attention_maps(model, tokens)

        assert "layer_0" in maps
        assert "layer_1" in maps

    def test_correct_shape(self):
        """Attention maps have (batch, heads, seq, seq)."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((1, 8), dtype=mx.int32)

        maps = extract_attention_maps(model, tokens)

        # GPT tiny: 2 heads
        w = maps["layer_0"]
        assert w.shape == (1, 2, 8, 8)

    def test_weights_sum_to_one(self):
        """Attention weights are valid probabilities."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((1, 8), dtype=mx.int32)

        maps = extract_attention_maps(model, tokens)
        w = maps["layer_0"]

        # Each row should sum to ~1
        row_sums = w.sum(axis=-1)
        mx.eval(row_sums)
        assert mx.allclose(
            row_sums,
            mx.ones_like(row_sums),
            atol=1e-5,
        ).item()

    def test_causal_mask_applied(self):
        """Future positions should have ~0 attention weight."""
        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((1, 8), dtype=mx.int32)

        maps = extract_attention_maps(model, tokens)
        w = maps["layer_0"]
        mx.eval(w)

        # Position 0 should not attend to position 1+
        assert w[0, 0, 0, 1].item() < 1e-5

    def test_llama_gqa(self):
        """Works with GQA attention (LLaMA)."""
        config = llama_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())
        tokens = mx.zeros((1, 8), dtype=mx.int32)

        maps = extract_attention_maps(model, tokens)

        assert "layer_0" in maps
        # LLaMA tiny: 4 heads
        assert maps["layer_0"].shape == (1, 4, 8, 8)


class TestLinearProbe:
    """Tests for LinearProbe."""

    def test_output_shape(self):
        """Probe outputs correct number of classes."""
        probe = LinearProbe(64, 256)
        mx.eval(probe.parameters())

        x = mx.random.normal((2, 8, 64))
        out = probe(x)
        mx.eval(out)

        assert out.shape == (2, 8, 256)

    def test_trainable(self):
        """Probe parameters are trainable."""
        probe = LinearProbe(64, 10)
        mx.eval(probe.parameters())

        x = mx.random.normal((4, 64))
        y = mx.zeros((4,), dtype=mx.int32)

        loss_fn = nn.value_and_grad(
            probe,
            lambda p, x, y: nn.losses.cross_entropy(p(x), y, reduction="mean"),
        )
        loss, grads = loss_fn(probe, x, y)
        mx.eval(loss, grads)

        assert loss.item() > 0


class TestPlotting:
    """Tests for plotting utilities."""

    @pytest.fixture(autouse=True)
    def _check_matplotlib(self):
        """Skip if matplotlib not installed."""
        pytest.importorskip("matplotlib")

    def test_plot_loss_curves(self):
        """plot_loss_curves returns a Figure."""
        from lmxlab.analysis.plotting import plot_loss_curves

        fig = plot_loss_curves([3.0, 2.5, 2.0, 1.8])
        from matplotlib.figure import Figure

        assert isinstance(fig, Figure)

    def test_plot_loss_curves_with_val(self):
        """plot_loss_curves handles val losses."""
        from lmxlab.analysis.plotting import plot_loss_curves

        fig = plot_loss_curves(
            [3.0, 2.5, 2.0, 1.8],
            [2.8, 2.3],
        )
        from matplotlib.figure import Figure

        assert isinstance(fig, Figure)

    def test_plot_layer_norms(self):
        """plot_layer_norms returns a Figure."""
        from lmxlab.analysis.plotting import plot_layer_norms

        acts = {
            "layer_0/output": mx.ones((1, 8, 64)),
            "layer_1/output": mx.ones((1, 8, 64)) * 2,
        }
        fig = plot_layer_norms(acts)
        from matplotlib.figure import Figure

        assert isinstance(fig, Figure)

    def test_plot_attention_heatmap(self):
        """plot_attention_heatmap returns a Figure."""
        from lmxlab.analysis.plotting import (
            plot_attention_heatmap,
        )

        weights = mx.ones((1, 2, 4, 4)) * 0.25
        fig = plot_attention_heatmap(weights)
        from matplotlib.figure import Figure

        assert isinstance(fig, Figure)

    def test_plot_gradient_flow(self):
        """plot_gradient_flow returns a Figure."""
        from lmxlab.analysis.plotting import plot_gradient_flow

        config = gpt_tiny()
        model = LanguageModel(config)
        mx.eval(model.parameters())

        fig = plot_gradient_flow(model)
        from matplotlib.figure import Figure

        assert isinstance(fig, Figure)
