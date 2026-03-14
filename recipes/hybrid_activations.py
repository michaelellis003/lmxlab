"""Activation analysis for hybrid architectures.

Loads trained checkpoints and compares per-layer activation
norms between SSM and attention layers. Answers: do SSM
layers contribute more or less to the residual stream than
attention layers at the same depth?

Requires: trained checkpoints from ``hybrid_baselines.py``

Usage:
    uv run python recipes/hybrid_activations.py
    uv run python recipes/hybrid_activations.py --arch falcon_h1_10m
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from lmxlab.analysis.activations import ActivationCapture
from lmxlab.data.dataset import HFDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.models.bamba import bamba_10m
from lmxlab.models.base import LanguageModel
from lmxlab.models.falcon import falcon_h1_10m
from lmxlab.models.gpt import gpt_10m
from lmxlab.models.jamba import jamba_10m
from lmxlab.models.llama import llama_10m

ARCHS = {
    "gpt_10m": gpt_10m,
    "llama_10m": llama_10m,
    "falcon_h1_10m": falcon_h1_10m,
    "jamba_10m": jamba_10m,
    "bamba_10m": bamba_10m,
}
CHECKPOINT_DIR = Path("experiments/checkpoints")
BATCH_SIZE = 4
SEQ_LEN = 256


def layer_type(model, layer_idx: int) -> str:
    """Classify a layer as 'ssm' or 'attn'."""
    cfg = model.config.get_block_config(layer_idx)
    if "mamba" in cfg.attention:
        return "ssm"
    return "attn"


def load_model(arch_name: str) -> LanguageModel:
    """Load model from checkpoint."""
    config = ARCHS[arch_name]()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    ckpt = CHECKPOINT_DIR / f"{arch_name}.safetensors"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. Run hybrid_baselines.py first."
        )
    model.load_weights(str(ckpt))
    model.eval()
    return model


def get_val_batch() -> mx.array:
    """Get a fixed validation batch of tokens."""
    tokenizer = TiktokenTokenizer("gpt2")
    val_ds = HFDataset(
        "roneneldan/TinyStories",
        tokenizer,
        seq_len=SEQ_LEN,
        split="validation",
    )
    batches = list(val_ds.batch_iterator(batch_size=BATCH_SIZE, max_batches=1))
    return batches[0][0]  # Just input tokens


def analyze_model(arch_name: str, tokens: mx.array) -> dict[str, Any]:
    """Run activation analysis on one model."""
    print(f"\n{'=' * 50}")
    print(f"Analyzing: {arch_name}")

    model = load_model(arch_name)
    n_layers = model.config.n_layers

    # Capture activations
    with ActivationCapture(model) as cap:
        model(tokens)

    # Compute per-layer norms
    norms = cap.layer_norms()

    # Classify layers and compute stats
    layers = []
    for i in range(n_layers):
        ltype = layer_type(model, i)
        in_key = f"layer_{i}/input"
        out_key = f"layer_{i}/output"
        in_norm = norms.get(in_key, 0.0)
        out_norm = norms.get(out_key, 0.0)

        # Residual contribution: output - input norm
        in_act = cap.activations.get(in_key)
        out_act = cap.activations.get(out_key)
        if in_act is not None and out_act is not None:
            mx.eval(in_act, out_act)
            residual = out_act - in_act
            resid_norm = mx.sqrt(mx.mean(residual * residual)).item()
        else:
            resid_norm = 0.0

        layers.append(
            {
                "layer": i,
                "type": ltype,
                "input_norm": in_norm,
                "output_norm": out_norm,
                "residual_norm": resid_norm,
            }
        )

    # Print per-layer results
    print(f"  {'Layer':>5} {'Type':>5} {'In':>8} {'Out':>8} {'Resid':>8}")
    print(f"  {'-' * 38}")
    for layer in layers:
        print(
            f"  {layer['layer']:>5} {layer['type']:>5} "
            f"{layer['input_norm']:>8.3f} "
            f"{layer['output_norm']:>8.3f} "
            f"{layer['residual_norm']:>8.3f}"
        )

    # Aggregate by layer type
    ssm_layers = [e for e in layers if e["type"] == "ssm"]
    attn_layers = [e for e in layers if e["type"] == "attn"]

    def mean_stat(entries, key):
        if not entries:
            return 0.0
        return sum(e[key] for e in entries) / len(entries)

    summary = {
        "arch": arch_name,
        "n_layers": n_layers,
        "n_ssm": len(ssm_layers),
        "n_attn": len(attn_layers),
        "ssm_avg_resid": mean_stat(ssm_layers, "residual_norm"),
        "attn_avg_resid": mean_stat(attn_layers, "residual_norm"),
        "ssm_avg_out": mean_stat(ssm_layers, "output_norm"),
        "attn_avg_out": mean_stat(attn_layers, "output_norm"),
        "layers": layers,
    }

    if ssm_layers and attn_layers:
        ratio = summary["ssm_avg_resid"] / max(summary["attn_avg_resid"], 1e-8)
        print(f"\n  SSM avg residual norm: {summary['ssm_avg_resid']:.4f}")
        print(f"  Attn avg residual norm: {summary['attn_avg_resid']:.4f}")
        print(f"  SSM/Attn ratio: {ratio:.3f}")

    return summary


def main() -> None:
    """Run activation analysis."""
    parser = argparse.ArgumentParser(description="Hybrid activation analysis")
    parser.add_argument(
        "--arch",
        default="all",
        choices=["all"] + list(ARCHS),
    )
    args = parser.parse_args()

    archs = list(ARCHS) if args.arch == "all" else [args.arch]

    tokens = get_val_batch()
    print(f"Val batch: {tokens.shape}")

    results = []
    for arch in archs:
        ckpt = CHECKPOINT_DIR / f"{arch}.safetensors"
        if not ckpt.exists():
            print(f"Skipping {arch} (no checkpoint)")
            continue
        result = analyze_model(arch, tokens)
        results.append(result)

    # Save results
    out = Path("experiments") / "hybrid_activations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Strip mx.array from layers before saving
    for r in results:
        for layer in r["layers"]:
            for k in list(layer):
                if isinstance(layer[k], mx.array):
                    layer[k] = float(layer[k].item())
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
