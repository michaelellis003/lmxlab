"""Linear probing of hybrid architecture representations.

For each trained model, trains linear probes at each layer
on next-token prediction. Compares probe accuracy curves
between SSM and attention layers.

Requires: trained checkpoints from ``hybrid_baselines.py``

Usage:
    uv run python recipes/hybrid_probing.py
    uv run python recipes/hybrid_probing.py --arch jamba_10m
    uv run python recipes/hybrid_probing.py --probe-steps 100
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from lmxlab.analysis.probing import (
    LinearProbe,
    probe_accuracy,
    train_probe,
)
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
SEQ_LEN = 256
PROBE_BATCH = 4
TRAIN_BATCHES = 20
VAL_BATCHES = 10


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
            f"No checkpoint: {ckpt}. Run hybrid_baselines.py first."
        )
    model.load_weights(str(ckpt))
    model.eval()
    return model


def probe_model(
    arch_name: str,
    probe_steps: int = 200,
) -> dict[str, Any]:
    """Train probes at each layer of a model."""
    print(f"\n{'=' * 50}")
    print(f"Probing: {arch_name}")

    model = load_model(arch_name)
    n_layers = model.config.n_layers
    d_model = model.config.block.d_model
    vocab_size = model.config.vocab_size

    tokenizer = TiktokenTokenizer("gpt2")
    train_ds = HFDataset(
        "roneneldan/TinyStories",
        tokenizer,
        seq_len=SEQ_LEN,
        split="train",
    )
    val_ds = HFDataset(
        "roneneldan/TinyStories",
        tokenizer,
        seq_len=SEQ_LEN,
        split="validation",
    )

    print(f"  {'Layer':>5} {'Type':>5} {'ValAcc':>8}")
    print(f"  {'-' * 22}")

    layers = []
    for i in range(n_layers):
        ltype = layer_type(model, i)

        # Fresh probe for this layer
        probe = LinearProbe(d_model, vocab_size)
        mx.eval(probe.parameters())

        # Train (fresh iterator each layer)
        train_iter = train_ds.batch_iterator(
            batch_size=PROBE_BATCH,
            max_batches=probe_steps,
        )
        train_probe(
            model,
            train_iter,
            layer=i,
            probe=probe,
            steps=probe_steps,
        )

        # Evaluate
        val_iter = val_ds.batch_iterator(
            batch_size=PROBE_BATCH,
            max_batches=VAL_BATCHES,
        )
        acc = probe_accuracy(
            model,
            val_iter,
            layer=i,
            probe=probe,
            max_batches=VAL_BATCHES,
        )

        layers.append(
            {
                "layer": i,
                "type": ltype,
                "val_acc": acc,
            }
        )
        print(f"  {i:>5} {ltype:>5} {acc:>8.4f}")

    # Aggregate by type
    ssm = [e for e in layers if e["type"] == "ssm"]
    attn = [e for e in layers if e["type"] == "attn"]

    def mean_acc(entries):
        if not entries:
            return 0.0
        return sum(e["val_acc"] for e in entries) / len(entries)

    summary = {
        "arch": arch_name,
        "n_layers": n_layers,
        "ssm_avg_acc": mean_acc(ssm),
        "attn_avg_acc": mean_acc(attn),
        "layers": layers,
    }

    if ssm and attn:
        print(f"\n  SSM avg val acc:  {summary['ssm_avg_acc']:.4f}")
        print(f"  Attn avg val acc: {summary['attn_avg_acc']:.4f}")

    return summary


def main() -> None:
    """Run probing analysis."""
    parser = argparse.ArgumentParser(description="Hybrid probing analysis")
    parser.add_argument(
        "--arch",
        default="all",
        choices=["all"] + list(ARCHS),
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=200,
        help="Steps per probe (default: 200)",
    )
    args = parser.parse_args()

    archs = list(ARCHS) if args.arch == "all" else [args.arch]

    results = []
    for arch in archs:
        ckpt = CHECKPOINT_DIR / f"{arch}.safetensors"
        if not ckpt.exists():
            print(f"Skipping {arch} (no checkpoint)")
            continue
        result = probe_model(arch, args.probe_steps)
        results.append(result)

    # Save
    out = Path("experiments") / "hybrid_probing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
