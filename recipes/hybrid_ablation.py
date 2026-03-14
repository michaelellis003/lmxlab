"""Layer ablation for hybrid architectures.

Zeroes out individual layers and measures val_loss impact.
Answers: is removing an SSM layer more or less damaging than
removing an attention layer at the same position?

Requires: trained checkpoints from ``hybrid_baselines.py``

Usage:
    uv run python recipes/hybrid_ablation.py
    uv run python recipes/hybrid_ablation.py --arch falcon_h1_10m
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

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
BATCH_SIZE = 8
SEQ_LEN = 256
EVAL_BATCHES = 20


class _IdentityBlock(nn.Module):
    """Passes input through unchanged (skip layer)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position = None

    def __call__(self, x, mask=None, cache=None):
        """Return input unchanged."""
        return x, cache


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


def evaluate(
    model: LanguageModel,
    val_batches: list[tuple[mx.array, mx.array]],
) -> float:
    """Compute val loss."""
    total = 0.0
    n = 0
    for x, y in val_batches:
        logits, _ = model(x)
        logits = logits.reshape(-1, logits.shape[-1])
        loss = nn.losses.cross_entropy(logits, y.reshape(-1), reduction="mean")
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def ablate_model(
    arch_name: str,
    val_batches: list[tuple[mx.array, mx.array]],
) -> dict[str, Any]:
    """Ablate each layer and measure val_loss impact."""
    print(f"\n{'=' * 50}")
    print(f"Ablating: {arch_name}")

    model = load_model(arch_name)
    n_layers = model.config.n_layers

    # Baseline val loss (all layers active)
    baseline_loss = evaluate(model, val_batches)
    print(f"  Baseline val_loss: {baseline_loss:.4f}")

    print(f"  {'Layer':>5} {'Type':>5} {'Loss':>8} {'Delta':>8}")
    print(f"  {'-' * 30}")

    layers = []
    for i in range(n_layers):
        ltype = layer_type(model, i)

        # Swap layer with identity
        original = model.blocks[i]
        block_cfg = model.config.get_block_config(i)
        model.blocks[i] = _IdentityBlock(block_cfg)

        ablated_loss = evaluate(model, val_batches)
        delta = ablated_loss - baseline_loss

        # Restore
        model.blocks[i] = original

        layers.append(
            {
                "layer": i,
                "type": ltype,
                "ablated_loss": ablated_loss,
                "delta": delta,
            }
        )
        print(f"  {i:>5} {ltype:>5} {ablated_loss:>8.4f} {delta:>+8.4f}")

    # Aggregate
    ssm = [e for e in layers if e["type"] == "ssm"]
    attn = [e for e in layers if e["type"] == "attn"]

    def mean_delta(entries):
        if not entries:
            return 0.0
        return sum(e["delta"] for e in entries) / len(entries)

    summary = {
        "arch": arch_name,
        "baseline_loss": baseline_loss,
        "n_layers": n_layers,
        "ssm_avg_delta": mean_delta(ssm),
        "attn_avg_delta": mean_delta(attn),
        "layers": layers,
    }

    if ssm and attn:
        print(f"\n  SSM avg delta:  {summary['ssm_avg_delta']:+.4f}")
        print(f"  Attn avg delta: {summary['attn_avg_delta']:+.4f}")
        more_critical = (
            "SSM"
            if summary["ssm_avg_delta"] > summary["attn_avg_delta"]
            else "Attn"
        )
        print(f"  More critical: {more_critical}")

    return summary


def main() -> None:
    """Run ablation analysis."""
    parser = argparse.ArgumentParser(description="Hybrid layer ablation")
    parser.add_argument(
        "--arch",
        default="all",
        choices=["all"] + list(ARCHS),
    )
    args = parser.parse_args()

    archs = list(ARCHS) if args.arch == "all" else [args.arch]

    # Shared val data
    tokenizer = TiktokenTokenizer("gpt2")
    val_ds = HFDataset(
        "roneneldan/TinyStories",
        tokenizer,
        seq_len=SEQ_LEN,
        split="validation",
    )
    val_batches = list(
        val_ds.batch_iterator(
            batch_size=BATCH_SIZE,
            max_batches=EVAL_BATCHES,
        )
    )

    results = []
    for arch in archs:
        ckpt = CHECKPOINT_DIR / f"{arch}.safetensors"
        if not ckpt.exists():
            print(f"Skipping {arch} (no checkpoint)")
            continue
        result = ablate_model(arch, val_batches)
        results.append(result)

    # Save
    out = Path("experiments") / "hybrid_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
