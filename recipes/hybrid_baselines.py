"""Train baseline models for hybrid architecture analysis.

Trains 5 architectures on TinyStories BPE to convergence
and saves checkpoints for downstream analysis (activation
capture, probing, layer ablation).

Requires: ``uv sync --extra hf --extra tokenizers``

Usage:
    uv run python recipes/hybrid_baselines.py
    uv run python recipes/hybrid_baselines.py --arch gpt_10m
    uv run python recipes/hybrid_baselines.py --target-steps 500
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from lmxlab.data.dataset import HFDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.experiments.flops import estimate_flops_per_step
from lmxlab.experiments.mlflow import (
    MLflowCallback,
    MLflowExperimentRunner,
)
from lmxlab.experiments.runner import ExperimentConfig
from lmxlab.models.bamba import bamba_10m
from lmxlab.models.base import LanguageModel
from lmxlab.models.falcon import falcon_h1_10m
from lmxlab.models.gpt import gpt_10m
from lmxlab.models.jamba import jamba_10m
from lmxlab.models.llama import llama_10m
from lmxlab.training.callbacks import (
    FLOPCounter,
    ValTracker,
    standard_callbacks,
)
from lmxlab.training.config import TrainConfig
from lmxlab.training.hardware import detect_peak_tflops
from lmxlab.training.trainer import Trainer

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
LEARNING_RATE = 3e-4
EVAL_BATCHES = 20


def train_model(
    arch_name: str,
    target_steps: int = 2000,
) -> dict[str, Any]:
    """Train a single model and save checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"Training: {arch_name}")

    mx.random.seed(42)
    factory = ARCHS[arch_name]
    config = factory()
    model = LanguageModel(config)
    mx.eval(model.parameters())
    n_params = model.count_parameters()
    print(f"  params={n_params:,}")

    # Data
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
    val_batches = list(
        val_ds.batch_iterator(
            batch_size=BATCH_SIZE,
            max_batches=EVAL_BATCHES,
        )
    )

    # FLOP budget
    flops_per_step = estimate_flops_per_step(config, BATCH_SIZE, SEQ_LEN)
    flop_budget = int(flops_per_step * target_steps)
    est_steps = flop_budget / flops_per_step
    print(f"  flop_budget={flop_budget:.2e}")
    print(f"  est_steps={est_steps:.0f}")

    # Callbacks (standard stack + MLflow)
    cbs = standard_callbacks(
        log_interval=100,
        tokens_per_step=BATCH_SIZE * SEQ_LEN,
        flops_per_step=flops_per_step,
        flop_budget=flop_budget,
        hardware_peak_tflops=detect_peak_tflops(),
        model=model,
        val_batches=val_batches,
        eval_interval=500,
    )
    flop_counter = next(c for c in cbs if isinstance(c, FLOPCounter))
    val_tracker = next(c for c in cbs if isinstance(c, ValTracker))
    mlflow_cb = MLflowCallback(
        log_interval=100,
        log_model_params=False,
    )
    cbs.append(mlflow_cb)

    # Train
    train_config = TrainConfig(
        learning_rate=LEARNING_RATE,
        max_steps=100_000,
        batch_size=BATCH_SIZE,
        warmup_steps=100,
        eval_interval=500,
        compile_step=False,
    )
    trainer = Trainer(
        model,
        train_config,
        callbacks=cbs,
    )

    # Experiment tracking (MLflow + results.jsonl)
    exp_config = ExperimentConfig(
        name="hybrid-baselines",
        description=arch_name,
        time_budget_s=600.0,
        seed=42,
        output_dir="experiments",
    )
    runner = MLflowExperimentRunner(
        exp_config,
        tags={"arch": arch_name},
    )
    runner.start()

    start = time.monotonic()

    def data_iter():
        for batch in train_ds.batch_iterator(
            batch_size=BATCH_SIZE,
        ):
            if flop_counter.should_stop:
                break
            yield batch

    history = trainer.train(data_iter())
    elapsed = time.monotonic() - start

    # ValTracker handles init/periodic/final eval
    train_loss = history[-1]["loss"] if history else float("inf")
    steps = len(history)

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{arch_name}.safetensors"
    model.save_weights(str(ckpt_path))
    print(f"  checkpoint saved: {ckpt_path}")

    # Log to MLflow + results.jsonl
    runner.finish(
        metrics={
            "val_loss": val_tracker.best_val_loss,
            "train_loss": train_loss,
            "train_val_gap": (train_loss - val_tracker.best_val_loss),
            "steps": steps,
            "total_flops": flop_counter.total_flops,
        },
        param_count=n_params,
        config_dict={
            "arch": arch_name,
            "lr": LEARNING_RATE,
            "flop_budget": flop_budget,
        },
    )

    result = {
        "arch": arch_name,
        "params": n_params,
        "steps": steps,
        "train_loss": train_loss,
        "val_loss": val_tracker.best_val_loss,
        "gap": train_loss - val_tracker.best_val_loss,
        "wall_time": elapsed,
        "total_flops": flop_counter.total_flops,
        "checkpoint": str(ckpt_path),
    }

    print(f"  Steps:      {steps}")
    print(f"  Train loss: {train_loss:.4f}")
    bv = val_tracker.best_val_loss
    print(f"  Best val:   {bv:.4f}")
    print(f"  Gap:        {train_loss - bv:+.4f}")
    print(f"  Wall time:  {elapsed:.1f}s")

    return result


def main() -> None:
    """Train baseline models."""
    parser = argparse.ArgumentParser(description="Train hybrid baselines")
    parser.add_argument(
        "--arch",
        default="all",
        choices=["all"] + list(ARCHS),
        help="Architecture to train (default: all)",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=2000,
        help="Target steps per model (default: 2000)",
    )
    args = parser.parse_args()

    archs = list(ARCHS) if args.arch == "all" else [args.arch]

    print("Hybrid Baselines Training")
    print(f"Architectures: {archs}")
    print(f"Target steps: {args.target_steps}")

    results = []
    for arch in archs:
        result = train_model(arch, args.target_steps)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    header = f"{'Arch':<20} {'Val':>8} {'Train':>8} {'Steps':>6} {'Time':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['arch']:<20} "
            f"{r['val_loss']:>8.4f} "
            f"{r['train_loss']:>8.4f} "
            f"{r['steps']:>6} "
            f"{r['wall_time']:>6.1f}s"
        )

    # Save results
    out = Path("experiments") / "hybrid_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
