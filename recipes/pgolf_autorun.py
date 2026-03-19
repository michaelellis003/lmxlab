"""Parameter Golf autorun recipe: autonomous BPB optimization.

Drives the OpenAI Parameter Golf challenge (https://github.com/openai/parameter-golf)
via the lmxlab /autorun skill. Each iteration modifies hyperparameters or architecture
in train_gpt_mlx.py, runs training, and logs results.

The agent loop (driven by Claude Code via /autorun) calls:
  1. propose(past_results) -> config dict
  2. run(config) -> metrics dict
  3. Log result and print JSON summary

Usage:
    # Single iteration (for agent loop):
    uv run python recipes/pgolf_autorun.py --single

    # Single iteration, quick smoke test (200 steps):
    uv run python recipes/pgolf_autorun.py --single --smoke

    # Full local loop (no agent):
    uv run python recipes/pgolf_autorun.py
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# lmxlab experiment tracking
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lmxlab.experiments.tracking import ExperimentLog, LogEntry

# ── Task identity ────────────────────────────────────────────
TASK_NAME = "pgolf"
MAX_ITERATIONS = 20
PGOLF_DIR = Path(__file__).resolve().parent.parent.parent / "parameter-golf"
RESULTS_FILE = "experiments/pgolf_results.jsonl"
ARTIFACT_LIMIT_BYTES = 16_000_000
SCRIPT_BACKUP_DIR = Path("experiments/pgolf_scripts")

# ── Baseline defaults (match train_gpt_mlx.py) ──────────────
BASELINE = {
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "9",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "2",
    "TIE_EMBEDDINGS": "1",
    "TRAIN_SEQ_LEN": "1024",
    "TRAIN_BATCH_TOKENS": "524288",
    "GRAD_ACCUM_STEPS": "8",
    "ITERATIONS": "20000",
    "WARMUP_STEPS": "20",
    "WARMDOWN_ITERS": "1200",
    "MAX_WALLCLOCK_SECONDS": "600",
    "MATRIX_LR": "0.04",
    "SCALAR_LR": "0.04",
    "TIED_EMBED_LR": "0.05",
    "MUON_MOMENTUM": "0.95",
    "LOGIT_SOFTCAP": "30.0",
    "ROPE_BASE": "10000.0",
    "QK_GAIN_INIT": "1.5",
}

# ── Local Mac overrides (smaller batch for memory) ───────────
LOCAL_OVERRIDES = {
    "TRAIN_BATCH_TOKENS": "8192",
    "VAL_BATCH_SIZE": "65536",
    "MLX_MAX_MICROBATCH_TOKENS": "4096",
    "GRAD_ACCUM_STEPS": "1",
    # Use truncated val set (2M tokens) for fast local iteration
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024_local",
}


# ── MUTABLE: Claude edits this function between iterations ────


def propose(
    past_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return experiment config for next Parameter Golf iteration.

    MUTABLE -- the agent edits this function between iterations.
    Everything else in this file is immutable infrastructure.

    Args:
        past_results: List of dicts from prior kept runs
            (filtered to TASK_NAME). Each has keys: val_bpb,
            artifact_size_bytes, config, description, etc.

    Returns:
        Config dict with keys:
        - env_overrides: dict of env var overrides for training script
        - description: str describing what changed and why
        - hypothesis: str (HYP-XXX ID this tests)
        - iterations: int (training steps, default from env)
        - smoke: bool (if True, override to 200 steps)
    """
    # HYP-021: Throughput optimization via Muon steps + microbatch
    # Reduce per-step time to get more steps in 600s wallclock.
    hyp021_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-021")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp021_runs)

    configs = [
        {
            "env_overrides": {
                "ITERATIONS": "5000",
                "UNIQUE_BLOCKS": "3",
            },
            "description": "Baseline refresh: 3u blocks, default Muon (5 NS steps)",
            "hypothesis": "HYP-021-baseline",
        },
        {
            "env_overrides": {
                "ITERATIONS": "5000",
                "UNIQUE_BLOCKS": "3",
                "MUON_BACKEND_STEPS": "3",
            },
            "description": "3u + 3 NS steps (vs default 5)",
            "hypothesis": "HYP-021-ns3",
        },
        {
            "env_overrides": {
                "ITERATIONS": "5000",
                "UNIQUE_BLOCKS": "3",
                "MLX_MAX_MICROBATCH_TOKENS": "8192",
            },
            "description": "3u + microbatch=8192 (vs default 4096)",
            "hypothesis": "HYP-021-mb8k",
        },
        {
            "env_overrides": {
                "ITERATIONS": "5000",
                "UNIQUE_BLOCKS": "3",
                "MUON_BACKEND_STEPS": "3",
                "MLX_MAX_MICROBATCH_TOKENS": "8192",
            },
            "description": "3u + 3 NS steps + microbatch=8192 (combined)",
            "hypothesis": "HYP-021-combined",
        },
    ]

    if n < len(configs):
        return configs[n]

    return {
        "env_overrides": {"ITERATIONS": "5000"},
        "description": "done",
        "hypothesis": "HYP-021-done",
    }


# ── IMMUTABLE: infrastructure below ──────────────────────────


def _parse_metrics(output: str) -> dict[str, Any]:
    """Parse training script stdout for metrics.

    Args:
        output: Full stdout from train_gpt_mlx.py.

    Returns:
        Dict with val_bpb, val_loss, artifact_size_bytes,
        param_count, and other parsed metrics.
    """
    metrics: dict[str, Any] = {}

    # Parse final int8 zlib roundtrip line
    m = re.search(
        r"final_int8_zlib_roundtrip\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)",
        output,
    )
    if m:
        metrics["val_loss"] = float(m.group(1))
        metrics["val_bpb"] = float(m.group(2))

    # Parse serialized model size
    m = re.search(r"serialized_model_int8_zlib:(\d+)\s+bytes", output)
    if m:
        metrics["artifact_size_bytes"] = int(m.group(1))

    # Parse param count
    m = re.search(r"model_params:(\d+)", output)
    if m:
        metrics["param_count"] = int(m.group(1))

    # Parse final train loss (last step log)
    for m in re.finditer(r"train_loss:([\d.]+)", output):
        metrics["train_loss"] = float(m.group(1))

    # Parse training time
    m = re.search(r"train_time:([\d.]+)ms", output)
    if m:
        metrics["wall_time_ms"] = float(m.group(1))

    # Parse total steps
    for m in re.finditer(r"step:(\d+)/", output):
        metrics["steps"] = int(m.group(1))

    # Check for early stopping
    if "stopping_early" in output:
        metrics["early_stopped"] = True

    return metrics


def _estimate_param_count(env: dict[str, str]) -> int:
    """Back-of-envelope parameter estimate before training.

    Args:
        env: Environment variables for the training script.

    Returns:
        Estimated parameter count.
    """
    vocab = int(env.get("VOCAB_SIZE", BASELINE["VOCAB_SIZE"]))
    dim = int(env.get("MODEL_DIM", BASELINE["MODEL_DIM"]))
    layers = int(env.get("NUM_LAYERS", BASELINE["NUM_LAYERS"]))
    mlp_mult = int(env.get("MLP_MULT", BASELINE["MLP_MULT"]))
    num_kv_heads = int(env.get("NUM_KV_HEADS", BASELINE["NUM_KV_HEADS"]))
    head_dim = dim // int(env.get("NUM_HEADS", BASELINE["NUM_HEADS"]))
    kv_dim = num_kv_heads * head_dim
    unique_blocks = int(env.get("UNIQUE_BLOCKS", "0"))
    n_stored = unique_blocks if unique_blocks > 0 else layers

    # Embedding (tied, so counted once)
    embed = vocab * dim
    # Per layer: Q + K + V + O projections + MLP (fc + proj)
    attn = dim * dim + dim * kv_dim + dim * kv_dim + dim * dim
    mlp = dim * (dim * mlp_mult) + (dim * mlp_mult) * dim
    per_layer = attn + mlp
    # Skip weights (always num_layers // 2, not affected by sharing)
    skip = (layers // 2) * dim

    return embed + n_stored * per_layer + skip


def _estimate_artifact_bytes(param_count: int) -> int:
    """Rough estimate of int8+zlib compressed artifact size.

    Args:
        param_count: Number of model parameters.

    Returns:
        Estimated artifact size in bytes.
    """
    # Int8: ~1 byte per param + scale overhead (~2%)
    # Zlib: ~20-30% compression on int8 weights
    int8_bytes = int(param_count * 1.02)
    return int(int8_bytes * 0.75)


def run(
    config: dict[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    """Run one Parameter Golf training iteration.

    IMMUTABLE infrastructure. Do not edit.

    Args:
        config: Config from ``propose()``.
        smoke: If True, run only 200 steps for quick validation.

    Returns:
        Metrics dict with val_bpb, artifact_size_bytes, etc.
    """
    description = config.get("description", "")
    hypothesis = config.get("hypothesis", "")
    env_overrides = config.get("env_overrides", {})

    # Build environment
    env = dict(os.environ)
    env.update(BASELINE)
    env.update(LOCAL_OVERRIDES)
    env.update(env_overrides)

    if smoke:
        env["ITERATIONS"] = "200"
        env["VAL_LOSS_EVERY"] = "0"

    env["RUN_ID"] = f"pgolf_{int(time.time())}"

    # Pre-flight: estimate artifact size
    est_params = _estimate_param_count(env)
    est_bytes = _estimate_artifact_bytes(est_params)
    if est_bytes > ARTIFACT_LIMIT_BYTES:
        print(
            f"WARNING: estimated artifact {est_bytes:,} bytes "
            f"exceeds {ARTIFACT_LIMIT_BYTES:,} limit"
        )

    # Back up script if modified
    script_path = PGOLF_DIR / "train_gpt_mlx.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    iteration_id = f"{hypothesis}_{int(time.time())}"
    SCRIPT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = SCRIPT_BACKUP_DIR / f"{iteration_id}.py"
    shutil.copy2(script_path, backup)

    # Run training
    print(f"\n{'=' * 60}")
    print(f"Parameter Golf: {description}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Estimated params: {est_params:,}")
    print(f"Estimated artifact: {est_bytes:,} bytes")
    print(f"{'=' * 60}\n")

    venv_python = PGOLF_DIR / ".venv" / "bin" / "python3"
    cmd = [str(venv_python), str(script_path)]

    t0 = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(PGOLF_DIR),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    wall_time_s = time.time() - t0

    # Print output for agent parsing
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        return {
            "val_bpb": float("inf"),
            "val_loss": float("inf"),
            "artifact_size_bytes": 0,
            "param_count": est_params,
            "wall_time_s": wall_time_s,
            "status": "crash",
            "error": result.stderr[-500:],
            "description": description,
            "hypothesis": hypothesis,
        }

    # Parse metrics from output
    metrics = _parse_metrics(result.stdout)
    metrics["wall_time_s"] = wall_time_s
    metrics["description"] = description
    metrics["hypothesis"] = hypothesis
    metrics["estimated_params"] = est_params

    # Check constraints
    artifact_bytes = metrics.get("artifact_size_bytes", 0)
    if artifact_bytes > ARTIFACT_LIMIT_BYTES:
        metrics["status"] = "constraint_violation"
        metrics["constraint"] = (
            f"artifact {artifact_bytes:,} > {ARTIFACT_LIMIT_BYTES:,}"
        )
    else:
        metrics["status"] = "keep"

    return metrics


def load_past_results() -> list[dict[str, Any]]:
    """Load kept results for this task from pgolf_results.jsonl.

    Returns:
        List of dicts for kept runs matching TASK_NAME.
    """
    log = ExperimentLog(RESULTS_FILE)
    entries = log.load()
    return [
        asdict(e)
        for e in entries
        if e.experiment == TASK_NAME and e.status == "keep"
    ]


def log_result(metrics: dict[str, Any], config: dict[str, Any]) -> None:
    """Log experiment result to pgolf_results.jsonl.

    Args:
        metrics: Parsed metrics from the training run.
        config: Full config dict from propose().
    """
    log = ExperimentLog(RESULTS_FILE)
    entry = LogEntry(
        experiment=TASK_NAME,
        status=metrics.get("status", "keep"),
        val_bpb=metrics.get("val_bpb", 0.0),
        val_loss=metrics.get("val_loss", 0.0),
        train_loss=metrics.get("train_loss", 0.0),
        param_count=metrics.get("param_count", 0),
        wall_time_s=metrics.get("wall_time_s", 0.0),
        description=metrics.get("description", ""),
        config=config,
        metrics={
            k: v
            for k, v in metrics.items()
            if k not in ("val_bpb", "val_loss", "train_loss", "param_count",
                         "wall_time_s", "description", "status")
        },
    )
    log.log(entry)


def main() -> None:
    """Run the Parameter Golf experiment loop."""
    parser = argparse.ArgumentParser(
        description="Parameter Golf autorun recipe"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single iteration (for agent loop)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (200 steps)",
    )
    args = parser.parse_args()

    iterations = 1 if args.single else MAX_ITERATIONS

    for i in range(iterations):
        past = load_past_results()
        config = propose(past)
        print(f"\n{'=' * 60}")
        print(f"Iteration {i + 1}/{iterations}")
        print(f"Config: {config['description']}")
        print(f"{'=' * 60}")

        metrics = run(config, smoke=args.smoke)

        # Log result
        log_result(metrics, config)

        # Print JSON summary for agent parsing
        summary = {
            "iteration": i + 1,
            "task": TASK_NAME,
            "val_bpb": metrics.get("val_bpb"),
            "val_loss": metrics.get("val_loss"),
            "artifact_size_bytes": metrics.get("artifact_size_bytes"),
            "param_count": metrics.get("param_count"),
            "wall_time_s": metrics.get("wall_time_s"),
            "status": metrics.get("status"),
            "description": metrics.get("description"),
        }
        print("\n=== RESULT ===")
        print(json.dumps(summary, indent=2))

        # Check for plateau (5 iterations, <0.001 improvement)
        if len(past) >= 5:
            recent_bpb = [
                r.get("val_bpb", float("inf"))
                for r in sorted(
                    past[-5:],
                    key=lambda r: r.get("timestamp", 0),
                )
            ]
            best_recent = min(recent_bpb)
            current = metrics.get("val_bpb", float("inf"))
            if best_recent - current < 0.001:
                print("Plateau detected (5 iterations, <0.001 improvement).")
                if not args.single:
                    print("Consider pivoting to a new approach.")
                    break


if __name__ == "__main__":
    main()
