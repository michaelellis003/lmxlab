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
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    # HYP-039: XSA (Exclusive Self Attention) iso-step test
    # arXiv 2603.09078 — removes self-value component from attention output
    # Zero new params, ~2% overhead. Used in top 3 competition submissions.
    # Best local config: 6L+3u+4h/4kv+EVAL_STRIDE=256+NORMUON=1
    hyp039_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-039")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp039_runs)

    # Best local config baseline for comparison
    best_local_base = {
        "UNIQUE_BLOCKS": "3",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "NUM_LAYERS": "6",
        "EVAL_STRIDE": "256",
        "NORMUON": "1",
    }

    configs = [
        {
            "env_overrides": {**best_local_base},
            "description": "HYP-039 baseline (6L+3u+4h+stride256+normuon, no XSA)",
            "hypothesis": "HYP-039-baseline",
        },
        {
            "env_overrides": {**best_local_base, "XSA": "1"},
            "description": "XSA: Exclusive Self Attention (arXiv 2603.09078), all layers",
            "hypothesis": "HYP-039-xsa",
        },
        {
            "env_overrides": {**best_local_base, "XSA": "1", "VALUE_RESID": "1"},
            "description": "XSA + Value Residual (test interaction)",
            "hypothesis": "HYP-039-xsa-vr",
        },
        {
            "env_overrides": {**best_local_base, "XSA": "1", "DENSE_DWA": "1", "VALUE_RESID": "1"},
            "description": "XSA + DWA + Value Residual (triple combo test)",
            "hypothesis": "HYP-039-xsa-dwa-vr",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-040: Partial XSA — apply XSA only to last N layers
    # Competition PRs apply XSA selectively. Test if early layers
    # need self-value for feature formation.
    hyp040_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-040")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp040_runs)

    configs = [
        {
            "env_overrides": {**best_local_base, "XSA": "1", "VALUE_RESID": "1",
                              "XSA_START_LAYER": "3"},
            "description": "Partial XSA: last 3 of 6 layers only + VR",
            "hypothesis": "HYP-040-last3",
        },
        {
            "env_overrides": {**best_local_base, "XSA": "1", "VALUE_RESID": "1",
                              "XSA_START_LAYER": "4"},
            "description": "Partial XSA: last 2 of 6 layers only + VR",
            "hypothesis": "HYP-040-last2",
        },
        {
            "env_overrides": {**best_local_base, "XSA": "1", "VALUE_RESID": "1",
                              "XSA_START_LAYER": "5"},
            "description": "Partial XSA: last 1 of 6 layers only + VR",
            "hypothesis": "HYP-040-last1",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-041: Serialization + loss function experiments
    # FP16_EMBED is serialization-only (same training, different quantization).
    # Label smoothing and z-loss are loss function changes (iso-step, batch-independent).
    hyp041_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-041")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp041_runs)

    best_with_xsa = {
        **best_local_base,
        "XSA": "1",
        "XSA_START_LAYER": "4",
        "VALUE_RESID": "1",
    }

    configs = [
        {
            "env_overrides": {**best_with_xsa, "FP16_EMBED": "1"},
            "description": "FP16 embeddings (skip int8 quant on tok_emb)",
            "hypothesis": "HYP-041-fp16embed",
        },
        {
            "env_overrides": {**best_with_xsa, "LABEL_SMOOTH": "0.1"},
            "description": "Label smoothing 0.1 (reduce overconfidence)",
            "hypothesis": "HYP-041-labelsmooth",
        },
        {
            "env_overrides": {**best_with_xsa, "Z_LOSS": "1e-4"},
            "description": "Z-loss 1e-4 (PaLM-style logit stabilization)",
            "hypothesis": "HYP-041-zloss",
        },
        {
            "env_overrides": {**best_with_xsa, "FP16_EMBED": "1", "Z_LOSS": "1e-4"},
            "description": "FP16 embed + z-loss (best combo test)",
            "hypothesis": "HYP-041-combined",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-042: Z-loss coefficient sweep + softcap interaction
    hyp042_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-042")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp042_runs)

    best_full = {
        **best_local_base,
        "XSA": "1",
        "XSA_START_LAYER": "4",
        "VALUE_RESID": "1",
        "FP16_EMBED": "1",
    }

    configs = [
        {
            "env_overrides": {**best_full, "Z_LOSS": "1e-3"},
            "description": "Z-loss 1e-3 (10x stronger than 1e-4)",
            "hypothesis": "HYP-042-zloss-1e3",
        },
        {
            "env_overrides": {**best_full, "Z_LOSS": "1e-5"},
            "description": "Z-loss 1e-5 (10x weaker than 1e-4)",
            "hypothesis": "HYP-042-zloss-1e5",
        },
        {
            "env_overrides": {**best_full, "Z_LOSS": "5e-4"},
            "description": "Z-loss 5e-4 (5x stronger than 1e-4)",
            "hypothesis": "HYP-042-zloss-5e4",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-043: Focal loss — downweight easy tokens
    hyp043_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-043")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp043_runs)

    best_zloss = {**best_full, "Z_LOSS": "1e-4"}

    configs = [
        {
            "env_overrides": {**best_zloss, "FOCAL_GAMMA": "0.5"},
            "description": "Focal loss gamma=0.5 (mild hard-token focus)",
            "hypothesis": "HYP-043-focal05",
        },
        {
            "env_overrides": {**best_zloss, "FOCAL_GAMMA": "1.0"},
            "description": "Focal loss gamma=1.0 (moderate hard-token focus)",
            "hypothesis": "HYP-043-focal10",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-044: MiLe loss — entropy-weighted (upweights uncertain tokens)
    hyp044_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-044")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp044_runs)

    configs = [
        {
            "env_overrides": {**best_zloss, "MILE_GAMMA": "1.0"},
            "description": "MiLe loss gamma=1.0 (entropy-weighted, arXiv 2310.19531)",
            "hypothesis": "HYP-044-mile10",
        },
        {
            "env_overrides": {**best_zloss, "MILE_GAMMA": "0.5"},
            "description": "MiLe loss gamma=0.5 (mild entropy weighting)",
            "hypothesis": "HYP-044-mile05",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-045: Softcap tuning with z-loss
    # Now that z-loss handles logit stabilization, softcap 30 may be redundant.
    hyp045_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-045")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp045_runs)

    configs = [
        {
            "env_overrides": {**best_zloss, "LOGIT_SOFTCAP": "50.0"},
            "description": "Softcap 50 + z-loss (less clipping, z-loss handles scale)",
            "hypothesis": "HYP-045-cap50",
        },
        {
            "env_overrides": {**best_zloss, "LOGIT_SOFTCAP": "20.0"},
            "description": "Softcap 20 + z-loss (tighter clipping + smooth penalty)",
            "hypothesis": "HYP-045-cap20",
        },
        {
            "env_overrides": {**best_zloss, "LOGIT_SOFTCAP": "100.0"},
            "description": "Softcap 100 + z-loss (near-disabled softcap, z-loss only)",
            "hypothesis": "HYP-045-cap100",
        },
        {
            "env_overrides": {**best_zloss, "LOGIT_SOFTCAP": "75.0"},
            "description": "Softcap 75 + z-loss (between 50 and 100)",
            "hypothesis": "HYP-045-cap75",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-046: QK gain + embedding init with softcap 50 + z-loss
    hyp046_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-046")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp046_runs)

    best_cap50 = {**best_zloss, "LOGIT_SOFTCAP": "50.0"}

    configs = [
        {
            "env_overrides": {**best_cap50, "QK_GAIN_INIT": "1.0"},
            "description": "QK gain 1.0 + softcap 50 (less attention sharpening)",
            "hypothesis": "HYP-046-qk10",
        },
        {
            "env_overrides": {**best_cap50, "QK_GAIN_INIT": "2.0"},
            "description": "QK gain 2.0 + softcap 50 (more attention sharpening)",
            "hypothesis": "HYP-046-qk20",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-047: Embedding init std with softcap 50 + z-loss
    hyp047_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-047")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp047_runs)

    configs = [
        {
            "env_overrides": {**best_cap50, "TIED_EMBED_INIT_STD": "0.01"},
            "description": "Embed init std 0.01 (2x default, warmer logits)",
            "hypothesis": "HYP-047-std01",
        },
        {
            "env_overrides": {**best_cap50, "TIED_EMBED_INIT_STD": "0.002"},
            "description": "Embed init std 0.002 (0.4x default, cooler logits)",
            "hypothesis": "HYP-047-std002",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-048: Multi-seed validation of best config
    # All prior results are seed 1337 (n=1). Validate with seeds 42, 43, 44.
    hyp048_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-048")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp048_runs)

    configs = [
        {
            "env_overrides": {**best_cap50, "SEED": "42"},
            "description": "Best config seed 42 (multi-seed validation)",
            "hypothesis": "HYP-048-seed42",
        },
        {
            "env_overrides": {**best_cap50, "SEED": "43"},
            "description": "Best config seed 43 (multi-seed validation)",
            "hypothesis": "HYP-048-seed43",
        },
        {
            "env_overrides": {**best_cap50, "SEED": "44"},
            "description": "Best config seed 44 (multi-seed validation)",
            "hypothesis": "HYP-048-seed44",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-049: Training sequence length
    hyp049_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-049")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp049_runs)

    configs = [
        {
            "env_overrides": {**best_cap50, "TRAIN_SEQ_LEN": "512"},
            "description": "Train seq_len 512 (2x batch diversity, shorter context)",
            "hypothesis": "HYP-049-seq512",
        },
        {
            "env_overrides": {**best_cap50, "TRAIN_SEQ_LEN": "2048"},
            "description": "Train seq_len 2048 (0.5x diversity, longer context)",
            "hypothesis": "HYP-049-seq2048",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-050: Model dimension (384 vs 512)
    # dim=384 is 56% fewer params → faster → more steps in 600s.
    # But each step learns less. Net effect is unclear.
    hyp050_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-050")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp050_runs)

    configs = [
        {
            "env_overrides": {
                **best_local_base,
                "XSA": "1", "XSA_START_LAYER": "4", "VALUE_RESID": "1",
                "FP16_EMBED": "1", "Z_LOSS": "1e-4", "LOGIT_SOFTCAP": "50.0",
                "MODEL_DIM": "384", "NUM_HEADS": "4", "NUM_KV_HEADS": "4",
            },
            "description": "dim=384 + 4h (head_dim=96, smaller+faster model)",
            "hypothesis": "HYP-050-dim384",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-051: dim=384 vs 512 iso-step (200 steps each)
    hyp051_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-051")
        and r.get("wall_time_s", 0) > 30  # smoke test threshold for iso-step
    ]
    n = len(hyp051_runs)

    configs = [
        {
            "env_overrides": {
                **best_local_base,
                "XSA": "1", "XSA_START_LAYER": "4", "VALUE_RESID": "1",
                "FP16_EMBED": "1", "Z_LOSS": "1e-4", "LOGIT_SOFTCAP": "50.0",
                "MODEL_DIM": "512", "NUM_HEADS": "4", "NUM_KV_HEADS": "4",
                "ITERATIONS": "200", "MAX_WALLCLOCK_SECONDS": "300",
            },
            "description": "dim=512 iso-step baseline (200 steps)",
            "hypothesis": "HYP-051-dim512",
            "smoke": True,
        },
        {
            "env_overrides": {
                **best_local_base,
                "XSA": "1", "XSA_START_LAYER": "4", "VALUE_RESID": "1",
                "FP16_EMBED": "1", "Z_LOSS": "1e-4", "LOGIT_SOFTCAP": "50.0",
                "MODEL_DIM": "384", "NUM_HEADS": "4", "NUM_KV_HEADS": "4",
                "ITERATIONS": "200", "MAX_WALLCLOCK_SECONDS": "300",
            },
            "description": "dim=384 iso-step comparison (200 steps)",
            "hypothesis": "HYP-051-dim384",
            "smoke": True,
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-052: V normalization — RMSNorm on V like Q and K
    hyp052_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-052")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp052_runs)

    configs = [
        {
            "env_overrides": {**best_cap50, "V_NORM": "1"},
            "description": "V normalization: RMSNorm on V (completes QKV norm)",
            "hypothesis": "HYP-052-vnorm",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-053: Stochastic depth (drop layers during training)
    hyp053_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-053")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp053_runs)

    configs = [
        {
            "env_overrides": {**best_cap50, "STOCH_DEPTH": "0.1"},
            "description": "Stochastic depth p=0.1 (10% layer drop rate)",
            "hypothesis": "HYP-053-sd01",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-054: sp2048 vocabulary test
    hyp054_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-054")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp054_runs)

    sp2048_base = {
        "UNIQUE_BLOCKS": "3",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "NUM_LAYERS": "6",
        "EVAL_STRIDE": "256",
        "NORMUON": "1",
        "XSA": "1",
        "XSA_START_LAYER": "4",
        "VALUE_RESID": "1",
        "FP16_EMBED": "1",
        "Z_LOSS": "1e-4",
        "LOGIT_SOFTCAP": "50.0",
        "VOCAB_SIZE": "2048",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_2048_bpe.model",
        "DATA_PATH": "./data/datasets/fineweb10B_sp2048_local",
        "VAL_BATCH_SIZE": "65536",
    }

    configs = [
        {
            "env_overrides": {**sp2048_base},
            "description": "sp2048 vocab with best config (BPB comparison vs sp1024)",
            "hypothesis": "HYP-054-sp2048",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-055: MinGRU hybrid layers (replace some attention with minGRU)
    hyp055_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-055")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp055_runs)

    # Use sp2048 best config as base
    best_sp2048 = {
        "UNIQUE_BLOCKS": "3",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "NUM_LAYERS": "6",
        "EVAL_STRIDE": "256",
        "NORMUON": "1",
        "XSA": "1",
        "XSA_START_LAYER": "4",
        "VALUE_RESID": "1",
        "FP16_EMBED": "1",
        "Z_LOSS": "1e-4",
        "LOGIT_SOFTCAP": "50.0",
        "VOCAB_SIZE": "2048",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_2048_bpe.model",
        "DATA_PATH": "./data/datasets/fineweb10B_sp2048_local",
        "VAL_BATCH_SIZE": "65536",
    }

    configs = [
        {
            "env_overrides": {**best_sp2048, "MINGRU_LAYERS": "0"},
            "description": "MinGRU layer 0 only (early layer = RNN, rest = attention)",
            "hypothesis": "HYP-055-mingru0",
        },
        {
            "env_overrides": {**best_sp2048, "MINGRU_LAYERS": "0,1"},
            "description": "MinGRU layers 0,1 (first 2 = RNN, last 4 = attention)",
            "hypothesis": "HYP-055-mingru01",
        },
        {
            "env_overrides": {**best_sp2048, "MINGRU_LAYERS": "0,1,2"},
            "description": "MinGRU layers 0,1,2 (half RNN, half attention)",
            "hypothesis": "HYP-055-mingru012",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-056: sp4096 vocabulary
    hyp056_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-056")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp056_runs)

    sp4096_base = {
        "UNIQUE_BLOCKS": "3",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "NUM_LAYERS": "6",
        "EVAL_STRIDE": "256",
        "NORMUON": "1",
        "XSA": "1",
        "XSA_START_LAYER": "4",
        "VALUE_RESID": "1",
        "FP16_EMBED": "1",
        "Z_LOSS": "1e-4",
        "LOGIT_SOFTCAP": "50.0",
        "VOCAB_SIZE": "4096",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_4096_bpe.model",
        "DATA_PATH": "./data/datasets/fineweb10B_sp4096_local",
        "VAL_BATCH_SIZE": "65536",
    }

    configs = [
        {
            "env_overrides": {**sp4096_base},
            "description": "sp4096 vocab with best config (BPB comparison)",
            "hypothesis": "HYP-056-sp4096",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-057: Causal convolution hybrid layers
    hyp057_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-057")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp057_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "CONV_LAYERS": "0"},
            "description": "CausalConv block 0 (layers 0,3 = conv, rest = attention)",
            "hypothesis": "HYP-057-conv0",
        },
        {
            "env_overrides": {**best_sp2048, "CONV_LAYERS": "0,1"},
            "description": "CausalConv blocks 0,1 (layers 0,1,3,4 = conv, 2,5 = attention)",
            "hypothesis": "HYP-057-conv01",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-058: Pre-attention causal conv (inside attention, minimal overhead)
    hyp058_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-058")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp058_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "PRE_CONV": "4"},
            "description": "Pre-attention causal conv k=4 (local context mixing)",
            "hypothesis": "HYP-058-preconv4",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-059: Vectorized non-attention layers (no Python loops)
    hyp059_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-059")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp059_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "CONV_LAYERS": "0"},
            "description": "Conv1d k=32 block 0 (nn.Conv1d, no loop, layers 0,3)",
            "hypothesis": "HYP-059-conv32",
        },
        {
            "env_overrides": {**best_sp2048, "SGU_LAYERS": "0"},
            "description": "gMLP SGU block 0 (causal spatial mixing, layers 0,3)",
            "hypothesis": "HYP-059-sgu0",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-060: Eval stride tuning with sp2048
    hyp060_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-060")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp060_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "EVAL_STRIDE": "128"},
            "description": "sp2048 + eval stride 128 (tighter sliding window)",
            "hypothesis": "HYP-060-stride128",
        },
        {
            "env_overrides": {**best_sp2048, "EVAL_STRIDE": "64"},
            "description": "sp2048 + eval stride 64 (competition standard)",
            "hypothesis": "HYP-060-stride64",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-061: RDOQ — keep output projections at fp16 (most sensitive to quant)
    hyp061_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-061")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp061_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048,
                              "INT8_KEEP_FLOAT_NAME_PATTERNS": "tok_emb,proj"},
            "description": "RDOQ: keep tok_emb + all proj layers at fp16",
            "hypothesis": "HYP-061-fp16proj",
        },
        {
            "env_overrides": {**best_sp2048,
                              "INT8_KEEP_FLOAT_NAME_PATTERNS": "tok_emb,attn_scale,mlp_scale,q_gain,resid_mix"},
            "description": "RDOQ: keep tok_emb + all control tensors at fp16",
            "hypothesis": "HYP-061-fp16ctrl",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-063: Eval-time temperature scaling (compression theory insight)
    hyp063_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-063")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp063_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "EVAL_TEMP": "0.9"},
            "description": "Eval temp 0.9 (sharpen — model may be underconfident)",
            "hypothesis": "HYP-063-temp09",
        },
        {
            "env_overrides": {**best_sp2048, "EVAL_TEMP": "0.8"},
            "description": "Eval temp 0.8 (sharpen more)",
            "hypothesis": "HYP-063-temp08",
        },
        {
            "env_overrides": {**best_sp2048, "EVAL_TEMP": "1.1"},
            "description": "Eval temp 1.1 (soften — model may be overconfident)",
            "hypothesis": "HYP-063-temp11",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-064: Gradient accumulation (bias-variance tradeoff from statistics)
    # Core problem: "What is the optimal noise-averaging vs step-count tradeoff?"
    hyp064_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-064")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp064_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "GRAD_ACCUM_STEPS": "2"},
            "description": "Grad accum 2 (16K effective batch, 50% fewer steps)",
            "hypothesis": "HYP-064-accum2",
        },
        {
            "env_overrides": {**best_sp2048, "GRAD_ACCUM_STEPS": "4"},
            "description": "Grad accum 4 (32K effective batch, 75% fewer steps)",
            "hypothesis": "HYP-064-accum4",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-065: BigramHash with sp2048 (context mixing from coding theory)
    # Core problem: "How do we combine local (bigram) and global (attention) predictions?"
    # From coding theory: context mixing improves compression by combining models.
    hyp065_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-065")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp065_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "BIGRAM_VOCAB_SIZE": "4096", "BIGRAM_DIM": "64"},
            "description": "BigramHash 4096 dim=64 with sp2048 (local context mixing)",
            "hypothesis": "HYP-065-bigram4k",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-066: Eval-time context mixing (from coding theory / James-Stein shrinkage)
    hyp066_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-066")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp066_runs)

    unigram_path = "./data/unigram_sp2048_logprobs.npy"

    configs = [
        {
            "env_overrides": {**best_sp2048,
                              "EVAL_MIX_ALPHA": "0.01",
                              "EVAL_MIX_LOGPROBS": unigram_path},
            "description": "Context mixing alpha=0.01 (1% unigram, 99% transformer)",
            "hypothesis": "HYP-066-mix001",
        },
        {
            "env_overrides": {**best_sp2048,
                              "EVAL_MIX_ALPHA": "0.05",
                              "EVAL_MIX_LOGPROBS": unigram_path},
            "description": "Context mixing alpha=0.05 (5% unigram, 95% transformer)",
            "hypothesis": "HYP-066-mix005",
        },
        {
            "env_overrides": {**best_sp2048,
                              "EVAL_MIX_ALPHA": "0.001",
                              "EVAL_MIX_LOGPROBS": unigram_path},
            "description": "Context mixing alpha=0.001 (0.1% unigram, 99.9% transformer)",
            "hypothesis": "HYP-066-mix0001",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-067: Progressive sequence length (multigrid-inspired)
    # Train at seq_len=512 for first 60%, then switch to 1024
    hyp067_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-067")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp067_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "PROGRESSIVE_SEQ": "0.6"},
            "description": "Progressive seq: 60% at 512, 40% at 1024 (multigrid)",
            "hypothesis": "HYP-067-prog06",
        },
        {
            "env_overrides": {**best_sp2048, "PROGRESSIVE_SEQ": "0.4"},
            "description": "Progressive seq: 40% at 512, 60% at 1024",
            "hypothesis": "HYP-067-prog04",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-068: Shared Q-K projections (Mahalanobis attention, from metric learning)
    hyp068_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-068")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp068_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048, "SHARE_QK": "1"},
            "description": "Shared Q-K weights (Mahalanobis attention, saves 11% params)",
            "hypothesis": "HYP-068-shareqk",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-069: Proper train/val split with more training data
    # Old: 8M train + 2M val from SAME val shard (1.95 epochs, data overlap)
    # New: 20M train from TRAIN shard + 2M val from VAL shard (0.78 epochs, no overlap)
    hyp069_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-069")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp069_runs)

    best_sp2048_v2 = dict(best_sp2048)
    best_sp2048_v2["DATA_PATH"] = "./data/datasets/fineweb10B_sp2048_local_v2"

    configs = [
        {
            "env_overrides": {**best_sp2048_v2},
            "description": "Proper train/val split: 20M train tokens, 0 data overlap",
            "hypothesis": "HYP-069-properdata",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-070: Re-validate key findings on proper v2 dataset
    hyp070_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-070")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp070_runs)

    # Baseline WITHOUT our innovations (v2 data)
    v2_base_plain = {
        "UNIQUE_BLOCKS": "3",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "NUM_LAYERS": "6",
        "EVAL_STRIDE": "256",
        "NORMUON": "1",
        "VOCAB_SIZE": "2048",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_2048_bpe.model",
        "DATA_PATH": "./data/datasets/fineweb10B_sp2048_local_v2",
        "VAL_BATCH_SIZE": "65536",
    }

    configs = [
        {
            "env_overrides": {**v2_base_plain},
            "description": "v2 baseline: sp2048, NO XSA/VR/z-loss/softcap50",
            "hypothesis": "HYP-070-v2plain",
        },
        {
            "env_overrides": {**best_sp2048_v2},
            "description": "v2 full stack: sp2048 + XSA+VR+z-loss+softcap50",
            "hypothesis": "HYP-070-v2full",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-071: Affine RMSNorm (learnable γ,β — from approximation theory)
    hyp071_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-071")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp071_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048_v2, "AFFINE_NORM": "1"},
            "description": "Affine RMSNorm: learned γ,β (preconditioning, ~7K extra params)",
            "hypothesis": "HYP-071-affine",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-072: Weight decay (James-Stein shrinkage, from estimation theory)
    # WD gradient = λ*θ is batch-independent. But relative strength vs CE gradient
    # scales with sqrt(batch). At 8K, use λ = 0.04/8 ≈ 0.005 to match GPU effect.
    hyp072_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-072")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp072_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048_v2, "WEIGHT_DECAY": "0.005"},
            "description": "Weight decay 0.005 (≈0.04 at 64K batch equivalent)",
            "hypothesis": "HYP-072-wd005",
        },
        {
            "env_overrides": {**best_sp2048_v2, "WEIGHT_DECAY": "0.001"},
            "description": "Weight decay 0.001 (very mild shrinkage)",
            "hypothesis": "HYP-072-wd001",
        },
    ]

    if n < len(configs):
        return configs[n]

    # HYP-073: Random MLP features (from random matrix theory / kernel methods)
    hyp073_runs = [
        r for r in past_results
        if r.get("config", {}).get("hypothesis", "").startswith("HYP-073")
        and r.get("wall_time_s", 0) > 500
    ]
    n = len(hyp073_runs)

    configs = [
        {
            "env_overrides": {**best_sp2048_v2, "RANDOM_MLP_FC": "1"},
            "description": "Random MLP fc (freeze input proj, learn output only)",
            "hypothesis": "HYP-073-randomfc",
        },
    ]

    if n < len(configs):
        return configs[n]

    return {
        "env_overrides": {"ITERATIONS": "5000"},
        "description": "done",
        "hypothesis": "DONE",
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
    # Write stdout/stderr to temp files to avoid accumulating
    # large strings in the parent process across many runs.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False
    ) as stdout_f, tempfile.NamedTemporaryFile(
        mode="w", suffix=".err", delete=False
    ) as stderr_f:
        stdout_path = stdout_f.name
        stderr_path = stderr_f.name
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(PGOLF_DIR),
            stdout=stdout_f,
            stderr=stderr_f,
            timeout=1800,
        )
    wall_time_s = time.time() - t0

    try:
        with open(stdout_path) as f:
            stdout_text = f.read()
        with open(stderr_path) as f:
            stderr_text = f.read()
    finally:
        os.unlink(stdout_path)
        os.unlink(stderr_path)

    # Print output for agent parsing
    print(stdout_text[-3000:] if len(stdout_text) > 3000 else stdout_text)
    if result.returncode != 0:
        print(f"STDERR:\n{stderr_text[-2000:]}")
        return {
            "val_bpb": float("inf"),
            "val_loss": float("inf"),
            "artifact_size_bytes": 0,
            "param_count": est_params,
            "wall_time_s": wall_time_s,
            "status": "crash",
            "error": stderr_text[-500:],
            "description": description,
            "hypothesis": hypothesis,
        }

    # Parse metrics from output
    metrics = _parse_metrics(stdout_text)
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

        # Explicit gc + brief pause to let OS reclaim subprocess memory
        gc.collect()
        time.sleep(2)

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
