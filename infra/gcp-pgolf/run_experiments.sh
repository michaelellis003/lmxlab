#!/bin/bash
# Cost-controlled experiment runner for GCP A100
# Tracks compute time and stops before exceeding budget
#
# Usage: bash run_experiments.sh [max_budget_dollars]
# Default budget: $25 (conservative, well under $50 limit)
#
set -euo pipefail

MAX_BUDGET="${1:-25}"
COST_PER_HOUR=3.67  # a2-highgpu-1g on-demand
START_TIME=$(date +%s)
REPO_DIR="$HOME/parameter-golf"

# Cost tracking
cost_so_far() {
    local now=$(date +%s)
    local elapsed_hours=$(echo "($now - $START_TIME) / 3600.0" | bc -l)
    echo "scale=2; $elapsed_hours * $COST_PER_HOUR" | bc -l
}

check_budget() {
    local cost=$(cost_so_far)
    local elapsed=$(($(date +%s) - START_TIME))
    local hours=$(echo "scale=1; $elapsed / 3600" | bc -l)
    echo "[COST] $hours hrs elapsed, ~\$$cost spent (budget: \$$MAX_BUDGET)"

    if (( $(echo "$cost > $MAX_BUDGET" | bc -l) )); then
        echo "[BUDGET EXCEEDED] Stopping! Cost \$$cost > \$$MAX_BUDGET"
        echo "Results saved in $REPO_DIR/logs/"
        exit 1
    fi
}

run_experiment() {
    local name="$1"
    shift
    local env_vars="$@"

    check_budget
    echo ""
    echo "=== Experiment: $name ==="
    echo "Env: $env_vars"
    echo "Started: $(date)"

    local exp_start=$(date +%s)

    cd "$REPO_DIR"
    env $env_vars python train_gpt.py 2>&1 | tee "logs/${name}_$(date +%Y%m%d_%H%M%S).log"

    local exp_end=$(date +%s)
    local exp_mins=$(( (exp_end - exp_start) / 60 ))
    echo "=== $name complete in ${exp_mins}m ==="

    # Extract BPB from log
    grep "final_int8_zlib_roundtrip " "logs/${name}"_*.log | tail -1 || true
    echo ""
}

echo "================================================"
echo "Parameter Golf A100 Experiment Runner"
echo "Budget: \$$MAX_BUDGET (cost: \$$COST_PER_HOUR/hr)"
echo "Max runtime: $(echo "scale=1; $MAX_BUDGET / $COST_PER_HOUR" | bc -l) hours"
echo "Started: $(date)"
echo "================================================"

# --- Setup (one-time) ---
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning parameter-golf repo..."
    git clone https://github.com/openai/parameter-golf.git "$REPO_DIR"
fi

cd "$REPO_DIR"
mkdir -p logs

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: No GPU detected!"
    exit 1
}

# --- Download sp1024 data if not present ---
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "Downloading sp1024 data (free ingress)..."
    python data/cached_challenge_fineweb.py --train-shards 10 --variant sp1024
    check_budget
fi

# Use the GPU submission script
TRAIN_SCRIPT="records/track_10min_16mb/2026-03-21_XSA_VR_ZLoss_sp2048/train_gpt.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: GPU training script not found at $TRAIN_SCRIPT"
    echo "Make sure you've pushed the latest code from your local repo"
    exit 1
fi

# --- Experiments (sp1024 first, no tokenization needed) ---

# Experiment 1: Baseline (no innovations)
run_experiment "01_baseline" \
    VOCAB_SIZE=1024 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024

# Experiment 2: XSA + Value Residual
run_experiment "02_xsa_vr" \
    VOCAB_SIZE=1024 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    XSA=1 XSA_START_LAYER=8 VALUE_RESID=1

# Experiment 3: Full innovation stack (without sp2048)
run_experiment "03_full_sp1024" \
    VOCAB_SIZE=1024 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 \
    Z_LOSS=1e-4 LOGIT_SOFTCAP=50 FP16_EMBED=1

# Experiment 4: Full + random MLP fc
run_experiment "04_full_randomfc" \
    VOCAB_SIZE=1024 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 \
    Z_LOSS=1e-4 LOGIT_SOFTCAP=50 FP16_EMBED=1 \
    RANDOM_MLP_FC=1 ORTHO_RANDOM_FC=1

# Experiment 5: Full + SWA (should work at larger batch!)
run_experiment "05_full_swa" \
    VOCAB_SIZE=1024 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 \
    Z_LOSS=1e-4 LOGIT_SOFTCAP=50 FP16_EMBED=1 \
    SWA_ENABLED=1 SWA_START_FRAC=0.5

check_budget

echo ""
echo "================================================"
echo "All experiments complete!"
echo "Total cost: ~\$$(cost_so_far)"
echo "Results in: $REPO_DIR/logs/"
echo ""
echo "IMPORTANT: Stop or delete this VM to avoid ongoing charges!"
echo "  gcloud compute instances delete pgolf-a100 --project=pgolf-lmxlab --zone=us-central1-a"
echo "================================================"
