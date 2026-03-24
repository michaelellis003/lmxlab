#!/bin/bash
# Validate batch-dependent findings on L4 GPU (24GB, ~64K batch)
# Budget: $2.50 compute (10 experiments × 15 min × $0.70/hr)
# MUST DELETE VM AFTER THIS SCRIPT FINISHES
#
set -euo pipefail

COST_PER_HOUR=0.70
START_TIME=$(date +%s)
MAX_BUDGET=5  # Hard cap $5 (safety margin)

cost_so_far() {
    local elapsed=$(( $(date +%s) - START_TIME ))
    echo "scale=2; $elapsed / 3600 * $COST_PER_HOUR" | bc -l
}

check_budget() {
    local cost=$(cost_so_far)
    echo "[COST] ~\$$cost spent (cap: \$$MAX_BUDGET)"
    if (( $(echo "$cost > $MAX_BUDGET" | bc -l) )); then
        echo "[BUDGET EXCEEDED] Stopping!"
        exit 1
    fi
}

log_result() {
    local name="$1" log="$2"
    local bpb=$(grep "final_int8_zlib_roundtrip " "$log" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    local steps=$(grep "stopping_early" "$log" | grep -oP "step:\K[0-9]+")
    echo "RESULT: $name | BPB=$bpb | steps=$steps"
    echo "$name,$bpb,$steps" >> ~/results_summary.csv
}

cd ~

# --- Setup ---
echo "=== L4 Batch-Dependent Validation ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Budget: \$$MAX_BUDGET"
nvidia-smi

# Clone repo if needed
if [ ! -d parameter-golf ]; then
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf
mkdir -p logs

# Download sp1024 data (10 shards, free ingress)
if [ ! -f data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ]; then
    echo "Downloading sp1024 data..."
    pip install -q huggingface_hub sentencepiece
    python3 data/cached_challenge_fineweb.py --train-shards 10 --variant sp1024
fi

check_budget

# Copy our training script
SCRIPT=~/train_gpt.py  # uploaded via SCP

# Common env vars for L4
# L4 has 24GB — use 16K batch (conservative to avoid OOM with 11L model)
# Can use grad accum to simulate larger effective batch
export TRAIN_BATCH_TOKENS=16384
export GRAD_ACCUM_STEPS=2  # Effective 32K batch
export MAX_WALLCLOCK_SECONDS=600
export VOCAB_SIZE=1024
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export NUM_LAYERS=11
export UNIQUE_BLOCKS=11
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MODEL_DIM=512
export MLP_MULT=3
export EVAL_STRIDE=256  # stride=64 is too slow on L4 (60+ min eval). 256 is fast enough.
export EVAL_BATCH_SEQS=16  # Larger eval batch for speed

echo ""
echo "name,val_bpb,steps" > ~/results_summary.csv

# --- Experiment 1: Baseline (no innovations) ---
echo "=== Exp 1: Baseline ==="
python3 "$SCRIPT" 2>&1 | tee logs/01_baseline.log
log_result "01_baseline" logs/01_baseline.log
check_budget

# --- Experiment 2: XSA + VR ---
echo "=== Exp 2: XSA + VR ==="
XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 \
python3 "$SCRIPT" 2>&1 | tee logs/02_xsa_vr.log
log_result "02_xsa_vr" logs/02_xsa_vr.log
check_budget

# --- Experiment 3: Full stack (XSA+VR+z-loss+softcap50) ---
echo "=== Exp 3: Full stack ==="
XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 Z_LOSS=1e-4 LOGIT_SOFTCAP=50 FP16_EMBED=1 \
python3 "$SCRIPT" 2>&1 | tee logs/03_full_stack.log
log_result "03_full_stack" logs/03_full_stack.log
check_budget

# --- Experiment 4: Full + SWA (BATCH-DEPENDENT: failed at 8K, test at 32K) ---
echo "=== Exp 4: Full + SWA ==="
XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 Z_LOSS=1e-4 LOGIT_SOFTCAP=50 FP16_EMBED=1 \
SWA_ENABLED=1 SWA_START_FRAC=0.5 \
python3 "$SCRIPT" 2>&1 | tee logs/04_full_swa.log
log_result "04_full_swa" logs/04_full_swa.log
check_budget

# --- Experiment 5: Full + momentum 0.99 (BATCH-DEPENDENT: failed at 8K) ---
echo "=== Exp 5: Full + mom 0.99 ==="
XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 Z_LOSS=1e-4 LOGIT_SOFTCAP=50 FP16_EMBED=1 \
MUON_MOMENTUM=0.99 \
python3 "$SCRIPT" 2>&1 | tee logs/05_full_mom99.log
log_result "05_full_mom99" logs/05_full_mom99.log
check_budget

# --- Experiment 6: Full + weight decay (GPU script has built-in WD) ---
echo "=== Exp 6: Full + WD 0.04 ==="
XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 Z_LOSS=1e-4 LOGIT_SOFTCAP=50 FP16_EMBED=1 \
MUON_WD=0.04 ADAM_WD=0.01 \
python3 "$SCRIPT" 2>&1 | tee logs/06_full_wd.log
log_result "06_full_wd" logs/06_full_wd.log
check_budget

# --- Experiment 7: Full + random MLP fc (CROSS-DISCIPLINARY: JL lemma) ---
echo "=== Exp 7: Full + random fc ==="
XSA=1 XSA_START_LAYER=8 VALUE_RESID=1 Z_LOSS=1e-4 LOGIT_SOFTCAP=50 FP16_EMBED=1 \
RANDOM_MLP_FC=1 ORTHO_RANDOM_FC=1 \
python3 "$SCRIPT" 2>&1 | tee logs/07_full_randomfc.log
log_result "07_full_randomfc" logs/07_full_randomfc.log
check_budget

echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "Total cost: ~\$(cost_so_far)"
echo ""
echo "Results:"
cat ~/results_summary.csv
echo ""
echo "IMPORTANT: DELETE THIS VM NOW!"
echo "gcloud compute instances delete pgolf-gpu --project=pgolf-lmxlab --zone=us-central1-a --quiet"
