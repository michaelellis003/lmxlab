#!/bin/bash
# Batch-invariant experiment runner for 22M competition config
# Logs BPP every 1000 steps up to 9000 steps (~32 min per run)
# Usage: bash scripts/run_batch_invariant.sh <experiment_name> [ENV_OVERRIDES...]
#
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment_name> [KEY=VALUE ...]}"
shift
OVERRIDES="$@"

PGOLF_DIR="/Users/michaelellis/Projects/parameter-golf"
RESULTS_DIR="/Users/michaelellis/Projects/lmxlab/pgolf/experiments/batch_invariant"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$RESULTS_DIR/${EXPERIMENT}_${TIMESTAMP}.log"

echo "=== Experiment: $EXPERIMENT ==="
echo "Overrides: $OVERRIDES"
echo "Log: $LOGFILE"
echo "Started: $(date)"

cd "$PGOLF_DIR"

# Base competition config (22M model)
export NUM_LAYERS=9
export UNIQUE_BLOCKS=9
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MODEL_DIM=512
export MLP_MULT=3
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS=8192
export GRAD_ACCUM_STEPS=1
export MLX_MAX_MICROBATCH_TOKENS=4096
export ITERATIONS=9000
export MAX_WALLCLOCK_SECONDS=3600
export VAL_BATCH_SIZE=65536
export EVAL_STRIDE=0
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=500

# Apply overrides
for kv in $OVERRIDES; do
    export "$kv"
done

# Run training
.venv/bin/python train_gpt_mlx.py 2>&1 | tee "$LOGFILE"

echo ""
echo "=== $EXPERIMENT complete ==="
echo "Learning curve:"
grep "val_bpb" "$LOGFILE" | grep "^step:" | while read line; do
    step=$(echo "$line" | grep -oP 'step:\K[0-9]+')
    bpb=$(echo "$line" | grep -oP 'val_bpb:\K[0-9.]+')
    echo "  step=$step bpb=$bpb"
done
echo ""
echo "Final result:"
grep "final_int8_zlib_roundtrip_exact" "$LOGFILE" | tail -1
