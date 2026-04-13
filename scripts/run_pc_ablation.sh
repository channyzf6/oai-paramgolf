#!/bin/bash
# A/B test: Predictive Coding ON vs OFF
# Run on 1xH100 for quick iteration (shorter runs, ~200 steps)
#
# Usage: bash scripts/run_pc_ablation.sh

set -euo pipefail

echo "=== PC Ablation Study ==="

# Download SP8192 dataset if not present
if [ ! -d "./data/datasets/fineweb10B_sp8192" ]; then
    echo "Downloading SP8192 dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp8192
fi

echo ""
echo "--- Run 1: PC DISABLED (baseline) ---"
SEED=42 \
PC_ENABLED=0 \
QK_GAIN_INIT=5.0 \
TTT_ENABLED=0 \
ITERATIONS=1000 \
MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
RUN_ID="ablation_no_pc" \
torchrun --standalone --nproc_per_node=1 train_gpt_pc.py

echo ""
echo "--- Run 2: PC ENABLED (alpha=0.1) ---"
SEED=42 \
PC_ENABLED=1 \
PC_ALPHA=0.1 \
PC_WARMUP_STEPS=50 \
QK_GAIN_INIT=5.0 \
TTT_ENABLED=0 \
ITERATIONS=1000 \
MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
RUN_ID="ablation_pc_01" \
torchrun --standalone --nproc_per_node=1 train_gpt_pc.py

echo ""
echo "--- Run 3: PC ENABLED (alpha=0.2) ---"
SEED=42 \
PC_ENABLED=1 \
PC_ALPHA=0.2 \
PC_WARMUP_STEPS=50 \
QK_GAIN_INIT=5.0 \
TTT_ENABLED=0 \
ITERATIONS=1000 \
MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
RUN_ID="ablation_pc_02" \
torchrun --standalone --nproc_per_node=1 train_gpt_pc.py

echo ""
echo "=== Compare val_bpb across runs in logs/ ==="
echo "grep 'val_bpb' logs/ablation_*.txt"
