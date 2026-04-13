#!/bin/bash
# Run the Predictive Coding model with SOTA baseline settings
# Use on 8xH100 for proper timing validation
#
# Usage: bash scripts/run_pc_baseline.sh [SEED]
#   SEED defaults to 42

set -euo pipefail

SEED="${1:-42}"

echo "=== Predictive Coding Transformer (SOTA + PC) ==="
echo "Seed: $SEED"
echo "PC enabled, alpha=0.1, warmup=200 steps"

# Download SP8192 dataset if not present
if [ ! -d "./data/datasets/fineweb10B_sp8192" ]; then
    echo "Downloading SP8192 dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp8192
fi

# Run training + eval
SEED=$SEED \
PC_ENABLED=1 \
PC_ALPHA=0.1 \
PC_WARMUP_STEPS=200 \
PC_DECAY_WITH_WARMDOWN=1 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt_pc.py
