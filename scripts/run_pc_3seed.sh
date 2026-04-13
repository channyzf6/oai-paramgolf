#!/bin/bash
# Full 3-seed validation run for submission
# Use on 8xH100 for proper timing
#
# Usage: bash scripts/run_pc_3seed.sh

set -euo pipefail

echo "=== 3-Seed Validation Run ==="

# Download SP8192 dataset if not present
if [ ! -d "./data/datasets/fineweb10B_sp8192" ]; then
    echo "Downloading SP8192 dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp8192
fi

for SEED in 42 314 999; do
    echo ""
    echo "=== Seed $SEED ==="
    SEED=$SEED \
    PC_ENABLED=1 \
    PC_ALPHA=0.1 \
    PC_WARMUP_STEPS=200 \
    PC_DECAY_WITH_WARMDOWN=1 \
    QK_GAIN_INIT=5.25 \
    TTT_ENABLED=1 \
    TTT_LR=0.005 \
    TTT_EPOCHS=3 \
    RUN_ID="pc_seed${SEED}" \
    torchrun --standalone --nproc_per_node=8 train_gpt_pc.py

    echo "Seed $SEED complete. Log: logs/pc_seed${SEED}.txt"
done

echo ""
echo "=== All 3 seeds complete ==="
echo "Results:"
for SEED in 42 314 999; do
    echo "Seed $SEED:"
    grep -E "val_bpb|quantized|submission" "logs/pc_seed${SEED}.txt" | tail -5
done
