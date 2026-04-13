#!/bin/bash
# =============================================================================
# Parameter Golf - Run Baseline (1xH100)
# Expected: ~1.22 BPB, ~10 minutes
# =============================================================================
set -euo pipefail

cd /workspace/parameter-golf

echo "=== Running Baseline ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Expected BPB: ~1.2244"
echo ""

RUN_ID=baseline_verify \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=== Baseline Complete ==="
echo "Check logs/baseline_verify.txt for full output"
