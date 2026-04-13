#!/bin/bash
# =============================================================================
# Parameter Golf - Run SOTA on 1xH100 (quick test, NOT for scoring)
# This verifies the SOTA script compiles and runs. BPB will differ from
# official scores since those require 8xH100 SXM for correct timing.
# =============================================================================
set -euo pipefail

cd /workspace/parameter-golf

SOTA_DIR="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072"

echo "=== Running SOTA Script (1xH100 test) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Script: $SOTA_DIR/train_gpt.py"
echo "NOTE: This is a quick validation run, not for official scoring."
echo ""

# Install SOTA-specific deps
pip install -q flash-attn-hopper zstandard 2>/dev/null || true

# Copy SOTA script to working dir
cp "$SOTA_DIR/train_gpt.py" train_gpt_sota.py

RUN_ID=sota_1gpu_test \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=42 \
torchrun --standalone --nproc_per_node=1 train_gpt_sota.py

echo ""
echo "=== SOTA 1xGPU Test Complete ==="
echo "Check logs/sota_1gpu_test.txt for full output"
