#!/bin/bash
# =============================================================================
# Parameter Golf - Run SOTA on 8xH100 SXM (official reproduction)
# Expected: 1.1147 BPB (±0.001), ~600s training + ~120s eval
# =============================================================================
set -euo pipefail

cd /workspace/parameter-golf

SOTA_DIR="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072"
SEED=${1:-42}

echo "=== Running SOTA Reproduction (8xH100 SXM) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Script: $SOTA_DIR/train_gpt.py"
echo "Expected BPB: 1.1147 (±0.001)"
echo "Seed: $SEED"
echo ""

# Install SOTA-specific deps
pip install -q flash-attn-hopper zstandard 2>/dev/null || true

# Copy SOTA script to working dir
cp "$SOTA_DIR/train_gpt.py" train_gpt_sota.py

RUN_ID="sota_repro_seed${SEED}" \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED="$SEED" \
torchrun --standalone --nproc_per_node=8 train_gpt_sota.py

echo ""
echo "=== SOTA Reproduction Complete ==="
echo "Check logs/sota_repro_seed${SEED}.txt for full output"
echo ""
echo "To run all 3 seeds:"
echo "  bash scripts/run_sota_8gpu.sh 42"
echo "  bash scripts/run_sota_8gpu.sh 314"
echo "  bash scripts/run_sota_8gpu.sh 999"
