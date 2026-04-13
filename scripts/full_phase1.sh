#!/bin/bash
# =============================================================================
# Parameter Golf - Full Phase 1 Pipeline
# Run this on a fresh RunPod pod to execute the complete Phase 1:
#   1. Setup environment + download data
#   2. Run baseline and verify ~1.22 BPB
#   3. (On 8xH100 only) Run SOTA and verify ~1.1147 BPB
# =============================================================================
set -euo pipefail

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

echo "============================================"
echo " Parameter Golf - Phase 1"
echo " GPUs: ${NUM_GPUS}x ${GPU_NAME}"
echo " Date: $(date -u)"
echo "============================================"
echo ""

# ---- Step 1: Setup ----
echo ">>> Step 1: Environment Setup"
cd /workspace

if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
else
    cd parameter-golf && git pull && cd /workspace
fi

cd parameter-golf

# Install core deps (most are in the RunPod PyTorch image already)
pip install -q sentencepiece numpy tqdm huggingface-hub datasets tiktoken 2>/dev/null || true
pip install -q zstandard 2>/dev/null || true

# flash_attn_interface (FA3) - needed for SOTA script
pip install -q flash-attn-hopper 2>/dev/null || {
    echo "WARN: flash-attn-hopper not available, SOTA script may fail"
    echo "      Baseline will still work fine."
}

# Download dataset
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo ">>> Downloading FineWeb dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo ">>> Dataset already present, skipping download"
fi

TRAIN_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "    Train shards: $TRAIN_SHARDS | Val shards: $VAL_SHARDS"
echo ""

# ---- Step 2: Baseline ----
echo ">>> Step 2: Running Baseline"
echo "    Expected BPB: ~1.2244"

RUN_ID=baseline_phase1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee /tmp/baseline_output.txt

# Extract final BPB
BASELINE_BPB=$(grep "final_int8_zlib_roundtrip_exact" /tmp/baseline_output.txt | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
echo ""
echo ">>> Baseline BPB: ${BASELINE_BPB:-FAILED}"
echo ""

# ---- Step 3: SOTA (8xH100 only) ----
if [ "$NUM_GPUS" -ge 8 ]; then
    echo ">>> Step 3: Running SOTA Reproduction (8xH100)"
    echo "    Expected BPB: ~1.1147"

    SOTA_DIR="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072"
    cp "$SOTA_DIR/train_gpt.py" train_gpt_sota.py

    RUN_ID=sota_phase1_seed42 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    SEED=42 \
    torchrun --standalone --nproc_per_node=8 train_gpt_sota.py 2>&1 | tee /tmp/sota_output.txt

    SOTA_BPB=$(grep "final_int8_zlib_roundtrip_exact\|final_int6_sliding_window_exact" /tmp/sota_output.txt | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
    echo ""
    echo ">>> SOTA BPB: ${SOTA_BPB:-FAILED}"
elif [ "$NUM_GPUS" -ge 1 ]; then
    echo ">>> Step 3: Running SOTA Quick Test (1xH100 - not for scoring)"
    echo "    This verifies the script runs. BPB may differ from 8xH100 results."

    SOTA_DIR="records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072"
    cp "$SOTA_DIR/train_gpt.py" train_gpt_sota.py

    RUN_ID=sota_1gpu_test \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    SEED=42 \
    MAX_WALLCLOCK_SECONDS=600 \
    torchrun --standalone --nproc_per_node=1 train_gpt_sota.py 2>&1 | tee /tmp/sota_output.txt

    SOTA_BPB=$(grep "final_int8_zlib_roundtrip_exact\|final_int6_sliding_window_exact" /tmp/sota_output.txt | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
    echo ""
    echo ">>> SOTA 1xGPU BPB: ${SOTA_BPB:-FAILED} (not comparable to 8xH100 scores)"
fi

echo ""
echo "============================================"
echo " Phase 1 Complete"
echo " Baseline BPB: ${BASELINE_BPB:-N/A}"
echo " SOTA BPB:     ${SOTA_BPB:-N/A}"
echo "============================================"
