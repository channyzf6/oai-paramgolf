#!/bin/bash
# =============================================================================
# Parameter Golf - Pod Setup Script
# Run this on a fresh RunPod pod to get everything ready.
# =============================================================================
set -euo pipefail

echo "=== Parameter Golf Pod Setup ==="
echo "Timestamp: $(date -u)"

cd /workspace

# Clone repo if not already present
if [ ! -d "parameter-golf" ]; then
    echo ">>> Cloning parameter-golf repo..."
    git clone https://github.com/openai/parameter-golf.git
else
    echo ">>> Repo already exists, pulling latest..."
    cd parameter-golf && git pull && cd ..
fi

cd parameter-golf

# Install dependencies
echo ">>> Installing Python dependencies..."
pip install -q sentencepiece numpy tqdm torch huggingface-hub datasets tiktoken flash-attn-hopper 2>/dev/null || true
pip install -q lzma zstandard 2>/dev/null || true

# Download dataset (full 80 shards + validation)
echo ">>> Downloading FineWeb dataset (sp1024, full)..."
python3 data/cached_challenge_fineweb.py --variant sp1024

echo ">>> Verifying dataset..."
TRAIN_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "    Train shards: $TRAIN_SHARDS"
echo "    Val shards:   $VAL_SHARDS"

echo ""
echo "=== Setup Complete ==="
echo "Run scripts:"
echo "  bash scripts/run_baseline.sh       # Baseline on 1xH100"
echo "  bash scripts/run_sota.sh           # SOTA on 8xH100"
echo "  bash scripts/run_sota_1gpu.sh      # SOTA on 1xH100 (quick test)"
