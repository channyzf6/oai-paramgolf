#!/bin/bash
# =============================================================================
# PARAMETER GOLF QUICKSTART
# Copy-paste this entire block into your RunPod terminal.
# =============================================================================

cd /workspace && \
git clone https://github.com/openai/parameter-golf.git 2>/dev/null; \
cd parameter-golf && \
pip install -q sentencepiece huggingface-hub datasets zstandard 2>/dev/null && \
echo ">>> Downloading dataset (this takes a few minutes)..." && \
python3 data/cached_challenge_fineweb.py --variant sp1024 && \
echo ">>> Dataset ready. Running baseline..." && \
RUN_ID=baseline_v1 \
torchrun --standalone --nproc_per_node=1 train_gpt.py && \
echo ">>> Baseline complete! Check output above for val_bpb."
