#!/usr/bin/env bash
set -euo pipefail

echo "Converting Qwen2.5-7B to TensorRT-LLM checkpoint format..."

python3 ~/TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir hf_models/qwen2.5-7b-instruct \
  --output_dir checkpoints/qwen2.5-7b \
  --dtype float16 \
  --tp_size 1 \
  --pp_size 1

echo "✓ Checkpoint conversion complete"
echo "  Output: checkpoints/qwen2.5-7b"
echo "  Format: FP16, TP=1, PP=1"
