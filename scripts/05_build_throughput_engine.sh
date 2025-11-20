#!/usr/bin/env bash
set -euo pipefail

echo "Building THROUGHPUT-optimized TensorRT engine..."
echo "Config: batch_size=8, FP16, FlashAttention, Paged KV-cache"

trtllm-build \
  --checkpoint_dir checkpoints/qwen2.5-7b \
  --output_dir engines/throughput \
  --max_batch_size 8 \
  --max_input_len 512 \
  --max_seq_len 640 \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --context_fmha enable \
  --paged_kv_cache enable \
  --log_level info

echo "✓ Throughput engine built at engines/throughput/"
