#!/usr/bin/env python3
import json, matplotlib.pyplot as plt
from pathlib import Path

pt_single = json.loads(Path("results/pytorch_single.json").read_text())
trt_lat = json.loads(Path("results/tensorrt_latency.json").read_text())
pt_batch = json.loads(Path("results/pytorch_batch.json").read_text())
trt_thr = json.loads(Path("results/tensorrt_throughput.json").read_text())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Latency comparison
axes[0,0].bar(['PyTorch', 'TensorRT'], [pt_single['latency_p50_s'], trt_lat['latency_p50_s']], color=['#1f77b4', '#ff7f0e'])
axes[0,0].set_ylabel('Latency P50 (s)')
axes[0,0].set_title('Single Request Latency (Lower is Better)')

# Throughput comparison
axes[0,1].bar(['PyTorch', 'TensorRT'], [pt_single['tokens_per_sec'], trt_lat['tokens_per_sec']], color=['#1f77b4', '#ff7f0e'])
axes[0,1].set_ylabel('Tokens/sec')
axes[0,1].set_title('Single Request Throughput (Higher is Better)')

# Batch throughput - use correct keys
pt_batch_tps = pt_batch.get('total_tokens_per_sec', pt_batch.get('tokens_per_sec', 0))
trt_thr_tps = trt_thr.get('total_tokens_per_sec', trt_thr.get('tokens_per_sec', 0))
axes[1,0].bar(['PyTorch', 'TensorRT'], [pt_batch_tps, trt_thr_tps], color=['#1f77b4', '#ff7f0e'])
axes[1,0].set_ylabel('Total Tokens/sec')
axes[1,0].set_title('Batch Throughput (Higher is Better)')

# GPU Utilization
axes[1,1].bar(['PyTorch', 'TensorRT'], [pt_single['gpu_util_pct'], trt_lat['gpu_util_pct']], color=['#1f77b4', '#ff7f0e'])
axes[1,1].set_ylabel('GPU Utilization (%)')
axes[1,1].set_title('GPU Utilization (Higher is Better)')
axes[1,1].set_ylim([0, 100])

plt.tight_layout()
plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to results/comparison.png")
