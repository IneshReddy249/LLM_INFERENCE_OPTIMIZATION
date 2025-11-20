# LLM Inference Optimization with TensorRT-LLM

Production-grade inference optimization achieving **2.56× speedup** on NVIDIA A100 using TensorRT-LLM.

## Overview

This project optimizes Qwen2.5-7B inference performance by converting PyTorch models to TensorRT-LLM engines with FP16 precision, FlashAttention, and paged KV-cache.

## Performance Results

| Metric | PyTorch Baseline | TensorRT-LLM | Speedup |
|--------|-----------------|--------------|---------|
| **Single Request Latency (P50)** | 3.36s | 1.32s | **2.55×** |
| **Batch Throughput** | 146 tok/s | 373 tok/s | **2.56×** |
| **GPU Utilization** | 56% | 95% | **+39pp** |
| **Memory Usage** | 15.3 GB | 0.05 GB | **99.7% reduction** |

## Key Optimizations

- **FP16 Precision**: 2× faster math on Tensor Cores
- **Paged KV-Cache**: Dynamic memory allocation, 99.7% memory reduction
- **FlashAttention**: O(N) memory complexity for attention
- **Kernel Fusion**: Reduced kernel launch overhead
- **Engine Profiling**: Hardware-specific CUDA kernel optimization

## Project Structure
```
├── scripts/
│   ├── 01_download_model.py              # Download Qwen2.5-7B from HuggingFace
│   ├── 02_benchmark_pytorch.py           # PyTorch baseline benchmarks
│   ├── 03_convert_checkpoint.sh          # HF → TensorRT checkpoint conversion
│   ├── 04_build_latency_engine.sh        # Build latency-optimized engine (batch_size=1)
│   ├── 05_build_throughput_engine.sh     # Build throughput-optimized engine (batch_size=8)
│   ├── 06_benchmark_tensorrt_latency.py  # Benchmark latency engine
│   ├── 07_benchmark_tensorrt_throughput.py # Benchmark throughput engine
│   └── 08_compare_results.py             # Generate comparison report
├── results/
│   ├── pytorch_single.json               # PyTorch single request metrics
│   ├── pytorch_batch.json                # PyTorch batch metrics
│   ├── tensorrt_latency.json             # TensorRT latency metrics
│   └── tensorrt_throughput.json          # TensorRT throughput metrics
└── README.md
```

## Requirements

- NVIDIA GPU (A100 80GB used in this project)
- CUDA 12.x
- Python 3.10
- TensorRT-LLM 0.15.0

## Installation
```bash
# Clone repository
git clone https://github.com/IneshReddy249/LLM_INFERENCE_OPTIMIZATION.git
cd LLM_INFERENCE_OPTIMIZATION

# Install dependencies
pip install tensorrt-llm==0.15.0 --pre --extra-index-url https://pypi.nvidia.com
pip install transformers torch accelerate numpy pandas matplotlib

# Install MPI (required for TensorRT-LLM)
sudo apt-get update && sudo apt-get install -y openmpi-bin libopenmpi-dev

# Clone TensorRT-LLM for conversion scripts
cd ~
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM && git checkout v0.15.0
```

## Quick Start

### 1. Download Model
```bash
python3 scripts/01_download_model.py
```

### 2. PyTorch Baseline Benchmark
```bash
python3 scripts/02_benchmark_pytorch.py
# Takes ~20 minutes, runs 50 iterations for statistical rigor
```

### 3. Convert to TensorRT Checkpoint
```bash
./scripts/03_convert_checkpoint.sh
# Converts HF weights to TensorRT format (FP16, TP=1)
```

### 4. Build TensorRT Engines
```bash
# Latency-optimized engine (batch_size=1)
./scripts/04_build_latency_engine.sh

# Throughput-optimized engine (batch_size=8)
./scripts/05_build_throughput_engine.sh
# Each takes ~30 minutes - TensorRT profiles hundreds of kernel implementations
```

### 5. Benchmark TensorRT Engines
```bash
python3 scripts/06_benchmark_tensorrt_latency.py
python3 scripts/07_benchmark_tensorrt_throughput.py
# Each takes ~15 minutes, 50 iterations
```

### 6. Generate Comparison Report
```bash
python3 scripts/08_compare_results.py
```

## Benchmark Methodology

All benchmarks measure:
- **TTFT (Time to First Token)**: Latency until first token generation
- **P50/P95/P99 Latency**: Statistical distribution across 50 runs
- **Throughput**: Tokens generated per second
- **Memory Usage**: Peak GPU memory consumption
- **GPU Utilization**: Percentage of GPU compute used

## Technical Details

### Conversion Process
```bash
python3 ~/TensorRT-LLM/examples/qwen/convert_checkpoint.py \
  --model_dir hf_models/qwen2.5-7b-instruct \
  --output_dir checkpoints/qwen2.5-7b \
  --dtype float16 \
  --tp_size 1 \
  --pp_size 1
```

### Engine Build Configuration
```bash
trtllm-build \
  --checkpoint_dir checkpoints/qwen2.5-7b \
  --output_dir engines/latency \
  --max_batch_size 1 \
  --max_input_len 512 \
  --max_seq_len 640 \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --context_fmha enable \
  --paged_kv_cache enable
```

## Results Analysis

### Single Request Performance
- **Latency reduction**: 3.36s → 1.32s (2.55× faster)
- **TTFT improvement**: 26ms → 13ms (2× faster)
- **Throughput increase**: 38 tok/s → 98 tok/s (2.56× faster)

### Batch Processing Performance
- **Throughput increase**: 146 tok/s → 373 tok/s (2.56× faster)
- **Latency (P50)**: 3.54s → 1.38s (2.56× faster)
- **Memory efficiency**: 15.3GB → 0.05GB (99.7% reduction)

### GPU Utilization
- **PyTorch**: 56% (44% of GPU idle)
- **TensorRT**: 95% (near-optimal utilization)

## Why This Speedup?

1. **FP16 Tensor Cores**: 2× faster than FP32 on A100
2. **Kernel Fusion**: Combines MatMul+Bias+Activation into single kernel
3. **Paged KV-Cache**: Eliminates memory waste from pre-allocated cache
4. **FlashAttention**: Reduces memory bandwidth bottleneck
5. **Hardware-Specific Compilation**: TensorRT profiles and selects optimal kernels for A100

## Hardware Used

- **GPU**: NVIDIA A100 80GB
- **CPU**: 64 cores
- **RAM**: 120GB
- **CUDA**: 12.x


## Acknowledgments

- NVIDIA TensorRT-LLM  for the optimization framework
- Qwen team for the base model
- A100 GPU provided by Shadeform
