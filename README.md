# LLM Inference Optimization: PyTorch vs vLLM on NVIDIA A100

A comprehensive performance benchmarking project comparing standard PyTorch inference against the optimized vLLM framework, demonstrating massive gains in throughput and latency for Large Language Model (LLM) serving.

## 🚀 Key Results & Impact

-   **+56x Throughput:** Increased throughput from **22 to 1232 tokens/sec** for batched requests.
-   **-98% Latency:** Slashed latency from **5.1s to 0.1s** per request under load.
-   **Enhanced Scalability:** Achieved superior performance for high-load applications, directly improving user experience and reducing compute costs.
-   **Data-Driven Deployment:** Provided clear, quantifiable metrics to guide production deployment strategies.

## ⚙️ Technical Approach

This project involved designing a robust benchmarking pipeline to evaluate two distinct inference methodologies:

1.  **Baseline (PyTorch):** Utilized `Hugging Face Transformers` and standard `PyTorch` to establish a performance baseline for the Qwen-7B model.
2.  **Optimized (vLLM):** Implemented the `vLLM` inference engine, leveraging its state-of-the-art optimizations:
    -   **PagedAttention:** For efficient KV Cache memory management, eliminating fragmentation.
    -   **Continuous Batching:** To dynamically batch incoming requests, maximizing GPU utilization.
    -   **FlashAttention:** For a faster, more memory-efficient attention computation algorithm.

## 📊 Performance Metrics

| Framework | Scenario | Avg. Latency (s) | Throughput (Tokens/Sec) | GPU Memory (GB) |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | Single Prompt | ~5.77 | ~22.18 | ~0.05 |
| **PyTorch** | Batch (16 prompts) | ~5.04 | ~22.76 | ~0.05 |
| **vLLM** | Single Prompt | ~1.46 | ~87.54 | N/A |
| **vLLM** | Batch (16 prompts) | **~0.10** | **~1285.96** | N/A |

*Results from benchmarking Qwen-7B on an NVIDIA A100-SXM4-80GB GPU.*

## 🛠️ Tech Stack

**Languages & Frameworks:** Python, PyTorch, Hugging Face Transformers
**Optimization Engines:** vLLM
**Key Technologies:** FlashAttention, PagedAttention (KV-Cache), Continuous Batching
**Hardware:** NVIDIA A100 GPU, CUDA

## 📁 Repository Structure
