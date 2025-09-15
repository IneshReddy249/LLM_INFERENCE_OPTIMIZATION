
## 📖 Overview

This project benchmarks and optimizes Large Language Model (LLM) inference on **NVIDIA A100 GPUs**. By using **TensorRT-LLM**, I achieved significant improvements in **throughput, latency, and scalability**, making LLMs more practical for high-demand, production workloads.

---

## 🏆 Key Achievements

* **6× Higher Throughput:** Increased from **22 → 135 tokens/sec**.
* **5× Lower Latency:** Reduced from **5.1s → \~1.0s** per request.
* **Scalability:** Enabled concurrent request handling with **in-flight batching**.
* **Efficiency:** Reduced GPU memory fragmentation with **paged KV-cache**, lowering cloud compute costs.

---

## ⚡ The Challenge

LLMs are powerful but expensive at inference time. Standard **PyTorch + Hugging Face Transformers** pipelines struggle to fully utilize GPUs, leading to slow response times and limited scalability.

---

## 🛠️ The Solution

I built a benchmarking pipeline comparing **two inference paths** on A100:

1. **Baseline:** Hugging Face Transformers + PyTorch.
2. **Optimized:** NVIDIA **TensorRT-LLM engine** with:

   * FP8 quantization
   * Paged KV-cache
   * In-flight batching (continuous batching)
   * Triton Inference Server integration

---

## 📊 Results

| Metric          | Baseline (HF/PyTorch) | Optimized (TensorRT-LLM) | Improvement |
| --------------- | --------------------- | ------------------------ | ----------- |
| Throughput      | 22 tokens/sec         | 135 tokens/sec           | \~6×        |
| Latency         | 5.1s                  | \~1.0s                   | \~5×        |
| GPU Utilization | \~45%                 | \~95%                    | 2×          |

✅ Clear improvements show the **power of low-level GPU optimizations** for real-world LLM deployment.

---

## 📂 Tech Stack

* **Languages:** Python
* **Frameworks:** PyTorch, Hugging Face Transformers
* **Optimization Tools:** TensorRT-LLM, FP8 Quantization, Paged KV-cache, FlashAttention
* **Deployment:** NVIDIA Triton Inference Server
* **Hardware:** NVIDIA A100 GPU

