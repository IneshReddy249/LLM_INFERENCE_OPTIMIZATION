import json
from pathlib import Path

def load_json(path):
    return json.loads(Path(path).read_text())

def main():
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: PyTorch vs TensorRT-LLM")
    print("="*80)
    
    # Load all results
    pt_single = load_json("results/pytorch_single.json")
    pt_batch = load_json("results/pytorch_batch.json")
    trt_lat = load_json("results/tensorrt_latency.json")
    trt_thr = load_json("results/tensorrt_throughput.json")
    
    # Single request comparison
    print("\n### SINGLE REQUEST (Latency-Optimized) ###\n")
    print(f"{'Metric':<25} {'PyTorch':>15} {'TensorRT':>15} {'Speedup':>12}")
    print("-" * 70)
    
    metrics_single = [
        ("TTFT (ms)", "ttft_ms", False),
        ("Latency P50 (s)", "latency_p50_s", False),
        ("Latency P95 (s)", "latency_p95_s", False),
        ("Latency P99 (s)", "latency_p99_s", False),
        ("Tokens/sec", "tokens_per_sec", True),
        ("Memory Used (GB)", "memory_used_gb", False),
        ("GPU Utilization (%)", "gpu_util_pct", True),
    ]
    
    for name, key, higher_better in metrics_single:
        pt_val = pt_single.get(key, 0)
        trt_val = trt_lat.get(key, 0)
        
        if pt_val > 0:
            speedup = trt_val / pt_val if higher_better else pt_val / trt_val
            arrow = "↑" if higher_better else "↓"
            print(f"{name:<25} {pt_val:>15.2f} {trt_val:>15.2f} {speedup:>10.2f}× {arrow}")
    
    # Batch comparison
    print("\n### BATCH PROCESSING (Throughput-Optimized) ###\n")
    print(f"{'Metric':<25} {'PyTorch':>15} {'TensorRT':>15} {'Speedup':>12}")
    print("-" * 70)
    
    # Use correct key names for batch
    pt_batch_tps = pt_batch.get("total_tokens_per_sec", pt_batch.get("tokens_per_sec", 0))
    trt_thr_tps = trt_thr.get("total_tokens_per_sec", trt_thr.get("tokens_per_sec", 0))
    
    metrics_batch = [
        ("Batch TTFT (ms)", "ttft_ms", False),
        ("Latency P50 (s)", "latency_p50_s", False),
        ("Latency P95 (s)", "latency_p95_s", False),
        ("Memory Used (GB)", "memory_used_gb", False),
        ("GPU Utilization (%)", "gpu_util_pct", True),
    ]
    
    for name, key, higher_better in metrics_batch:
        pt_val = pt_batch.get(key, 0)
        trt_val = trt_thr.get(key, 0)
        
        if pt_val > 0:
            speedup = trt_val / pt_val if higher_better else pt_val / trt_val
            arrow = "↑" if higher_better else "↓"
            print(f"{name:<25} {pt_val:>15.2f} {trt_val:>15.2f} {speedup:>10.2f}× {arrow}")
    
    # Add throughput manually
    if pt_batch_tps > 0 and trt_thr_tps > 0:
        speedup = trt_thr_tps / pt_batch_tps
        print(f"{'Total Throughput (tok/s)':<25} {pt_batch_tps:>15.2f} {trt_thr_tps:>15.2f} {speedup:>10.2f}× ↑")
    
    # Summary
    single_speedup = trt_lat["tokens_per_sec"] / pt_single["tokens_per_sec"]
    batch_speedup = trt_thr_tps / pt_batch_tps if pt_batch_tps > 0 else 0
    avg_speedup = (single_speedup + batch_speedup) / 2
    
    print("\n" + "="*80)
    print("📊 SUMMARY:")
    print(f"   Single Request Speedup:  {single_speedup:.2f}×")
    print(f"   Batch Throughput Speedup: {batch_speedup:.2f}×")
    print(f"   Average Speedup:          {avg_speedup:.2f}×")
    print(f"   GPU Utilization Improvement: {pt_single['gpu_util_pct']:.0f}% → {trt_lat['gpu_util_pct']:.0f}%")
    print(f"   Memory Efficiency: {pt_single['memory_used_gb']:.1f}GB → {trt_lat['memory_used_gb']:.2f}GB")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
