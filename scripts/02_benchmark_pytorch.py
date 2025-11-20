#!/usr/bin/env python3
import time, json, torch, numpy as np, subprocess
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "hf_models/qwen2.5-7b-instruct"
SINGLE_PROMPT = "Explain paged KV cache in transformers in simple English."
BATCH_PROMPTS = ["Explain paged KV cache.", "What does remove_input_padding do in TensorRT-LLM?", 
                 "Give 3 ways to improve LLM throughput on A100.", "Difference between paged KV cache and normal KV cache?"]
MAX_NEW_TOKENS, NUM_WARMUP, NUM_RUNS = 128, 3, 50

def get_gpu_util():
    try: return float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 
                                     capture_output=True, text=True).stdout.strip())
    except: return None

def benchmark(model, tokenizer, prompts, is_batch):
    print(f"\n=== PyTorch {'Batch' if is_batch else 'Single'} Benchmark ===")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    prompt_tokens = sum(len(tokenizer.encode(p)) for p in (prompts if is_batch else [prompts]))
    
    # Warmup
    for _ in range(NUM_WARMUP): 
        with torch.no_grad(): model.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize()
    
    # TTFT
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e9
    t0 = time.time()
    with torch.no_grad(): model.generate(**inputs, max_new_tokens=1)
    torch.cuda.synchronize()
    ttft = time.time() - t0
    
    # Full generation
    latencies = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(): outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)
    
    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    latencies = np.array(latencies)
    
    # Calculate tokens
    total_generated = sum(len(tokenizer.encode(tokenizer.decode(o, skip_special_tokens=True))) for o in outputs) - prompt_tokens
    gen_time = latencies.mean() - ttft
    
    results = {
        "mode": "batch" if is_batch else "single",
        "batch_size": len(prompts) if is_batch else 1,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": total_generated,
        "ttft_ms": ttft * 1000,
        "latency_avg_s": latencies.mean(),
        "latency_p50_s": np.percentile(latencies, 50),
        "latency_p95_s": np.percentile(latencies, 95),
        "latency_p99_s": np.percentile(latencies, 99),
        "tokens_per_sec": total_generated / gen_time if gen_time > 0 else 0,
        "memory_used_gb": mem_peak - mem_before,
        "memory_peak_gb": mem_peak,
        "gpu_util_pct": get_gpu_util()
    }
    print(json.dumps(results, indent=2))
    return results

def main():
    print("Loading PyTorch model (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if not tokenizer.pad_token_id: tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    
    single = benchmark(model, tokenizer, SINGLE_PROMPT, False)
    batch = benchmark(model, tokenizer, BATCH_PROMPTS, True)
    
    Path("results").mkdir(exist_ok=True)
    Path("results/pytorch_single.json").write_text(json.dumps(single, indent=2))
    Path("results/pytorch_batch.json").write_text(json.dumps(batch, indent=2))
    print("\n✓ Results saved to results/pytorch_*.json")

if __name__ == "__main__": main()
