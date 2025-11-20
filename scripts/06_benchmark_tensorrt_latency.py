#!/usr/bin/env python3
import time, json, torch, numpy as np, subprocess
from pathlib import Path
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner

MODEL_DIR = "hf_models/qwen2.5-7b-instruct"
ENGINE_DIR = "engines/latency"
PROMPT = "Explain paged KV cache in transformers in simple English."
MAX_NEW_TOKENS, NUM_WARMUP, NUM_RUNS = 128, 3, 50

def get_gpu_util():
    try: 
        return float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", 
                                    "--format=csv,noheader,nounits"], 
                                   capture_output=True, text=True).stdout.strip())
    except: 
        return None

def main():
    print("=== TensorRT-LLM Latency Engine Benchmark ===")
    
    # Load tokenizer and engine
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if not tokenizer.pad_token_id: 
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    runner = ModelRunner.from_dir(ENGINE_DIR)
    
    # Prepare input
    prompt_ids = tokenizer.encode(PROMPT)
    inputs = [torch.tensor(prompt_ids, dtype=torch.int32)]
    
    # Warmup
    print("Running warmup...")
    for _ in range(NUM_WARMUP): 
        runner.generate(inputs, max_new_tokens=10, 
                       end_id=tokenizer.eos_token_id, 
                       pad_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    
    # Measure TTFT
    print("Measuring TTFT...")
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e9
    
    t0 = time.time()
    runner.generate(inputs, max_new_tokens=1, 
                   end_id=tokenizer.eos_token_id, 
                   pad_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    ttft = time.time() - t0
    
    # Full generation - 50 runs
    print(f"Running {NUM_RUNS} iterations...")
    latencies = []
    for i in range(NUM_RUNS):
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{NUM_RUNS}")
        
        torch.cuda.synchronize()
        t0 = time.time()
        out = runner.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, 
                            end_id=tokenizer.eos_token_id, 
                            pad_id=tokenizer.pad_token_id, 
                            return_dict=True, 
                            output_sequence_lengths=True)
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)
    
    # Memory stats
    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    
    # Extract output tokens
    output_ids = out["output_ids"]
    seq_len = out["sequence_lengths"]
    
    if output_ids.ndim == 3: 
        output_ids = output_ids[0, 0, :]
    else: 
        output_ids = output_ids[0, :]
    
    if seq_len.ndim == 2: 
        seq_len = int(seq_len[0, 0])
    else: 
        seq_len = int(seq_len[0])
    
    # Calculate metrics
    latencies = np.array(latencies)
    generated_tokens = seq_len - len(prompt_ids)
    gen_time = latencies.mean() - ttft
    
    results = {
        "engine": "tensorrt_latency",
        "mode": "single_request",
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": generated_tokens,
        "total_tokens": seq_len,
        "ttft_ms": ttft * 1000,
        "latency_avg_s": float(latencies.mean()),
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
        "latency_p99_s": float(np.percentile(latencies, 99)),
        "tokens_per_sec": generated_tokens / gen_time if gen_time > 0 else 0,
        "memory_used_gb": mem_peak - mem_before,
        "memory_peak_gb": mem_peak,
        "gpu_util_pct": get_gpu_util()
    }
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    Path("results/tensorrt_latency.json").write_text(json.dumps(results, indent=2))
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(json.dumps(results, indent=2))
    print("="*60)
    print("\n✓ Saved to results/tensorrt_latency.json")

if __name__ == "__main__": 
    main()
