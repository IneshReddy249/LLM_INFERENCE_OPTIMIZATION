#!/usr/bin/env python3
import time, json, torch, numpy as np, subprocess
from pathlib import Path
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner

MODEL_DIR = "hf_models/qwen2.5-7b-instruct"
ENGINE_DIR = "engines/throughput"
BATCH_PROMPTS = [
    "Explain paged KV cache.",
    "What does remove_input_padding do in TensorRT-LLM?",
    "Give 3 ways to improve LLM throughput on A100.",
    "Difference between paged KV cache and normal KV cache?"
]
MAX_NEW_TOKENS, NUM_WARMUP, NUM_RUNS = 128, 3, 50

def get_gpu_util():
    try: 
        return float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", 
                                    "--format=csv,noheader,nounits"], 
                                   capture_output=True, text=True).stdout.strip())
    except: 
        return None

def main():
    print("=== TensorRT-LLM Throughput Engine Benchmark ===")
    
    # Load tokenizer and engine
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if not tokenizer.pad_token_id: 
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    runner = ModelRunner.from_dir(ENGINE_DIR)
    
    # Prepare batch inputs
    inputs = [torch.tensor(tokenizer.encode(p), dtype=torch.int32) for p in BATCH_PROMPTS]
    prompt_tokens = [len(tokenizer.encode(p)) for p in BATCH_PROMPTS]
    
    # Warmup
    print("Running warmup...")
    for _ in range(NUM_WARMUP): 
        runner.generate(inputs, max_new_tokens=10, 
                       end_id=tokenizer.eos_token_id, 
                       pad_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    
    # Measure batch TTFT
    print("Measuring batch TTFT...")
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e9
    
    t0 = time.time()
    runner.generate(inputs, max_new_tokens=1, 
                   end_id=tokenizer.eos_token_id, 
                   pad_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    ttft = time.time() - t0
    
    # Full batch generation - 50 runs
    print(f"Running {NUM_RUNS} batch iterations...")
    latencies = []
    for i in range(NUM_RUNS):
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{NUM_RUNS}")
        
        torch.cuda.synchronize()
        t0 = time.time()
        outs = runner.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, 
                             end_id=tokenizer.eos_token_id, 
                             pad_id=tokenizer.pad_token_id, 
                             return_dict=True, 
                             output_sequence_lengths=True)
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)
    
    # Memory stats
    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    
    # Calculate total generated tokens
    seq_lens = outs["sequence_lengths"]
    total_generated = 0
    for i in range(len(BATCH_PROMPTS)):
        if seq_lens.ndim == 2: 
            total_generated += int(seq_lens[i, 0])
        else: 
            total_generated += int(seq_lens[i])
    total_generated -= sum(prompt_tokens)
    
    # Calculate metrics
    latencies = np.array(latencies)
    gen_time = latencies.mean() - ttft
    
    results = {
        "engine": "tensorrt_throughput",
        "mode": "batch_processing",
        "batch_size": len(BATCH_PROMPTS),
        "prompts": BATCH_PROMPTS,
        "total_prompt_tokens": sum(prompt_tokens),
        "total_generated_tokens": total_generated,
        "ttft_ms": ttft * 1000,
        "latency_avg_s": float(latencies.mean()),
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
        "latency_p99_s": float(np.percentile(latencies, 99)),
        "total_tokens_per_sec": total_generated / gen_time if gen_time > 0 else 0,
        "memory_used_gb": mem_peak - mem_before,
        "memory_peak_gb": mem_peak,
        "gpu_util_pct": get_gpu_util()
    }
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    Path("results/tensorrt_throughput.json").write_text(json.dumps(results, indent=2))
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(json.dumps(results, indent=2))
    print("="*60)
    print("\n✓ Saved to results/tensorrt_throughput.json")

if __name__ == "__main__": 
    main()
