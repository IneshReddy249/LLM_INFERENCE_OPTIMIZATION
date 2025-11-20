#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "hf_models/qwen2.5-7b-instruct"

def main():
    print(f"Downloading {MODEL_NAME}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✓ Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
