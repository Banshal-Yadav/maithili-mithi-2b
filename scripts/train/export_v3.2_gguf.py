import os
import sys
import subprocess
import torch
from unsloth import FastLanguageModel

# CRITICAL WINDOWS FIXES: Disable all Triton/Compile logic and apt-get/uv routines
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["UNSLOTH_USE_UV"] = "0"

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
LLAMA_CPP_SCRIPT = os.path.join(BASE_DIR, "llama.cpp", "convert_hf_to_gguf.py")

MODELS_TO_EXPORT = [
    (os.path.join(BASE_DIR, "v3.2", "maithili-lora-v3.2-ep1"), os.path.join(BASE_DIR, "v3.2", "GGUF_ep1")),
    (os.path.join(BASE_DIR, "v3.2", "maithili-lora-v3.2"), os.path.join(BASE_DIR, "v3.2", "GGUF_final"))
]

for lora_path, target_dir in MODELS_TO_EXPORT:
    print(f"\n=======================================================")
    print(f"Starting Pure-Python Export for: {lora_path}")
    print(f"=======================================================")
    
    if not os.path.exists(lora_path):
        print(f"WARNING: LoRA path does not exist: {lora_path}. Skipping.")
        continue

    hf_merged_dir = os.path.join(target_dir, "hf_merged_16bit")
    os.makedirs(hf_merged_dir, exist_ok=True)

    print(f"Loading LoRA from {lora_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = False,
    )

    print("Saving HF Merged 16-bit to temporary folder...")
    model.save_pretrained_merged(hf_merged_dir, tokenizer, save_method="merged_16bit")
    
    # Free VRAM explicitly
    del model
    del tokenizer
    torch.cuda.empty_cache()

    name = "maithili-v3.2-ep1" if "ep1" in lora_path else "maithili-v3.2-final"

    print("\n1. Exporting F16 GGUF (Python-only)...")
    f16_gguf = os.path.join(target_dir, f"{name}-f16.gguf")
    subprocess.run([sys.executable, LLAMA_CPP_SCRIPT, hf_merged_dir, "--outfile", f16_gguf, "--outtype", "f16"])

    print("\n2. Exporting Q8_0 GGUF (Python-only)...")
    q8_gguf = os.path.join(target_dir, f"{name}-q8_0.gguf")
    subprocess.run([sys.executable, LLAMA_CPP_SCRIPT, hf_merged_dir, "--outfile", q8_gguf, "--outtype", "q8_0"])

    print("\n[NOTE] Q4_K_M requires local compiled `llama-quantize` binares (C++). Skipped for 'python-only' pipeline.")

print("\nAll Pure-Python Stage-3.2 GGUF exports completed successfully!")
