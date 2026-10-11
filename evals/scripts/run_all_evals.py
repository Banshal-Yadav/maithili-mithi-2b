import os
import subprocess

MODELS = [
    ("v3", r"v3\maithili-lora-v3"),
    ("v3.1", r"v3.1\maithili-stage3.1-output\checkpoint-543"),
    ("v3.2", r"v3.2\maithili-lora-v3.2-ep1")
]

EVAL_SCRIPT = r"scripts\eval\run_eval.py"
DATASET = "maithili_eval_v2.json"
RESULTS_DIR = "eval_results"

# Ensure PYTORCH_CUDA_ALLOC_CONF is set for the subprocesses
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for model_key, model_path in MODELS:
    output_path = os.path.join(RESULTS_DIR, f"eval_results_{model_key}.json")
    
    # Skip if already exists to save time in case of restart
    if os.path.exists(output_path):
        print(f"Skipping {model_key}, {output_path} already exists.")
        continue
        
    print(f"\n=============================================")
    print(f"Starting Evaluation for {model_key.upper()}")
    print(f"Path: {model_path}")
    print(f"=============================================")
    
    cmd = ["python", EVAL_SCRIPT, "--model", model_path, "--output", output_path, "--dataset", DATASET]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {model_key}: {e}")
        break

print("\nAll sequential evaluations completed.")
