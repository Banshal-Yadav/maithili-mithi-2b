import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Define the models to evaluate
MODELS = {
    "v2": r"v2\maithili-lora-v2",
    "v3": r"v3\maithili-lora-v3",
    "v3.1": r"v3.1\maithili-stage3.1-output\checkpoint-543",
    "v3.2": r"v3.2\maithili-lora-v3.2-ep1"
}

DATASET_PATH = "maithili_realworld_eval_v3.json"
RESULTS_DIR = "eval_results"

def generate_evaluations():
    print(f"\nLoading real-world benchmark dataset from {DATASET_PATH} ...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    for model_name, model_path in MODELS.items():
        output_file = os.path.join(RESULTS_DIR, f"realworld_results_{model_name}.json")
        
        if os.path.exists(output_file):
            print(f"Skipping {model_name}, {output_file} already exists.")
            continue
            
        print(f"\n=============================================")
        print(f"Starting Evaluation for {model_name.upper()}")
        print(f"Path: {model_path}")
        print(f"=============================================")
        
        # Free memory before loading
        torch.cuda.empty_cache()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model.eval()
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

        results = []
        print(f"Generating responses for {len(eval_data)} prompts...")
        
        for i, item in enumerate(tqdm(eval_data, desc=model_name)):
            prompt_text = item["prompt"]
            messages = [{"role": "user", "content": prompt_text}]
            
            # Using standard model-agnostic chat templates to support BOTH Llama-3 (V2) and Gemma-2 (V3) effortlessly
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = 512,
                    use_cache = True,
                    temperature = 0.3,
                    top_p = 0.9,
                    repetition_penalty = 1.15
                )
            
            # Decode only the generated text (ignoring the prompt)
            response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            # Save to dict
            out_item = item.copy()
            out_item["response"] = response_text
            results.append(out_item)
            
        print(f"Saving results to {output_file} ...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
    print("\nAll sequential real-world evaluations completed.")

if __name__ == "__main__":
    generate_evaluations()
