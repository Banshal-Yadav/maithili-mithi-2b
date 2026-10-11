import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import json
import torch
import argparse
from unsloth import FastLanguageModel

# Constants
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_FILE = os.path.join(BASE_DIR, "evals", "benchmark_legacy", "academic_eval", "maithili_eval_v1.json")
MAX_SEQ_LENGTH = 1024

def format_prompt(instruction):
    # Gemma 2 format
    return f"<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n"

def run_evaluation(model_path, output_file, dataset_path):
    print(f"Loading Model from {model_path} ...")
    # Load model and tokenizer
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = True, # Use 4-bit config for faster inference and lower VRAM usage
        )
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return

    # Enable native inference optimizations
    print("Enabling native inference optimizations...")
    FastLanguageModel.for_inference(model)

    print(f"Loading evaluation dataset from {dataset_path} ...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    results = []
    
    print(f"Starting evaluation on {len(eval_data)} prompts...\n")
    for i, item in enumerate(eval_data):
        prompt_text = item["prompt"]
        formatted_prompt = format_prompt(prompt_text)
        
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

        # Generate response
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 512, 
            use_cache = True,
            temperature = 0.3, # low temp for deterministic benchmarking
            top_p = 0.9,
            repetition_penalty = 1.15
        )
        
        # Decode only the generated text (ignoring the prompt)
        output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        print(f"[{i+1}/{len(eval_data)}] {item['name']}")
        print(f"Prompt: {prompt_text}")
        print(f"Response: {output_text}\n")
        
        item["response"] = output_text
        results.append(item)
    
    print(f"Saving results to {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation complete. Results saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Maithili model checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model or LoRA checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the JSON results.")
    parser.add_argument("--dataset", type=str, default="maithili_eval_v1.json", help="Path to the evaluation benchmark dataset JSON file.")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.output, args.dataset)
