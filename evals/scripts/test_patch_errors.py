import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Define the models to compare
MODELS = {
    "V3.1 (Baseline)": r"v3.1\maithili-stage3.1-output\checkpoint-543",
    "V3.2 (Patched)": r"v3.2\maithili-lora-v3.2-ep1"
}

# The specific failure cases we patched
PROMPTS = [
    {
        "name": "Action Vocab Error (Drive Storage)",
        "instruction": "System Alert: Your Google Drive storage is 95% full. Because you are out of storage, you cannot upload new files to Drive, back up photos or videos to Google Photos. You have run out of your 15 GB initial free storage. Delete old files to recover your system."
    },
    {
        "name": "Mixed English Instruction",
        "instruction": "Here is an English sentence: 'The database migration completed successfully.' Translate it perfectly into Maithili."
    },
    {
        "name": "System Alert (Battery)",
        "instruction": "Your device battery is severely degraded and running critically low at 5%. Please plug in your charger immediately to prevent unsaved data loss and system shutdown."
    },
    {
        "name": "System Alert (OTP)",
        "instruction": "Your OTP has expired."
    },
    {
        "name": "Mixed Instruction (Notification)",
        "instruction": "सिर्फ की करबाक चाही बताउ — How to turn off notifications?"
    }
]

max_seq_length = 2048

def format_prompt(instruction):
    return f"<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n"

def generate_comparisons():
    print("\n=============================================")
    print("Testing Stage-3.2 Behavior Patch Specifics")
    print("Comparing V3.1 (Baseline) vs V3.2 (Patched)")
    print("=============================================\n")
    
    results = {p["name"]: {} for p in PROMPTS}
    
    for model_name, model_path in MODELS.items():
        print(f"\nLoading {model_name} from: {model_path} ...")
        
        # Free CUDA memory before loading new model
        torch.cuda.empty_cache()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Remove FastLanguageModel.for_inference(model) which tries to compile Triton
            # Just put it in eval mode standardly
            model.eval()
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

        for i, p in enumerate(PROMPTS):
            formatted_prompt = format_prompt(p["instruction"])
            inputs = tokenizer(
            [
                formatted_prompt
            ], return_tensors = "pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = 256,
                    use_cache = False,
                    temperature = 0.3, # match evaluation temperature
                    top_p = 0.9,
                    repetition_penalty = 1.15
                )
            
            # Decode only the generated text (ignoring the prompt)
            response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            results[p["name"]][model_name] = response_text
            print(f"  -> Generated response for: {p['name']}")
    
    print("\n\n=============================================")
    print("               FINAL COMPARISON               ")
    print("=============================================\n")
    
    with open(r"eval_results\patch_comparison_report.txt", "w", encoding="utf-8") as f:
        for p in PROMPTS:
            name = p["name"]
            f.write(f"[{name}]\n")
            f.write(f"PROMPT: {p['instruction']}\n\n")
            
            f.write(f"V3.1 (Baseline) Output:\n")
            f.write(f"{results[name].get('V3.1 (Baseline)', 'ERROR')}\n\n")
            
            f.write(f"V3.2 (Patched) Output:\n")
            f.write(f"{results[name].get('V3.2 (Patched)', 'ERROR')}\n\n")
            f.write("-" * 50 + "\n\n")
            
            # Print to console too
            print(f"[{name}]")
            print(f"PROMPT: {p['instruction']}")
            print(f"\n❌ V3.1 (Baseline): \n{results[name].get('V3.1 (Baseline)', 'ERROR')}")
            print(f"\n✅ V3.2 (Patched): \n{results[name].get('V3.2 (Patched)', 'ERROR')}")
            print("\n--------------------------------------------------\n")
            
    print(r"Comparison saved to eval_results\patch_comparison_report.txt")

if __name__ == "__main__":
    generate_comparisons()
