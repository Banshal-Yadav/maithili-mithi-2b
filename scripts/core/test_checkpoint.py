import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
from tqdm import tqdm
import os

from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
base_model_path = str(BASE_DIR / "base_models" / "llama-3.2-3b")
checkpoint_path = str(BASE_DIR / "models" / "llama-3.2-3b-maithili-peft" / "checkpoint-100")
output_file = str(BASE_DIR / "evals" / "results" / "llama_3.2_checkpoint_100.json")

# Load base model with 4-bit quantization
print(f"Loading base model from {base_model_path}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load PEFT adapter
print(f"Loading checkpoint from {checkpoint_path}...")
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()

prompts = [
    # EASY
    {"id": 1, "text": "[eng] The cow gives milk. [maithili]"},
    {"id": 2, "text": "[eng] I am hungry. [maithili]"},
    {"id": 3, "text": "[eng] It is raining outside. [maithili]"},
    {"id": 4, "text": "[eng] My father is a farmer. [maithili]"},
    {"id": 5, "text": "[eng] The child is sleeping. [maithili]"},
    # MEDIUM (Negation)
    {"id": 6, "text": "[eng] He did not come today. [maithili]"},
    {"id": 7, "text": "[eng] I will not go to the market tomorrow. [maithili]"},
    {"id": 8, "text": "[eng] She does not know the answer. [maithili]"},
    {"id": 9, "text": "[eng] Do not lie to your parents. [maithili]"},
    {"id": 10, "text": "[eng] The dog does not bite. [maithili]"},
    # MEDIUM (M -> E)
    {"id": 11, "text": "[maithili] हम काल्हि बजार जाएब। [eng]"},
    {"id": 12, "text": "[maithili] हमर माए खाना बनबैत छथि। [eng]"},
    {"id": 13, "text": "[maithili] बच्चा कानि रहल अछि। [eng]"},
    {"id": 14, "text": "[maithili] आइ बड्ड गर्मी अछि। [eng]"},
    {"id": 15, "text": "[maithili] ओ आइ नइ आएल। [eng]"},
    # HARD (Confusable)
    {"id": 16, "text": "[eng] The goat is eating grass. [maithili]"},
    {"id": 17, "text": "[eng] The tree has many mangoes. [maithili]"},
    {"id": 18, "text": "[eng] The buffalo is bigger than the cow. [maithili]"},
    {"id": 19, "text": "[eng] The cat sat on the roof, not the dog. [maithili]"},
    {"id": 20, "text": "[eng] Grandmother told a story, grandfather went to sleep. [maithili]"},
    # HARD (Cultural)
    {"id": 21, "text": "[eng] Makhana grows in the ponds of Mithila. [maithili]"},
    {"id": 22, "text": "[eng] Vidyapati wrote poems in Maithili. [maithili]"},
    {"id": 23, "text": "[eng] Chhath Puja is celebrated in Kartik month. [maithili]"},
    {"id": 24, "text": "[maithili] मखाना मिथिलाक पोखरिमे उगैत अछि। [eng]"},
    {"id": 25, "text": "[maithili] विद्यापति मैथिलीमे कविता लिखलाह। [eng]"}
]

results = []

print(f"Running inference for 25 prompts on checkpoint 100...")

for p in tqdm(prompts):
    # Use Llama Instruct template matching the training
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the assistant's part
    token_len = inputs.input_ids.shape[1]
    assistant_part = tokenizer.decode(outputs[0][token_len:], skip_special_tokens=True).strip()
    
    results.append({
        "id": p["id"],
        "prompt": p["text"],
        "response": assistant_part
    })

# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nEvaluation complete. Results saved to {output_file}")
