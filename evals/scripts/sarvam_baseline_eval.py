from unsloth import FastLanguageModel
import torch
import json
import os
from tqdm import tqdm

from pathlib import Path

# 1. Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
model_name = str(BASE_DIR / "models" / "sarvam-1" / "base")
output_file = str(BASE_DIR / "evals" / "results" / "sarvam_base_baseline.json")
max_seq_length = 2048

# 2. Load Model & Tokenizer
print(f"Loading base model from {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 3. Prompts
prompts = [
    {"id": 1, "text": "[eng] The cow gives milk. [maithili]"},
    {"id": 2, "text": "[eng] I am hungry. [maithili]"},
    {"id": 3, "text": "[eng] It is raining outside. [maithili]"},
    {"id": 4, "text": "[eng] My father is a farmer. [maithili]"},
    {"id": 5, "text": "[eng] The child is sleeping. [maithili]"},
    {"id": 6, "text": "[eng] He did not come today. [maithili]"},
    {"id": 7, "text": "[eng] I will not go to the market tomorrow. [maithili]"},
    {"id": 8, "text": "[eng] She does not know the answer. [maithili]"},
    {"id": 9, "text": "[eng] Do not lie to your parents. [maithili]"},
    {"id": 10, "text": "[eng] The dog does not bite. [maithili]"},
    {"id": 11, "text": "[maithili] हम काल्हि बजार जाएब। [eng]"},
    {"id": 12, "text": "[maithili] हमर माए खाना बनबैत छथि। [eng]"},
    {"id": 13, "text": "[maithili] बच्चा कानि रहल अछि। [eng]"},
    {"id": 14, "text": "[maithili] आइ बड्ड गर्मी अछि। [eng]"},
    {"id": 15, "text": "[maithili] ओ आइ नइ आएल। [eng]"},
    {"id": 16, "text": "[eng] The goat is eating grass. [maithili]"},
    {"id": 17, "text": "[eng] The tree has many mangoes. [maithili]"},
    {"id": 18, "text": "[eng] The buffalo is bigger than the cow. [maithili]"},
    {"id": 19, "text": "[eng] The cat sat on the roof, not the dog. [maithili]"},
    {"id": 20, "text": "[eng] Grandmother told a story, grandfather went to sleep. [maithili]"},
    {"id": 21, "text": "[eng] Makhana grows in the ponds of Mithila. [maithili]"},
    {"id": 22, "text": "[eng] Vidyapati wrote poems in Maithili. [maithili]"},
    {"id": 23, "text": "[eng] Chhath Puja is celebrated in Kartik month. [maithili]"},
    {"id": 24, "text": "[maithili] मखाना मिथिलाक पोखरिमे उगैत अछि। [eng]"},
    {"id": 25, "text": "[maithili] विद्यापति मैथिलीमे कविता लिखलाह। [eng]"}
]

results = []
print("Running baseline evaluation...")

for p in tqdm(prompts):
    inputs = tokenizer([p["text"]], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    response = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens = True)[0]
    
    results.append({
        "id": p["id"],
        "prompt": p["text"],
        "response": response.strip()
    })

# 4. Save
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Baseline results saved to {output_file}")
