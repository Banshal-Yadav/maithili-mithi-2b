from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

model_path = "models/sarvam-1/v1/hf_merged_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

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

print(f"Running batch inference for 25 prompts...")

for p in tqdm(prompts):
    formatted_prompt = f"User: {p['text']}\nAssistant: "
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        temperature=0.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the assistant's part
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_part = full_output.split("Assistant: ")[-1].strip()
    
    results.append({
        "id": p["id"],
        "prompt": p["text"],
        "response": assistant_part
    })

    print(f"\nID {p['id']}: {p['text']}")
    print(f"Response: {assistant_part}")
    print("-" * 30)

output_file = "evals/results/batch_hf_test_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nBatch inference complete. Results saved to {output_file}")
