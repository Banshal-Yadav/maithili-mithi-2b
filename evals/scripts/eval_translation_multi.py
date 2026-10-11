import ollama
import json
import os
from tqdm import tqdm

def run_translation_eval(model_name, output_file):
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
    print(f"Evaluating translations for model: {model_name}")
    for p in tqdm(prompts):
        try:
            response = ollama.generate(model=model_name, prompt=p["text"])
            results.append({
                "id": p["id"],
                "prompt": p["text"],
                "response": response['response'].strip()
            })
        except Exception as e:
            print(f"Error evaluating {model_name} on prompt {p['id']}: {e}")
            results.append({"id": p["id"], "prompt": p["text"], "response": "ERROR"})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    models = ["maithili-v4.0", "maithili-v3.2.2", "maithili-v3.2"]
    for m in models:
        run_translation_eval(m, f"eval_translation_{m}_results.json")
