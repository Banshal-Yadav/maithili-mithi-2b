import json
import ollama
from tqdm import tqdm

PROMPTS = [
    {"id": 1, "category": "QA", "prompt": "मिथिला क्षेत्र कतऽ अछि?"},
    {"id": 2, "category": "QA", "prompt": "विद्यापति के छलथि?"},
    {"id": 3, "category": "QA", "prompt": "छठ पूजा कखन होइत अछि?"},
    {"id": 4, "category": "QA", "prompt": "मखाना कतय होइत अछि?"},
    {"id": 5, "category": "QA", "prompt": "Tell me about Chhath Puja."},
    {"id": 6,  "category": "Translation", "prompt": "[eng] I will never forget this day. [maithili]"},
    {"id": 7,  "category": "Translation", "prompt": "[eng] The cow gives milk every morning. [maithili]"},
    {"id": 8,  "category": "Translation", "prompt": "[eng] Never lie to your parents. [maithili]"},
    {"id": 9,  "category": "Translation", "prompt": "[maithili] हम काल्हि बजार जाएब। [eng]"},
    {"id": 10, "category": "Translation", "prompt": "[maithili] हमर बाबूजी किसान छथि। [eng]"},
    {"id": 11, "category": "Instruction", "prompt": "एहि वाक्यकेँ सरल बनाउ: वर्षाऋतुमे नदीमे जलस्तर अत्यधिक वृद्धि पाबैत अछि।"},
    {"id": 12, "category": "Instruction", "prompt": "Correct this: मैं कल बाजार जाऊंगा।"},
    {"id": 13, "category": "Instruction", "prompt": "एहि notification कें मैथिलीमे बुझाउ: Your order has been shipped and will arrive in 3 days."},
    {"id": 14, "category": "Instruction", "prompt": "१ लाइनमे बताउ — What is democracy?"},
    {"id": 15, "category": "Instruction", "prompt": "Correct this: वह आज नहीं आएगा।"},
    {"id": 16, "category": "Conversational", "prompt": "हमरा बुखार अछि, की करी?"},
    {"id": 17, "category": "Conversational", "prompt": "हमर बच्चा राति भरि कानैत अछि, की करी?"},
    {"id": 18, "category": "Conversational", "prompt": "गाममे रहब नीक की शहरमे?"},
    {"id": 19, "category": "Conversational", "prompt": "हम बहुत उदास छी, की करी?"},
    {"id": 20, "category": "Conversational", "prompt": "मोबाइल चार्ज नइ होइत अछि, की करी?"},
]

def run_eval(model_name, output_file):
    print(f"\nEvaluating model: {model_name}")
    results = []
    
    for p in tqdm(PROMPTS):
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': p['prompt']},
        ])
        
        results.append({
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "response": response['message']['content'],
            "model": model_name
        })
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Evaluate v4.0
    run_eval("maithili-v4.0", "eval_v4_0_results.json")
