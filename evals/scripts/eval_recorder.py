"""
Maithili Mithi 2B - Eval Recorder
Run this script, type responses from Jan AI, saves everything to eval_results.json
"""

import json
import os
from datetime import datetime

PROMPTS = [
    # QA (5)
    {"id": 1, "category": "QA", "prompt": "मिथिला क्षेत्र कतऽ अछि?"},
    {"id": 2, "category": "QA", "prompt": "विद्यापति के छलथि?"},
    {"id": 3, "category": "QA", "prompt": "छठ पूजा कखन होइत अछि?"},
    {"id": 4, "category": "QA", "prompt": "मखाना कतय होइत अछि?"},
    {"id": 5, "category": "QA", "prompt": "Tell me about Chhath Puja."},

    # Translation (5)
    {"id": 6,  "category": "Translation", "prompt": "[eng] I will never forget this day. [maithili]"},
    {"id": 7,  "category": "Translation", "prompt": "[eng] The cow gives milk every morning. [maithili]"},
    {"id": 8,  "category": "Translation", "prompt": "[eng] Never lie to your parents. [maithili]"},
    {"id": 9,  "category": "Translation", "prompt": "[maithili] हम काल्हि बजार जाएब। [eng]"},
    {"id": 10, "category": "Translation", "prompt": "[maithili] हमर बाबूजी किसान छथि। [eng]"},

    # Instruction Following (5)
    {"id": 11, "category": "Instruction", "prompt": "एहि वाक्यकेँ सरल बनाउ: वर्षाऋतुमे नदीमे जलस्तर अत्यधिक वृद्धि पाबैत अछि।"},
    {"id": 12, "category": "Instruction", "prompt": "Correct this: मैं कल बाजार जाऊंगा।"},
    {"id": 13, "category": "Instruction", "prompt": "एहि notification कें मैथिलीमे बुझाउ: Your order has been shipped and will arrive in 3 days."},
    {"id": 14, "category": "Instruction", "prompt": "१ लाइनमे बताउ — What is democracy?"},
    {"id": 15, "category": "Instruction", "prompt": "Correct this: वह आज नहीं आएगा।"},

    # Conversational / Fluency (5)
    {"id": 16, "category": "Conversational", "prompt": "हमरा बुखार अछि, की करी?"},
    {"id": 17, "category": "Conversational", "prompt": "हमर बच्चा राति भरि कानैत अछि, की करी?"},
    {"id": 18, "category": "Conversational", "prompt": "गाममे रहब नीक की शहरमे?"},
    {"id": 19, "category": "Conversational", "prompt": "हम बहुत उदास छी, की करी?"},
    {"id": 20, "category": "Conversational", "prompt": "मोबाइल चार्ज नइ होइत अछि, की करी?"},
]

def record_eval(model_name):
    results = []
    output_file = f"eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    print(f"\n{'='*60}")
    print(f"EVAL RECORDER — {model_name}")
    print(f"{'='*60}")
    print("Load the model in Jan AI with NO system prompt.")
    print("For each prompt, copy the model response and paste here.")
    print("Press Enter twice when done pasting a response.")
    print(f"Results will be saved to: {output_file}")
    print(f"{'='*60}\n")

    input("Press Enter when ready...")

    for p in PROMPTS:
        print(f"\n[{p['id']}/20] [{p['category']}]")
        print(f"PROMPT: {p['prompt']}")
        print("RESPONSE (paste below, press Enter twice when done):")

        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        response = "\n".join(lines).strip()

        results.append({
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "response": response,
            "model": model_name
        })

        print(f"✓ Saved response {p['id']}/20")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Done! Saved to {output_file}")
    print(f"Share this file to get judged.")
    print(f"{'='*60}")

    return output_file

if __name__ == "__main__":
    print("Which model are you testing?")
    print("1. v3.2")
    print("2. v3.2.2")
    print("3. other")
    choice = input("Enter choice (1/2/3): ").strip()

    model_map = {"1": "v3.2", "2": "v3.2.2", "3": input("Enter model name: ")}
    model_name = model_map.get(choice, "unknown")

    record_eval(model_name)