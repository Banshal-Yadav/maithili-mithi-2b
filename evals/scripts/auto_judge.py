import os
import json
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Define the structured output format for Gemini
class Judgment(BaseModel):
    fluency: int = Field(description="Score 1-5 for naturalness and grammatical correctness of Maithili/English.")
    accuracy: int = Field(description="Score 1-5 for factual accuracy and translation fidelity.")
    instruction_following: int = Field(description="Score 1-5 for strictly following constraints (e.g., word count, JSON format).")
    reasoning: str = Field(description="One concise sentence explaining the scores.")

def judge_results(model_name, results_file, output_file):
    print(f"\n--- Judging {model_name.upper()} ---")
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    client = genai.Client() # Automatically picks up GEMINI_API_KEY
    
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    judged_data = []
    
    system_instruction = """
    You are an expert bilingual judge for Maithili and English.
    Rate the model's response based on the provided prompt/instruction.
    1. Fluency (1-5): Does it sound natural, grammatically correct, and free of hallucinations or English leakage (unless requested)?
    2. Accuracy (1-5): Is the translation accurate? Are facts correct? Are cultural nuances captured?
    3. Instruction Following (1-5): Did it follow specific constraints (length, formatting, yes/no only)?
    Provide a 1-sentence reasoning.
    """

    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] Judging: {item['name']}")
        
        prompt = f"Prompt/Instruction: {item['prompt']}\n\nModel Response: {item.get('response', '')}"
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=Judgment,
                    temperature=0.1,
                ),
            )
            
            judgement = json.loads(response.text)
            item["judge_scores"] = judgement
            
        except Exception as e:
            print(f"Error judging item {i}: {e}")
            item["judge_scores"] = {"fluency": 0, "accuracy": 0, "instruction_following": 0, "reasoning": f"Error: {str(e)}"}
            
        judged_data.append(item)
        time.sleep(1) # Prevent rate limits

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(judged_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved judged data to {output_file}")

if __name__ == "__main__":
    from pathlib import Path
    MODELS = ["v2", "v3", "v3.1", "v3.2"]
    EVALS_DIR = Path(__file__).resolve().parents[1]
    BASE_DIR = EVALS_DIR / "benchmark_legacy" / "realworld_eval"
    
    for m in MODELS:
        in_file = os.path.join(BASE_DIR, f"eval_results_{m}.json")
        out_file = os.path.join(BASE_DIR, f"llm_judged_{m}.json")
        
        if os.path.exists(in_file):
            if not os.path.exists(out_file):
                judge_results(m, in_file, out_file)
            else:
                print(f"Skipping {m}, already judged.")
