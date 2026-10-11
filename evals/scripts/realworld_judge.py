import os
import json
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Define the structured output format for Gemini
class Judgment(BaseModel):
    fluency: int = Field(description="Score 1-5 for naturalness and grammatical correctness of Maithili/English.")
    accuracy: int = Field(description="Score 1-5 for factual accuracy and semantic understanding.")
    instruction_following: int = Field(description="Score 1-5 for strictly following constraints (e.g., formatting, translation vs summarization).")
    action_clarity: int = Field(description="Score 1-5 for how clear and actionable the response is, especially for system alerts and errors.")
    reasoning: str = Field(description="One concise sentence explaining the given scores.")

def judge_results(model_name, results_file, output_file):
    print(f"\n--- Judging {model_name.upper()} ---")
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    # Check for API key explicitly
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable is not set. Cannot run LLM Judge.")
        return

    client = genai.Client()
    
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    judged_data = []
    
    system_instruction = """
    You are an expert bilingual judge for Maithili and English evaluating a real-world AI Assistant.
    Rate the model's response based on the provided prompt/instruction. Be STRICT and objective.
    
    1. Fluency (1-5): Naturalness, grammatical correctness, lack of 'robot-speak' or unnatural English leakage.
    2. Accuracy (1-5): Semantic understanding, factual correctness, successful translation of meaning.
    3. Instruction Following (1-5): Adherence to specific prompt constraints (e.g., "explain in 1 sentence", "JSON format", "only say what to do").
    4. Action Clarity (1-5): Does the response actually help the user? Are instructions/steps clear, safe, and direct for system alerts?
    """

    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] Judging: {item['name']}")
        
        prompt = f"Category: {item['category']}\nPrompt/Instruction: {item['prompt']}\n\nModel Response: {item.get('response', '')}"
        
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
            item["judge_scores"] = {"fluency": 0, "accuracy": 0, "instruction_following": 0, "action_clarity": 0, "reasoning": f"Error: {str(e)}"}
            
        judged_data.append(item)
        time.sleep(1.5) # Prevent rate limits

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(judged_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved judged data to {output_file}")

if __name__ == "__main__":
    from pathlib import Path
    MODELS = ["v2", "v3", "v3.1", "v3.2"]
    EVALS_DIR = Path(__file__).resolve().parents[1]
    BASE_DIR = EVALS_DIR / "benchmark_legacy" / "realworld_eval"
    
    for m in MODELS:
        in_file = os.path.join(BASE_DIR, f"realworld_results_{m}.json")
        out_file = os.path.join(BASE_DIR, f"realworld_judged_{m}.json")
        
        if os.path.exists(in_file):
            if not os.path.exists(out_file):
                judge_results(m, in_file, out_file)
            else:
                print(f"Skipping {m}, already judged.")
