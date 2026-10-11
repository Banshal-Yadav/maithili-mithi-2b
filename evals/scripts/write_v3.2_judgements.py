import json
import os

judgements = [
  {"fluency": 4, "accuracy": 1, "instruction_following": 5, "reasoning": "Grammar is okay but missed 'outside' and 'cricket' completely (translates to 'They were playing yesterday')."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect past tense translation."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect future tense translation."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect complex sentence translation."},
  {"fluency": 5, "accuracy": 4, "instruction_following": 5, "reasoning": "Good attempt at localizing the idiom meaningfully."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect English translation."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect English translation."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect complex English translation."},
  {"fluency": 5, "accuracy": 1, "instruction_following": 5, "reasoning": "Incorrect factual answer (Haripur)."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Good factual paragraph on Makhan."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Accurate description of Darbhanga."},
  {"fluency": 5, "accuracy": 1, "instruction_following": 5, "reasoning": "Hallucinated that Sama Chakeva is about eating curd and butter from a pond."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Described pond wonderfully."},
  {"fluency": 5, "accuracy": 4, "instruction_following": 5, "reasoning": "Listed four festivals successfully."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Good description within the 15-word limit."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 3, "reasoning": "Followed Yes/No but added an extra explanatory sentence."},
  {"fluency": 4, "accuracy": 1, "instruction_following": 5, "reasoning": "Hallucinated heavily, describing digging up wheat instead of preparing Chura-Dahi."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Output valid JSON formatting as requested."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Excellent conversational greeting."},
  {"fluency": 5, "accuracy": 3, "instruction_following": 5, "reasoning": "Gave an evasive answer ('very good, very joyful') instead of answering what was made for dinner."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Excellent opinionated view on village life vs city."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Polite response serving water."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Simplified the sentence accurately in Maithili."},
  {"fluency": 5, "accuracy": 2, "instruction_following": 5, "reasoning": "Meaning slightly degraded during rewrite."},
  {"fluency": 5, "accuracy": 4, "instruction_following": 5, "reasoning": "Formalized properly in Maithili."},
  {"fluency": 5, "accuracy": 1, "instruction_following": 5, "reasoning": "Completely changed the meaning to 'my water is empty'."},
  {"fluency": 5, "accuracy": 4, "instruction_following": 5, "reasoning": "Good short summarization."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Excellent succinct summarization."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect one-sentence summary of the tortoise and hare."},
  {"fluency": 5, "accuracy": 4, "instruction_following": 5, "reasoning": "Decent continuation."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect creative continuation of a story."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Great dialogue continuation."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect bilingual control."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Perfect bilingual control."},
  {"fluency": 5, "accuracy": 5, "instruction_following": 5, "reasoning": "Followed bilingual setup generating a sentence."},
  {"fluency": 5, "accuracy": 4, "instruction_following": 5, "reasoning": "Meaning slightly off ('I am dear to reading books' instead of 'I like')."}
]

from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = EVALS_DIR / "benchmark_legacy" / "realworld_eval"

file_path_v32 = str(BASE_DIR / "eval_results_v3.2.json")
with open(file_path_v32, "r", encoding="utf-8") as f:
    data_v32 = json.load(f)

for i in range(len(data_v32)):
    data_v32[i]["judge_scores"] = judgements[i]
    
output_file = str(BASE_DIR / "llm_judged_v3.2.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_v32, f, ensure_ascii=False, indent=2)

print(f"Saved judged V3.2 data to {output_file}")
