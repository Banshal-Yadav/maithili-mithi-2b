import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = str(EVALS_DIR / "benchmark_legacy" / "realworld_eval")
MODELS = ["v2", "v3.1", "v3.2"]

def plot_judge_scores():
    final_scores = {model: {"fluency": 0, "accuracy": 0, "instruction_following": 0, "overall": 0} for model in MODELS}
    
    for model in MODELS:
        filepath = os.path.join(RESULTS_DIR, f"llm_judged_{model}.json")
        if not os.path.exists(filepath):
            print(f"Waiting on {filepath}")
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        total_f, total_a, total_i = 0, 0, 0
        valid_count = 0
        
        for item in data:
            scores = item.get("judge_scores", {})
            if "fluency" in scores and scores["fluency"] > 0:
                total_f += scores["fluency"]
                total_a += scores["accuracy"]
                total_i += scores["instruction_following"]
                valid_count += 1
                
        if valid_count > 0:
            final_scores[model]["fluency"] = (total_f / valid_count) * 20 # Convert 1-5 to percentage
            final_scores[model]["accuracy"] = (total_a / valid_count) * 20
            final_scores[model]["instruction_following"] = (total_i / valid_count) * 20
            final_scores[model]["overall"] = ((total_f + total_a + total_i) / (3 * valid_count)) * 20

    # Plotting
    metrics = ["Fluency", "Accuracy", "Instruction Following", "Overall"]
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for i, model in enumerate(MODELS):
        if final_scores[model]["overall"] == 0: continue
        
        vals = [
            final_scores[model]["fluency"],
            final_scores[model]["accuracy"],
            final_scores[model]["instruction_following"],
            final_scores[model]["overall"]
        ]
        
        offset = (i - len(MODELS)/2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=model.upper(), color=colors[i])
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    ax.set_ylabel('Score (Scaled to 100%)', fontsize=12)
    ax.set_title('LLM-as-a-Judge Evaluation: Maithili V2 vs V3.1 vs V3.2', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "llm_judge_chart.png")
    plt.savefig(chart_path)
    print(f"\nLLM Judge chart saved to: {chart_path}")
    
    # Print raw scores
    print("\n--- Final Scores ---")
    for model in MODELS:
        print(f"{model.upper()}: {final_scores[model]['overall']:.1f}%")

if __name__ == "__main__":
    plot_judge_scores()
