import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = str(EVALS_DIR / "benchmark_legacy" / "realworld_eval")
MODELS = ["v2", "v3", "v3.1", "v3.2"]

def plot_realworld_scores():
    final_scores = {model: {"fluency": 0, "accuracy": 0, "instruction_following": 0, "action_clarity": 0, "overall": 0} for model in MODELS}
    
    for model in MODELS:
        filepath = os.path.join(RESULTS_DIR, f"realworld_judged_{model}.json")
        if not os.path.exists(filepath):
            print(f"Waiting on {filepath}")
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        total_f, total_a, total_i, total_c = 0, 0, 0, 0
        valid_count = 0
        
        for item in data:
            scores = item.get("judge_scores", {})
            if "fluency" in scores and scores["fluency"] > 0:
                total_f += scores["fluency"]
                total_a += scores["accuracy"]
                total_i += scores["instruction_following"]
                total_c += scores.get("action_clarity", 0) # Handle missing key if older script used
                valid_count += 1
                
        if valid_count > 0:
            final_scores[model]["fluency"] = (total_f / valid_count) * 20 # Convert 1-5 to percentage
            final_scores[model]["accuracy"] = (total_a / valid_count) * 20
            final_scores[model]["instruction_following"] = (total_i / valid_count) * 20
            final_scores[model]["action_clarity"] = (total_c / valid_count) * 20
            final_scores[model]["overall"] = ((total_f + total_a + total_i + total_c) / (4 * valid_count)) * 20

    # Plotting
    metrics = ["Fluency", "Accuracy", "Instruction Following", "Action Clarity", "Overall"]
    x = np.arange(len(metrics))
    width = 0.20
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plotted = False
    for i, model in enumerate(MODELS):
        if final_scores[model]["overall"] == 0: continue
        plotted = True
        
        vals = [
            final_scores[model]["fluency"],
            final_scores[model]["accuracy"],
            final_scores[model]["instruction_following"],
            final_scores[model]["action_clarity"],
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
                        
    if not plotted:
        print("No valid scores to plot.")
        return

    ax.set_ylabel('Score (Scaled to 100%)', fontsize=12)
    ax.set_title('Real-World Assistant Evaluation: V2 vs V3 vs V3.1 vs V3.2', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right', fontsize=11, framealpha=0.9)
    
    fig.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "realworld_eval_chart.png")
    plt.savefig(chart_path, dpi=300)
    print(f"\nReal-World Evaluation chart saved to: {chart_path}")
    
    # Print raw scores
    print("\n--- Final Scores ---")
    for model in MODELS:
        if final_scores[model]["overall"] > 0:
            print(f"{model.upper()}: {final_scores[model]['overall']:.1f}%")

if __name__ == "__main__":
    plot_realworld_scores()
