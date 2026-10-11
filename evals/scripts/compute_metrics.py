import os
import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# This is a basic diagnostic string matching metric script.
# For full LLM-as-a-judge you would use another model, but here
# we will use rule-based metrics to score the diagnostic benchmark reliably.

EVALS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = str(EVALS_DIR / "benchmark_legacy" / "realworld_eval")
MODELS = ["v2", "v3", "v3.1", "v3.2"]

# Define rules for specific test cases to automatically score them (0.0 to 1.0)
def score_response(item):
    name = item["name"]
    prompt = item["prompt"]
    response = item.get("response", "").strip().lower()
    
    score = 0.5 # Default middle score for manual review
    
    # Yes/no format instruction
    if name == "i4_yes_no":
        if "नहि" in response and len(response.split()) < 5:
            score = 1.0
        else:
            score = 0.0
            
    # Length constraint formatting
    elif name == "i3_length_constraint":
        words = len(response.split())
        if 8 <= words <= 12:
            score = 1.0
        elif 5 <= words <= 15:
            score = 0.5
        else:
            score = 0.0
            
    # JSON formatting
    elif name == "i6_json":
        if "{" in response and "}" in response and ":" in response:
            score = 1.0
        else:
            score = 0.0
            
    # Bilingual Control Check
    elif "bilingual" in item["category"]:
        if "eng_to_mai" in name:
            if not any(c.isascii() and c.isalpha() for c in response): # Checks if there is no English
                score = 1.0
            else:
                score = 0.0
        elif "mai_to_eng" in name:
            if all(c.isascii() or c.isspace() or c in ".,!?" for c in response): # Checks if it's purely English
                score = 1.0
            else:
                score = 0.0
                
    # Factual Hallucinations
    elif name == "f1_vidyapati":
        if "कवि" in response or "लेखक" in response:
            score = 1.0
        else:
            score = 0.0
            
    return score

def compute_metrics():
    model_scores = {model: {"total": 0, "count": 0, "categories": {}} for model in MODELS}
    
    for model in MODELS:
        filepath = os.path.join(RESULTS_DIR, f"eval_results_{model}.json")
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}. Skipping {model}.")
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            cat = item["category"]
            score = score_response(item)
            
            model_scores[model]["total"] += score
            model_scores[model]["count"] += 1
            
            if cat not in model_scores[model]["categories"]:
                model_scores[model]["categories"][cat] = {"total": 0, "count": 0}
            
            model_scores[model]["categories"][cat]["total"] += score
            model_scores[model]["categories"][cat]["count"] += 1

    # Print Report
    print("="*40)
    print("BENCHMARK METRICS REPORT")
    print("="*40)
    
    final_scores = {}
    category_scores = {model: {} for model in MODELS}
    
    for model in MODELS:
        if model_scores[model]["count"] > 0:
            avg_score = (model_scores[model]["total"] / model_scores[model]["count"]) * 100
            final_scores[model] = avg_score
            print(f"Model {model.upper()} Overall Score: {avg_score:.1f}%")
            
            for cat, scores in model_scores[model]["categories"].items():
                if scores["count"] > 0:
                    category_scores[model][cat] = (scores["total"] / scores["count"]) * 100
                else:
                    category_scores[model][cat] = 0
        else:
            final_scores[model] = 0

    return final_scores, category_scores

def plot_bar_chart(final_scores, category_scores):
    # Collect all unique categories
    categories = set()
    for model_cats in category_scores.values():
        categories.update(model_cats.keys())
    categories = sorted(list(categories))
    
    # We will plot categories plus "Overall"
    x_labels = categories + ["Overall"]
    x = np.arange(len(x_labels))  # the label locations
    width = 0.25  # width of the bars
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, model in enumerate(MODELS):
        # Gather scores for this model corresponding to x_labels
        model_vals = []
        for cat in categories:
            model_vals.append(category_scores[model].get(cat, 0))
        model_vals.append(final_scores[model])
        
        offset = (i - len(MODELS)/2 + 0.5) * width
        rects = ax.bar(x + offset, model_vals, width, label=model.upper(), color=colors[i])
        
        # Add labels on bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Add labels
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Maithili LLM Benchmark Progression by Category', fontsize=14, pad=20)
    ax.set_xticks(x)
    
    # Format labels
    formatted_labels = [lbl.replace('_', ' ').title() for lbl in x_labels]
    ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=11)
    
    ax.legend(title="Models", fontsize=11, title_fontsize=12)
    ax.set_ylim(0, 115) 
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "benchmark_grouped_chart.png")
    plt.savefig(chart_path)
    print(f"\nGrouped bar chart saved to: {chart_path}")

if __name__ == "__main__":
    print("Computing metrics based on strict formatting/factual rules...")
    final_scores, category_scores = compute_metrics()
    plot_bar_chart(final_scores, category_scores)
