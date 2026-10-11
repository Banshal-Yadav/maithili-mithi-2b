import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_chart():
    # Load results
    with open('eval_v3.2_results.json', 'r', encoding='utf-8') as f:
        v32 = json.load(f)
    with open('eval_v3.2.2_results.json', 'r', encoding='utf-8') as f:
        v322 = json.load(f)

    # Convert to DataFrames
    df_v32 = pd.DataFrame(v32)
    df_v322 = pd.DataFrame(v322)

    # Calculate response lengths (characters)
    df_v32['resp_len'] = df_v32['response'].str.len()
    df_v322['resp_len'] = df_v322['response'].str.len()

    # Aggregate by category
    avg_v32 = df_v32.groupby('category')['resp_len'].mean()
    avg_v322 = df_v322.groupby('category')['resp_len'].mean()

    # Create plot
    categories = avg_v32.index
    x = range(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], avg_v32, width, label='v3.2', color='#3498db', alpha=0.8)
    rects2 = ax.bar([i + width/2 for i in x], avg_v322, width, label='v3.2.2', color='#e74c3c', alpha=0.8)

    # Add labels and formatting
    ax.set_ylabel('Avg Response Length (chars)')
    ax.set_title('Maithili Model Comparison: Response Verbosity by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    
    # Save chart
    chart_path = 'eval_comparison_chart.png'
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    generate_chart()
