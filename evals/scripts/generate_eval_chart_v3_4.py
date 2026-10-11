import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_chart():
    # Load results
    results = []
    files = {
        'v3.2': 'eval_v3.2_results.json',
        'v3.2.2': 'eval_v3.2.2_results.json',
        'v3.4 (Idioms)': 'eval_v3.4_results.json'
    }
    
    for label, filename in files.items():
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df['model_label'] = label
                df['resp_len'] = df['response'].str.len()
                results.append(df)
        else:
            print(f"Warning: {filename} not found.")

    if not results:
        print("No results to plot.")
        return

    df_all = pd.concat(results)

    # Aggregate by category and model
    pivot_df = df_all.pivot_table(index='category', columns='model_label', values='resp_len', aggfunc='mean')

    # Create plot
    ax = pivot_df.plot(kind='bar', figsize=(12, 7), width=0.8, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)

    # Add labels and formatting
    ax.set_ylabel('Avg Response Length (chars)')
    ax.set_xlabel('Category')
    ax.set_title('Maithili Model Evolution: Response Verbosity (v3.2 vs v3.2.2 vs v3.4)')
    ax.legend(title='Model Version')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    
    # Save chart
    chart_path = 'eval_comparison_v3.4_chart.png'
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    generate_chart()
