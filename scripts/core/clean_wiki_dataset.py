import json
import re
import os

def clean_hindi_contamination(input_file, output_file):
    # Dictionary mapping Hindi target words to Maithili replacements.
    # The \b boundary ensures we only match whole words.
    replacements = {
        r'\bकि\b': 'जे',
        r'\bथा\b': 'छल',
        r'\bआर\b': 'आ',
        r'\bहै\b': 'अछि',
        r'\bऔर\b': 'आ',
        r'\bथी\b': 'छलीह'
    }
    
    # Compile regex patterns for performance across thousands of lines
    compiled_patterns = {re.compile(pattern): replacement for pattern, replacement in replacements.items()}
    
    processed_count = 0
    replacement_counts = {k.replace(r'\b', ''): 0 for k in replacements.keys()}

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                
                # Apply each compiled regex pattern
                for pattern, replacement in compiled_patterns.items():
                    text, count = pattern.subn(replacement, text)
                    if count > 0:
                        key = pattern.pattern.replace(r'\b', '')
                        replacement_counts[key] += count
                        
                data["text"] = text
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError:
                continue

    print(f"Processed {processed_count} lines. Saved to {output_file}")
    print("Replacement Summary:")
    for word, count in replacement_counts.items():
        print(f"  {word} -> {count} replacements")

# Paths based on current environment
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
input_path = str(BASE_DIR / "Dataset" / "Wikipedia Articles cleaned" / "wiki_processed.jsonl")
output_path = str(BASE_DIR / "Dataset" / "Wikipedia Articles cleaned" / "wiki_cleaned.jsonl")

# Run the cleaner
clean_hindi_contamination(input_path, output_path)
