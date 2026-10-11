import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
file_path = str(BASE_DIR / "Dataset" / "Wikipedia Articles cleaned" / "wiki_cleaned.jsonl")

targets = {
    "कि": r"\bकि\b",
    "और": r"\bऔर\b",
    "आर": r"\bआर\b",
    "नहीं": r"\bनहीं\b",
    "तथा": r"\bतथा\b",
    "है": r"\bहै\b",
    "हैं": r"\bहैं\b",
    "था": r"\bथा\b",
    "थी": r"\bथी\b",
    "लिए": r"\bलिए\b",
}

counts = Counter()

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            for label, pattern in targets.items():
                matches = re.findall(pattern, text)
                counts[label] += len(matches)

    print(json.dumps(dict(counts), indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
