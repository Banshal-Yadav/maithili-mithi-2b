import json
import subprocess
import os
import time
from pathlib import Path

# Configuration
MODEL = "maithili-v32:latest"
EVALS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = str(EVALS_DIR / "results" / "detailed_eval_results.jsonl")
SCRIPT_DIR = str(EVALS_DIR / "scripts")

# Prompt Categories
# 1. Detailed Summarization (Higher complexity passages)
# 2. Maithili to English Translation (Nuanced sentences)

test_prompts = [
    # --- DETAILED SUMMARIZATION (Full detail requested) ---
    {
        "category": "Detailed Summary",
        "prompt": "एहि अनुच्छेदक पूरा विस्तारसँ सार लिखू: मिथिलाक लोक संस्कृति बड्ड समृद्ध अछि। एतय अनेक पावनि-तिहार मनाओल जाइत अछि जाहिमे सामा-चकेबा, मधुश्रावणी आ कोजागरा प्रमुख अछि। लोकगीत आ लोकनृत्यक सेहो अपन महत्त्व अछि। अरिपन आ मधुबनी पेंटिंगक परम्परा तँ आब विश्वप्रसिद्ध भए गेल अछि।"
    },
    {
        "category": "Detailed Summary",
        "prompt": "एहि गद्यांशक विस्तृत विवरण दऽ कए सार लिखू: दरभंगा महाराज कामेश्वर सिंहक समयमे मिथिलाक औद्योगिक आ शैक्षणिक विकास अपन चरम पर छल। ओ एतय कतेको चीनी मिल आ शिक्षण संस्थानक स्थापना कएने छलाह। हुनक उदारता आ दूरदर्शिताक कारणेँ आजुक दरभंगा एकटा प्रमुख शहर बनि सकल अछि।"
    },
    {
        "category": "Detailed Summary",
        "prompt": "नीचाँ देल गेल जानकारीक पूरा विस्तारसँ सार लिखू: कृषि मिथिलाक अर्थव्यवस्थाक मुख्य आधार थिक। धान, गहुँ आ मकै मुख्य फसल थिक। मुदा एतय मखान, पान आ मछरीक उत्पादन सेहो पैघ पैमाना पर होइत अछि जाहि सँ किसान सभक आय बढ़ैत छनि। जल-जमावक समस्याक बादो मिथिलाक भूमि उपजाऊ अछि।"
    },
    {
        "category": "Detailed Summary",
        "prompt": "विस्तारपूर्वक सार लिखू: आधुनिक युगमे इन्टरनेट आ सोशल मीडियाक प्रभाव मिथिलामे सेहो देखाइत अछि। युवक-युवती सभ आइकाल्हि मैथिलीमे कंटेंट बना रहल छथि जाहि सँ भाषाक प्रचार-प्रसार भऽ रहल अछि। डिजिटल लाइब्रेरी आ ई-बुकक माध्यम सँ प्राचीन मैथिली साहित्य आब सबकेँ उपलब्ध अछि।"
    },
    {
        "category": "Detailed Summary",
        "prompt": "पूरा विवरणक साथ सार लिखू: विद्यापति केवल एकटा कविक नाम नहि, अपितु ओ एकटा युगक प्रतीक छलाह। हुनक पदावलीमे प्रेम, भक्ति आ श्रृंगारक जे समन्वय देखाइत अछि, ओ कतहु अन्यत्र भेटब दुर्लभ अछि। ओ मैथिली साहित्यकेँ एकटा नव दिशा देलाह आ कतेको पीढ़ि हुनका सँ प्रभावित भेल अछि।"
    },

    # --- MAITHILI TO ENGLISH (Nuanced/Translation) ---
    {
        "category": "Maithili-to-English",
        "prompt": "नीचाँ देल मैथिली वाक्यक अंग्रेजीमे अनुवाद करू: हमरा अपन मिथिलाक माटि सँ बड्ड प्रेम अछि। [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "अंग्रेजीमे अनुवाद करू: जँ अहाँ परिश्रम करब तँ निश्चित रूप सँ सफल होयब। [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "Translate to English: मिथिलाक संस्कार आ परंपरा पूर्ण विश्व मे अद्वितीय अछि। [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "English Translation: ओ आइ-काल्हि अपन गाम कतहु नहि जाइत छथि। [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "नीचाँ देल वाक्यक अंग्रेजीमे अनुवाद करू: अहाँक घर कतय अछि आ अहाँ की करैत छी? [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "Translate: माछ, मखान आ पान मिथिलाक शान थिक। [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "English Translation: विद्यापति मैथिलीक कोकिल कहल जाइत छथि। [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "अंग्रेजीमे अनुवाद करू: ज्ञान बिना जीवन अंधकारमय अछि। [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "Translate to English: आई आकाश मे कारी मेघ छाएल अछि। [eng]"
    },
    {
        "category": "Maithili-to-English",
        "prompt": "English translation: कोशी नदीक बाढि सँ प्रतिवर्ष तबाही होइत अछि। [eng]"
    }
]

def get_ollama_response(prompt):
    try:
        # Increase timeout for detailed summaries (90s)
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=120,
            encoding='utf-8',
            errors='ignore',
            shell=True
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {str(e)}"

def run_evaluation():
    print(f"Starting detailed evaluation on {MODEL}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, item in enumerate(test_prompts):
            print(f"[{i+1}/{len(test_prompts)}] Processing {item['category']}...")
            response = get_ollama_response(item['prompt'])
            
            result_entry = {
                "id": i + 1,
                "category": item['category'],
                "prompt": item['prompt'],
                "response": response
            }
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            
    print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()
