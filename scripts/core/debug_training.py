from unsloth import FastLanguageModel
from datasets import load_dataset
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
model_name = str(BASE_DIR / "models" / "sarvam-1" / "base")
dataset_file = str(BASE_DIR / "Dataset" / "v4.1_combined" / "train_shuffled_v4_1.jsonl")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    load_in_4bit = True,
)

print(f"EOS Token: {tokenizer.eos_token}")
print(f"PAD Token: {tokenizer.pad_token}")

def format_prompts(examples):
    texts = examples["text"]
    new_texts = []
    for text in texts:
        text = text.replace("<start_of_turn>user\n", "User: ")
        text = text.replace("\n<end_of_turn>\n<start_of_turn>model\n", "\nAssistant: ")
        if tokenizer.eos_token:
            text = text.replace("\n<end_of_turn>", tokenizer.eos_token)
        else:
            text = text.replace("\n<end_of_turn>", "</s>") # Default fallback
        new_texts.append(text)
    return { "text" : new_texts }

dataset = load_dataset("json", data_files=dataset_file, split="train[:5]")
mapped_dataset = dataset.map(format_prompts, batched = True)

for i in range(3):
    print(f"Sample {i}:\n{mapped_dataset[i]['text']}\n{'-'*20}")
