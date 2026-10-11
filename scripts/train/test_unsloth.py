import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
from unsloth import FastLanguageModel
import torch

model_name = "unsloth/gemma-2-2b-it-bnb-4bit"
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 512,
        load_in_4bit = True,
    )
    print("Model loaded successfully.")
    FastLanguageModel.for_inference(model)
    print("Inference mode enabled.")
except Exception as e:
    print(f"Error: {e}")
