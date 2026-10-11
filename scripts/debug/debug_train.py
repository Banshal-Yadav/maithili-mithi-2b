import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Minimal test
model_name = "unsloth/gemma-2-2b-it-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 512,
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"])

data = Dataset.from_list([{"text": "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>"}])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = data,
    dataset_text_field = "text",
    max_seq_length = 512,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        bf16 = True,
        logging_steps = 1,
        output_dir = "debug_out",
        report_to = "none",
    ),
)
print("Starting debug train...")
trainer.train()
print("Done.")
