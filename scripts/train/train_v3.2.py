"""
Stage-3.2 Maithili LoRA Training (Behavior Patch) - Final Eager Version
======================================================================
Target: Patch behavior using 8e-6 LR.
Base: v3.1 checkpoint 543
Fixes:
- Disables Inductor/Triton explicitly.
- Resumes V3.1 via get_peft_model + set_peft_model_state_dict.
- use_gradient_checkpointing = True (Eager mode, no Unsloth Triton kernels).
- Pre-formats dataset correctly for TRL latest.
"""

import os, sys, glob, json, torch, random

# ENV FIXES (Disable Torch Compile and Inductor completely BEFORE any import)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import load_peft_weights, set_peft_model_state_dict
from pathlib import Path

# CONFIG
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "v3.2 dataset")
CHECKPOINT_31 = os.path.join(BASE_DIR, "v3.1", "maithili-stage3.1-output", "checkpoint-543")
OUTPUT_DIR = os.path.join(BASE_DIR, "v3.2", "maithili-stage3.2-output")
FINAL_LORA = os.path.join(BASE_DIR, "v3.2", "maithili-lora-v3.2")
EP1_SAVE_PATH = os.path.join(BASE_DIR, "v3.2", "maithili-lora-v3.2-ep1")

DATASETS = [
    {"file": "v3_action_vocab_errors.jsonl", "weight": 1.5},
    {"file": "v2_system_alerts.jsonl", "weight": 1.2},
    {"file": "v2_mixed_instructions.jsonl", "weight": 1.0}
]

EPOCHS = 2
LR = 8e-6
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 4

# LOAD AND WEIGHT DATA
print("Preparing Datasets...")
all_samples = []
for entry in DATASETS:
    fpath = os.path.join(DATASET_DIR, entry["file"])
    file_samples = []
    with open(fpath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            text = obj["text"].replace("\n<end_of_turn>", "<end_of_turn>")
            file_samples.append({"text": text})
    
    random.shuffle(file_samples)
    num_to_add = int(len(file_samples) * entry["weight"])
    weighted_samples = list(file_samples)
    extra_needed = num_to_add - len(file_samples)
    if extra_needed > 0:
        weighted_samples.extend(random.sample(file_samples, k=min(extra_needed, len(file_samples))))
    all_samples.extend(weighted_samples)
    print(f"  {entry['file']}: {len(file_samples)} base -> {len(weighted_samples)} weighted")

random.shuffle(all_samples)
dataset = Dataset.from_list(all_samples)
print(f"Total Patch Dataset Size: {len(dataset)}")

# LOAD BASE MODEL
print("Loading Base Model: unsloth/gemma-2-2b-it-bnb-4bit")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-2b-it-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
)

# INIT LoRA (Identical to V3.1 Config, but using Unsloth native True checkpointing)
print("Applying PEFT Configuration...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True, # Explicitly using standard PyTorch checkpointing to avoid compiling
    random_state = 42,
)

# LOAD V3.1 WEIGHTS IF AVAILABLE
if os.path.exists(CHECKPOINT_31):
    print(f"Restoring V3.1 adapters from {CHECKPOINT_31}...")
    adapters_weights = load_peft_weights(CHECKPOINT_31)
    set_peft_model_state_dict(model, adapters_weights)
    print("V3.1 Adapters successfully loaded.")
else:
    print(f"Notice: Local checkpoint {CHECKPOINT_31} not found.")
    print("Proceeding with fresh LoRA configuration on base model.")

# SETUP TRAINER (Minimal Signature)
training_args = TrainingArguments(
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_steps = 5,
    num_train_epochs = EPOCHS,
    learning_rate = LR,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_torch",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 42,
    output_dir = OUTPUT_DIR,
    report_to = "none",
    save_strategy = "epoch",
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    args = training_args,
    processing_class = tokenizer,
)

print("Starting Stage-3.2 Behavior Patch Training (Eager Mode)...")
try:
    trainer.train()
except Exception as e:
    print(f"TRAIN FAILED: {e}")
    sys.exit(1)

# SAVE
print(f"Saving to {FINAL_LORA}...")
model.save_pretrained(FINAL_LORA)
tokenizer.save_pretrained(FINAL_LORA)

# Check Ep1 checkpoint
print("Checking for Epoch 1 checkpoint...")
ep1_checkpoint = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
if ep1_checkpoint:
    ep1_checkpoint.sort(key=lambda x: int(x.split("-")[-1]))
    src = ep1_checkpoint[0]
    import shutil
    if os.path.exists(EP1_SAVE_PATH):
        shutil.rmtree(EP1_SAVE_PATH)
    shutil.copytree(src, EP1_SAVE_PATH)
    print(f"Epoch 1 checkpoint copied to {EP1_SAVE_PATH}")

print("Training Complete.")
