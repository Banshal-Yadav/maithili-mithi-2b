import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

from unsloth import FastLanguageModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
# We will test EP1 first, as behavior patches are often best at epoch 1
MODEL_PATH = os.path.join(BASE_DIR, "v3.2", "maithili-lora-v3.2-ep1")
# MODEL_PATH = os.path.join(BASE_DIR, "v3.2", "maithili-lora-v3.2") # Uncomment to test final

print(f"Loading model for Validation: {MODEL_PATH}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

PROMPTS = [
    {
        "category": "System Alert - Drive Storage",
        "prompt": "System Alert: Your Google Drive storage is 95% full. Because you are out of storage, you cannot upload new files to Drive, back up photos or videos to Google Photos. You have run out of your 15 GB initial free storage. Delete old files to recover your system."
    },
    {
        "category": "System Alert - Battery",
        "prompt": "Your device battery is severely degraded and running critically low at 5%. Please plug in your charger immediately to prevent unsaved data loss and system shutdown."
    },
    {
        "category": "Mixed Instruction",
        "prompt": "Here is an English sentence: 'The database migration completed successfully.' Translate it perfectly into Maithili."
    },
    {
        "category": "Product Wording - Technical Action",
        "prompt": "Click on the 'Settings' gear icon, navigate to the 'Privacy and Security' tab, and disable 'Third-party cookies' to secure your browsing session."
    },
    {
        "category": "Product Wording - Installation",
        "prompt": "The software installation wizard requires administrative privileges to proceed. Please click 'Run as Administrator' when the User Account Control prompt appears."
    }
]

def generate_response(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=256, use_cache=True, temperature=0.3, repetition_penalty=1.1)
    response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
    return response

print("\n--- BEHAVIOR PATCH VALIDATION TESTS ---")
for p in PROMPTS:
    print(f"\n[{p['category']}]")
    print(f"User: {p['prompt']}")
    response = generate_response(p['prompt'])
    print(f"Model: {response}")
    print("-" * 60)

print("\nValidation complete.")
