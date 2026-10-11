import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

model_path = r"models\v3.2\lora\maithili-lora-v3.2"

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

long_prompt = """Translate this long paragraph to Maithili:
Artificial intelligence (AI) is transforming the way we interact with technology, powering everything from virtual assistants on our smartphones to advanced medical diagnostics and autonomous vehicles. By analyzing vast amounts of data and identifying complex patterns, AI systems can make decisions, predict outcomes, and automate repetitive tasks with unprecedented speed and accuracy. However, this rapid technological advancement also raises important ethical considerations, such as data privacy, algorithmic bias, and the potential displacement of jobs. As AI continues to evolve and integrate deeper into daily life, it is crucial for society to establish robust frameworks and guidelines to ensure these powerful tools are developed responsibly and used for the greater benefit of humanity."""

messages = [{"role": "user", "content": long_prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

print("\nPrompt:")
print(long_prompt)
print("\nGenerating response...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        use_cache=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.15
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
with open(r"eval\realworld_eval\long_paragraph_test.txt", "w", encoding="utf-8") as f:
    f.write(response)
print("Saved response to eval\\realworld_eval\\long_paragraph_test.txt")
