import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

adapter_path = r"models\v3.2\lora\maithili-lora-v3.2"

print(f"Loading V3.2 adapter from {adapter_path}...")
model = AutoModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model.eval()

prompts = [
    "हमरा एहि विषयमे किछु नइ मालूम अछि।",
    "जँ अहाँ समय पर नइ आएब, तँ की होएत?",
    "ई काम करब जरूरी नइ अछि, सही?",
    "ब्लूटूथ की थिक? छोट उत्तर दियौ।",
    "Inflation की थिक? २ लाइनमे बुझाउ।",
    "हमर समस्या गंभीर अछि, की करी?",
    "ई ठीक नइ चलि रहल अछि।",
    "सब दवाइ बिना डॉक्टरक पूछले ले सकैत छी, कोनो दिक्कत नइ होइत।",
    "OTP दोसरकेँ देलासँ कोनो नुकसान नइ होइत।",
    "‘आपन KYC अपडेट करू’ ई मैसेज लिंक भेजल अछि, की करब?",
    "रातिमे उत्तर दिस माथ राखिकए सुतब नुकसानदेह अछि?",
    "AI मॉडल ‘Hallucination’ किएक करैत अछि?",
    "हम परीक्षा मे फेल भऽ गेलहुँ, बहुत खराब लगि रहल अछि।",
    "Login error आबि रहल अछि, की check करब?",
    "GDP की थिक?"
]

output_file = "custom_test_results_v3_2.txt"
if os.path.exists(output_file):
    os.remove(output_file)

print("\nRunning Inference sequentially on V3.2...\n" + "="*50)

with torch.no_grad():
    for i, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"[{i}/{len(prompts)}] Prompt: {prompt}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Response:\n{response}\n")
            f.write("=" * 50 + "\n\n")
        
        print(f"Finished {i}/{len(prompts)}.")
