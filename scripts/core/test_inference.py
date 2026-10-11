from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "models/sarvam-1/v1/hf_merged_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

def test_model(prompt_text):
    # Use the EXACT format we trained on
    formatted_prompt = f"User: {prompt_text}\nAssistant: "
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        temperature=0.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n--- Test ---\n{result}\n------------\n")

# Run the Bidirectional Tests
test_model("[eng] The dog does not bite. [maithili]")
test_model("[maithili] मखाना मिथिलाक पोखरिमे उगैत अछि। [eng]")
