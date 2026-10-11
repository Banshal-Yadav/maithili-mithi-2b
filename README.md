<p align="center">
  <img src="assets/Maithili-Mithi-2B.png" alt="Maithili Mithi 2B" width="100%">
</p>

# Maithili Mithi 2B

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Bansal123%2Fmaithili--mithi--2b-blue)](https://huggingface.co/Bansal123/maithili-mithi-2b)
[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-Bansal123%2Fmaithili--instruction--tuning-yellow)](https://huggingface.co/datasets/Bansal123/maithili-instruction-tuning)
[![Base Model](https://img.shields.io/badge/Base%20Model-Google%20Gemma--2--2B--it-orange)](https://huggingface.co/google/gemma-2-2b-it)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Maithili%20(mai)-teal)](https://en.wikipedia.org/wiki/Maithili_language)

Source code and fine-tuning pipeline for Maithili Mithi 2B, an instruction-tuned language model based on Google's Gemma 2 2B architecture specialized for the Maithili language (mai).

- Model Weights: [Hugging Face (Bansal123/maithili-mithi-2b)](https://huggingface.co/Bansal123/maithili-mithi-2b)
- Instruction Dataset: [Hugging Face (Bansal123/maithili-instruction-tuning)](https://huggingface.co/datasets/Bansal123/maithili-instruction-tuning)
- Base Architecture: `google/gemma-2-2b-it` / `unsloth/gemma-2-2b-it-bnb-4bit`
- License: MIT (derivative of Gemma 2; subject to Google Gemma Terms of Use)

---

## Overview

Maithili Mithi is optimized for:
1. High-accuracy bidirectional translation between English and Maithili.
2. Direct conversational question-answering in natural Maithili without Hindi lexical mixing.
3. Proper handling of Maithili grammatical features, including native verb inflections (छी, छथि, अछि, छल, आयल, करैत) and negative particles (preferring नइ over Hindi नहीं/नहि).

The model was trained on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) using Unsloth 4-bit LoRA.

---

## Available Models

| Release | Checkpoint / Format | File | Target Use Case |
| :--- | :--- | :--- | :--- |
| v3.2 | GGUF (Q8_0, 2.78 GB) | `maithili-v3.2-q8_0.gguf` | Recommended base model. Stable translations, structured responses. |
| v3.2.2 | GGUF (Q8_0, 2.78 GB) | `maithili-v3.2.2-q8_0.gguf` | Improved conversational grammar and agent identity. |
| v3.2 LoRA | HF PEFT Adapters | Adapter weights | Direct integration via PyTorch / Hugging Face Transformers. |

---

## Recommended System Prompt

For optimal translation fidelity and dialectal consistency in local inference engines (LM Studio, Jan AI, Ollama), use the following system prompt:

```text
You are a bilingual Maithili <-> English translator and assistant.
Your name is मिथि.

Language rules:
- Use natural native Maithili; avoid Hindi grammar and sentence structures.
- Use correct Maithili verb endings: छी, छथि, अछि, छल, आयल, करैत etc.
- Default third-person pronoun: ओ.
- Never use नहि or नहीं; always use नइ.
- Standard lexical mapping: today -> आइ | tomorrow -> काल्हि | yesterday -> गत दिन | never -> कहियो नइ.

Translation markers:
- [eng] <text> [maithili] -> translate to Maithili
- [maithili] <text> [eng] -> translate to English
```

---

## Manual Step-by-Step Execution Guide

This section describes how to set up the environment, run the data checks, execute fine-tuning, evaluate benchmarks, and export quantized models locally.

### Phase 1: Environment Setup

Clone the repository and install dependencies in an isolated Python 3.10+ virtual environment.

```bash
git clone https://github.com/Banshal-Yadav/maithili-mithi-2b.git
cd maithili-mithi-2b
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Verify that your GPU and CUDA installation can execute an Unsloth training step:

```bash
python scripts/debug/debug_train.py
```
This script loads a minimal batch, runs one forward and backward pass, and verifies that bfloat16 and gradient calculations execute properly.

---

### Phase 2: Data Preprocessing and Leak Verification

The training dataset (~24,000 samples) combines Wikipedia dumps, the AI4Bharat IN22 corpus, synthetic dialogues, and curated domain subsets (`Dataset/Culture/`, `Dataset/Conversation/`, `Dataset/Translation/`).

Before training, verify corpus hygiene:

```bash
python scripts/core/count_leaks.py
```

This script parses the cleaned articles in `Dataset/Wikipedia Articles cleaned/wiki_cleaned.jsonl` and counts common Hindi leakage tokens (तथा, नहीं, है, हैं, था, थी). High corpus purity is indicated by zero or negligible counts of Hindi-specific verb markers.

---

### Phase 3: Fine-Tuning (LoRA Behavioral Patch)

To train the v3.2 behavioral patch using Unsloth LoRA on top of base Gemma 2 2B:

```bash
python scripts/train/train_v3.2.py
```

What this script executes:
- Loads the weighted dataset from `Dataset/v3.2 dataset/` (59 action vocabulary error corrections, 61 system alert corrections, 61 mixed instruction samples).
- Loads `unsloth/gemma-2-2b-it-bnb-4bit`.
- Configures LoRA with rank r=32, alpha=32 across all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Trains for 2 epochs at a learning rate of 8e-6 with gradient accumulation.
- Saves the trained LoRA adapter to `models/v3.2/maithili-lora-v3.2/`.

To quickly verify that the trained patch resolved technical alert errors without full benchmarking:

```bash
python scripts/train/validate_patch.py
```

---

### Phase 4: Diagnostic Benchmarking and Evaluation

The evaluation suite tests formatting adherence, translation fidelity, and cultural question answering.

#### 1. Rule-Based Scoring and Metrics Plot
```bash
python evals/scripts/compute_metrics.py
```
Outputs model accuracy metrics across diagnostic categories and saves a comparative performance chart to `evals/benchmark_legacy/realworld_eval/benchmark_grouped_chart.png`.

#### 2. Comparative Baseline Evaluation
To evaluate how foundational Indian language models compare against Maithili tasks:
```bash
python evals/scripts/sarvam_baseline_eval.py
```
Runs 25 translation prompts through the Sarvam-1 baseline to demonstrate script and dialectal breakdown in non-specialized models.

#### 3. LLM-as-a-Judge Evaluation (Optional)
If you have a Google Gemini API key configured:
```bash
# Windows:
$env:GEMINI_API_KEY="your-api-key"
# Linux/macOS:
export GEMINI_API_KEY="your-api-key"

python evals/scripts/auto_judge.py
python evals/scripts/write_v3.2_judgements.py
```
This evaluates responses on a 1 to 5 scale for Fluency, Accuracy, and Instruction Adherence, outputting `llm_judged_v3.2.json`.

---

### Phase 5: GGUF Quantization and Local Deployment

#### 1. Export GGUF Weights
To merge LoRA adapters into 16-bit safetensors and convert them to GGUF format for local runtimes:

```bash
python scripts/train/export_v3.2_gguf.py
```

#### 2. Run with Ollama
```bash
ollama create maithili-mithi -f Modelfile
ollama run maithili-mithi "मैथिलीमे अनुवाद करू: I am learning about artificial intelligence."
```

#### 3. Run with Python Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "google/gemma-2-2b-it"
lora_id = "Bansal123/maithili-mithi-2b"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_id)

prompt = "<start_of_turn>user\n[eng] Where can I find good food in Mithila? [maithili]<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Automated Agent Execution Option

If you prefer an automated setup using an AI coding agent (such as [NirCode](https://banshal-yadav.github.io/nirCode-page/), Cursor, Claude Code, Cline, Windsurf, or Codex), provide the agent with:

**[`AGENT.md`](AGENT.md)**

That file contains a strict machine-executable specification detailing step-by-step commands, dependency installation, smoke tests, and expected outputs to execute the complete pipeline autonomously.

---

## License

- Code and fine-tuned weights: [MIT License](LICENSE)
- Base model: [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- Datasets:
  - Wikimedia Maithili (CC BY-SA 4.0)
  - AI4Bharat IN22 (CC BY 4.0)
  - [Bansal123/maithili-instruction-tuning](https://huggingface.co/datasets/Bansal123/maithili-instruction-tuning) (MIT License)
