# AGENT.md: Autonomous Pipeline Execution Guide

> **Instructions for AI Coding Agents** ([NirCode](https://banshal-yadav.github.io/nirCode-page/), Cursor, Claude Code, Cline, Windsurf, Codex, etc.)  
> This file is a self-contained execution specification for reproducing, running, evaluating, and exporting **Maithili Mithi 2B**. Execute these steps in order.

---

## 1. System & Environment Specifications

- **Language:** Python 3.10+
- **Hardware Requirement:** NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (Tested on RTX 4050 Laptop GPU)
- **Primary Libraries:** PyTorch (with CUDA), Unsloth, Hugging Face TRL, Transformers, Datasets, PEFT
- **Base Model:** `unsloth/gemma-2-2b-it-bnb-4bit` (Hugging Face)
- **Trained Model Artifact:** `Bansal123/maithili-mithi-2b` (Hugging Face)
- **Curated Dataset:** [Bansal123/maithili-instruction-tuning](https://huggingface.co/datasets/Bansal123/maithili-instruction-tuning) (Hugging Face)

### Setup Command:
```bash
pip install -r requirements.txt
```

---

## 2. Repository Layout & Roles

```text
maithili-mithi-2b/
├── Dataset/                 # Curated Maithili instruction datasets
│   ├── Conversation/        # Natural everyday dialogues
│   ├── Culture/             # Cultural Mithila QA (Chhath, Sakraat, etc.)
│   ├── Translation/         # English/Hindi <-> Maithili pairs
│   ├── v3.2 dataset/        # Surgical patch data (alerts, action vocab)
│   └── Wikipedia Articles cleaned/ # Cleaned Wikipedia articles
├── scripts/
│   ├── core/
│   │   ├── count_leaks.py   # Checks for residual Hindi contamination
│   │   └── clean_wiki_dataset.py # Corpus cleaning logic
│   ├── train/
│   │   ├── train_v3.2.py    # Unsloth LoRA behavior patch trainer
│   │   ├── validate_patch.py# Quick inference patch verification
│   │   └── export_v3.2_gguf.py # Merge LoRA & export to GGUF
│   └── debug/
│       └── debug_train.py   # 7-second GPU/Unsloth smoke test
├── evals/
│   ├── scripts/
│   │   ├── compute_metrics.py # Rule-based diagnostic benchmarking
│   │   ├── auto_judge.py     # Gemini LLM-as-a-judge scoring
│   │   └── sarvam_baseline_eval.py # Comparative baseline eval
│   └── results/             # Evaluation output JSON files and charts
├── Modelfile                # Ollama deployment manifest
└── requirements.txt         # Pinned python dependencies
```

---

## 3. End-to-End Execution Sequence

### Step 0: GPU & Unsloth Smoke Test
Before running training or evaluation, verify that CUDA and Unsloth are working on the host:
```bash
python scripts/debug/debug_train.py
```
*Expected Result:* Compiles 1 training step, prints loss, and exits with code 0 in under 30 seconds.

---

### Step 1: Data Hygiene Check (Hindi Decontamination)
Run the leak auditor to verify the training corpus has no untracked Hindi tokens:
```bash
python scripts/core/count_leaks.py
```
*Expected Result:* Returns JSON count of Hindi tokens (e.g. `कि`, `और`, `नहीं`). High purity means counts close to 0.

---

### Step 2: Run Behavior Patch Training
Fine-tune the model with Unsloth LoRA on the curated patch dataset:
```bash
python scripts/train/train_v3.2.py
```
*What this script does:*
1. Loads the 222 weighted samples from `Dataset/v3.2 dataset/`.
2. Loads `unsloth/gemma-2-2b-it-bnb-4bit` with 4-bit quantization.
3. Applies LoRA ($r=32, \alpha=32$) over all linear projection layers.
4. Trains for 2 epochs with learning rate $8\times 10^{-6}$.
5. Saves final LoRA adapter to `models/v3.2/maithili-lora-v3.2/`.

---

### Step 3: Validate Patch Results
Check that the model correctly answers technical alerts and instructions without Hindi mixing:
```bash
python scripts/train/validate_patch.py
```
*Expected Result:* Generates responses for test prompts (Drive storage full, low battery, OTP expired) and prints clean Maithili outputs.

---

### Step 4: Run Diagnostic Benchmark & Metrics
Compute rule-based performance scores and generate the comparative visualization chart:
```bash
python evals/scripts/compute_metrics.py
```
*Expected Result:* Prints accuracy percentages across categories (Translation, Yes/No, Factual) and outputs `benchmark_grouped_chart.png`.

---

### Step 5: (Optional) Run LLM-as-a-Judge Evaluation
If a `GEMINI_API_KEY` is set in the environment:
```bash
python evals/scripts/auto_judge.py
```
*Expected Result:* Queries Gemini to score fluency, accuracy, and instruction adherence on a 1-5 scale.

---

### Step 6: Export & Deploy to Ollama

1. **Merge to FP16 & Export GGUF:**
   ```bash
   python scripts/train/export_v3.2_gguf.py
   ```
2. **Serve with Ollama:**
   ```bash
   ollama create maithili-mithi -f Modelfile
   ollama run maithili-mithi "प्रणाम! अहाँक परिचय दिअ।"
   ```

---

## 4. Key Architectural Constraints & Rules for Agents
- **Path Resolution:** Always resolve paths dynamically using `pathlib.Path(__file__).resolve()`. Do not introduce hardcoded absolute paths (`C:\Users\...` or `/home/...`).
- **Git Safety:** Never commit model weights (`*.safetensors`, `*.gguf`, `*.pt`, `*.bin`), intermediate checkpoints, or `llama.cpp/`. Keep `.gitignore` intact.
- **Language Fidelity:** The model must always output `नइ` instead of `नहीं`/`नहि`, and use native Maithili verb suffixes (`छी`, `छथि`, `अछि`, `छल`).
