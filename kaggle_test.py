#!/usr/bin/env python3
"""Kaggle T4 quick test — run 5 prompts against the fine-tuned model.

Paste into a Kaggle notebook cell after training completes.
"""

# ── Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run(
    ["pip", "install", "-q", "unsloth"],
    check=True,
)

# ── Config ────────────────────────────────────────────────────────────
import os
os.environ["UNSLOTH_USE_FUSED_CE_LOSS"] = "0"
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"

MODEL_NAME = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
ADAPTER_DIR = "/kaggle/working/lora_adapter"
MAX_SEQ_LEN = 1024

SYSTEM_PROMPT = (
    "You are an expert data science coding assistant. "
    "Return clean, runnable Python code with inline comments. "
    "No explanations outside code blocks unless asked."
)

# ── Load model with adapter ──────────────────────────────────────────
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

print(f"Loading adapter from {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model = model.merge_and_unload()
FastLanguageModel.for_inference(model)
print("Ready!\n")


# ── Generate helper ──────────────────────────────────────────────────
def generate(prompt, max_new_tokens=1024, temperature=0.7):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Test prompts ─────────────────────────────────────────────────────
TEST_PROMPTS = [
    "Write a function to train a logistic regression model"
    " with sklearn and print the classification report.",
    "Show me how to build a PyTorch CNN for CIFAR-10"
    " image classification.",
    "How do I perform time series forecasting with ARIMA"
    " in statsmodels?",
    "Write code to do hyperparameter tuning with Optuna"
    " for an XGBoost model.",
    "Implement a complete EDA pipeline for a new dataset"
    " including missing values, distributions, and correlations.",
]

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"{'='*60}")
    print(f"Prompt {i}: {prompt}")
    print(f"{'='*60}")
    response = generate(prompt)
    print(response)
    print()
