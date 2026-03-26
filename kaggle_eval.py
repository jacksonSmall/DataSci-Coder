#!/usr/bin/env python3
"""Kaggle T4 evaluation notebook — test fine-tuned vs base 14B.

Paste this into a Kaggle notebook cell AFTER training completes
(or load the adapter from a dataset). Requires GPU T4 + Internet.
"""

# ── Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run(
    ["pip", "install", "-q", "unsloth", "datasets"],
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

# ── Load fine-tuned model ─────────────────────────────────────────────
from unsloth import FastLanguageModel
import torch
import time
import json

print(f"Loading {MODEL_NAME} with adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

# Load LoRA adapter weights
from peft import PeftModel
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model = model.merge_and_unload()
print("Adapter loaded and merged!")

# Put in eval mode
FastLanguageModel.for_inference(model)


# ── Generation helper ─────────────────────────────────────────────────
def generate_response(prompt, max_new_tokens=1024, temperature=0.1):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
        )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Sanity checks ────────────────────────────────────────────────────
SANITY_CHECKS = [
    {
        "prompt": "Write a function to train a logistic regression model with sklearn.",
        "keywords": ["LogisticRegression", "fit", "predict", "import"],
        "type": "code",
        "category": "machine_learning",
    },
    {
        "prompt": "Show me how to create a correlation heatmap with seaborn.",
        "keywords": ["heatmap", "corr", "seaborn", "plt"],
        "type": "code",
        "category": "visualization",
    },
    {
        "prompt": "Write a PyTorch training loop for a neural network.",
        "keywords": ["torch", "backward", "optimizer", "loss"],
        "type": "code",
        "category": "deep_learning",
    },
    {
        "prompt": "How do I handle missing values in a pandas DataFrame?",
        "keywords": ["fillna", "dropna", "isna", "pandas"],
        "type": "code",
        "category": "data_wrangling",
    },
    {
        "prompt": "Implement k-means clustering from scratch in Python.",
        "keywords": ["centroids", "cluster", "distance", "def"],
        "type": "code",
        "category": "machine_learning",
    },
    {
        "prompt": "Write code to perform a train-test split and cross-validation.",
        "keywords": ["train_test_split", "cross_val", "sklearn", "score"],
        "type": "code",
        "category": "machine_learning",
    },
    {
        "prompt": "Show how to build a random forest classifier with feature importance.",
        "keywords": ["RandomForest", "feature_importances", "fit", "import"],
        "type": "code",
        "category": "machine_learning",
    },
    {
        "prompt": "Write a function to perform PCA dimensionality reduction.",
        "keywords": ["PCA", "fit_transform", "explained_variance", "import"],
        "type": "code",
        "category": "machine_learning",
    },
    {
        "prompt": "How do I build an XGBoost model with early stopping?",
        "keywords": ["xgboost", "early_stopping", "eval_set", "fit"],
        "type": "code",
        "category": "machine_learning",
    },
    {
        "prompt": "Write code to perform time series forecasting with ARIMA.",
        "keywords": ["ARIMA", "forecast", "statsmodels", "fit"],
        "type": "code",
        "category": "statistics",
    },
]

print("\n" + "=" * 60)
print("SANITY CHECKS — Fine-tuned Model")
print("=" * 60)

ft_results = []
ft_pass = 0

for i, check in enumerate(SANITY_CHECKS):
    prompt = check["prompt"]
    keywords = check["keywords"]

    print(f"\n  [{i+1}/{len(SANITY_CHECKS)}] {check['category']}")
    print(f"  Prompt: {prompt}")

    start = time.time()
    response = generate_response(prompt)
    elapsed = time.time() - start

    resp_lower = response.lower()
    kw_hits = {kw: kw.lower() in resp_lower for kw in keywords}
    kw_score = sum(kw_hits.values()) / len(kw_hits)

    # Failure detection
    failures = []
    if len(response) < 50:
        failures.append("too_short")
    if "<|im_end|>" in response or "<|im_start|>" in response:
        failures.append("special_token_leak")
    if "![" in response and response.count("![") >= 2:
        failures.append("image_artifact")

    passed = kw_score >= 0.5 and not failures and len(response) > 50

    result = {
        "prompt": prompt,
        "passed": passed,
        "kw_score": round(kw_score, 3),
        "kw_hits": kw_hits,
        "failures": failures,
        "response_len": len(response),
        "time_s": round(elapsed, 1),
        "response_preview": response[:400],
    }
    ft_results.append(result)
    if passed:
        ft_pass += 1

    status = "PASS" if passed else "FAIL"
    print(f"  {status} | KW: {sum(kw_hits.values())}/{len(kw_hits)}"
          f" | Len: {len(response)} | {elapsed:.1f}s")
    if not passed:
        missing = [k for k, v in kw_hits.items() if not v]
        if missing:
            print(f"  Missing: {missing}")
        if failures:
            print(f"  Flags: {failures}")
        print(f"  Preview: {response[:200]}...")

print(f"\nFine-tuned: {ft_pass}/{len(SANITY_CHECKS)} passed")


# ── Base model comparison ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading BASE model (no adapter) for comparison...")
print("=" * 60)

# Free memory
import gc
del model
gc.collect()
torch.cuda.empty_cache()

base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(base_model)


def generate_base(prompt, max_new_tokens=1024, temperature=0.1):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = base_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = base_tokenizer(text, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return base_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print("\n" + "=" * 60)
print("SANITY CHECKS — Base Model")
print("=" * 60)

base_results = []
base_pass = 0

for i, check in enumerate(SANITY_CHECKS):
    prompt = check["prompt"]
    keywords = check["keywords"]

    print(f"\n  [{i+1}/{len(SANITY_CHECKS)}] {check['category']}")
    print(f"  Prompt: {prompt}")

    start = time.time()
    response = generate_base(prompt)
    elapsed = time.time() - start

    resp_lower = response.lower()
    kw_hits = {kw: kw.lower() in resp_lower for kw in keywords}
    kw_score = sum(kw_hits.values()) / len(kw_hits)

    failures = []
    if len(response) < 50:
        failures.append("too_short")
    if "<|im_end|>" in response or "<|im_start|>" in response:
        failures.append("special_token_leak")

    passed = kw_score >= 0.5 and not failures and len(response) > 50

    result = {
        "prompt": prompt,
        "passed": passed,
        "kw_score": round(kw_score, 3),
        "kw_hits": kw_hits,
        "failures": failures,
        "response_len": len(response),
        "time_s": round(elapsed, 1),
        "response_preview": response[:400],
    }
    base_results.append(result)
    if passed:
        base_pass += 1

    status = "PASS" if passed else "FAIL"
    print(f"  {status} | KW: {sum(kw_hits.values())}/{len(kw_hits)}"
          f" | Len: {len(response)} | {elapsed:.1f}s")

print(f"\nBase model: {base_pass}/{len(SANITY_CHECKS)} passed")


# ── Comparison summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"Fine-tuned: {ft_pass}/{len(SANITY_CHECKS)} passed")
print(f"Base model: {base_pass}/{len(SANITY_CHECKS)} passed")
print(f"Lift: {ft_pass - base_pass:+d}")

print(f"\n{'Prompt':<55} {'FT':>4} {'Base':>4}")
print("-" * 67)
for ft_r, base_r in zip(ft_results, base_results):
    ft_s = "PASS" if ft_r["passed"] else "FAIL"
    base_s = "PASS" if base_r["passed"] else "FAIL"
    prompt_short = ft_r["prompt"][:52] + "..."
    print(f"{prompt_short:<55} {ft_s:>4} {base_s:>4}")

# Save results
results = {
    "fine_tuned": {
        "passed": ft_pass,
        "total": len(SANITY_CHECKS),
        "details": ft_results,
    },
    "base": {
        "passed": base_pass,
        "total": len(SANITY_CHECKS),
        "details": base_results,
    },
    "lift": ft_pass - base_pass,
}
with open("/kaggle/working/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /kaggle/working/eval_results.json")
