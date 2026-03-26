#!/usr/bin/env python3
"""Kaggle full eval notebook — load adapter, quick test, hard eval vs base.

Cell 0: Install dependencies (kernel restarts)
Cell 1: Everything below
"""

# ── Install (uncomment for first cell, then comment out) ─────────
# import subprocess, sys, os
# subprocess.run([
#     sys.executable, "-m", "pip", "install", "-q",
#     "unsloth==2026.3.4",
#     "transformers==4.51.3",
#     "accelerate==0.34.2",
#     "bitsandbytes==0.45.5",
#     "peft",
# ], check=True)
# os.kill(os.getpid(), 9)

# ── Config ───────────────────────────────────────────────────────
import os
os.environ["UNSLOTH_USE_FUSED_CE_LOSS"] = "0"
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"

from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import time
import gc

MODEL_NAME = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
ADAPTER_DIR = "/kaggle/input/models/jacksonsm/datasci-coder-14b-lora/pytorch/default/1"

SYSTEM_PROMPT = (
    "You are an expert data science coding assistant. "
    "Return clean, runnable Python code with inline comments. "
    "No explanations outside code blocks unless asked."
)

# ── Load fine-tuned model ────────────────────────────────────────
print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,
    load_in_4bit=True,
    gpu_memory_utilization=0.85,
)

print(f"Loading adapter from {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
# Do NOT merge_and_unload — merging into 4-bit causes garbage output
FastLanguageModel.for_inference(model)
print("Ready!\n")


# ── Generate helper ──────────────────────────────────────────────
def generate(prompt, max_new_tokens=1024, temperature=0.7):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Sanity check ─────────────────────────────────────────────────
print("Sanity check:")
print(generate("Write a Python function that adds two numbers.", max_new_tokens=256))
print()


# ── Quick Test: 5 prompts ────────────────────────────────────────
print("=" * 60)
print("QUICK TEST")
print("=" * 60)

prompts = [
    "Write a function to train a logistic regression model with sklearn and print the classification report.",
    "Show me how to build a PyTorch CNN for CIFAR-10 image classification.",
    "How do I perform time series forecasting with ARIMA in statsmodels?",
    "Write code to do hyperparameter tuning with Optuna for an XGBoost model.",
    "Implement a complete EDA pipeline for a new dataset including missing values, distributions, and correlations.",
]

for i, p in enumerate(prompts, 1):
    print("=" * 60)
    print(f"Prompt {i}: {p}")
    print("=" * 60)
    print(generate(p))
    print()


# ── Hard Eval Setup ──────────────────────────────────────────────
HARD_CHECKS = [
    {"prompt": "Implement Bayesian A/B testing with a Beta-Binomial model. Show prior updates, posterior distributions, and probability that variant B beats A.",
     "keywords": ["beta", "posterior", "prior", "probability", "scipy"], "category": "stats"},
    {"prompt": "Build a stacking ensemble classifier with cross-validated base models (RandomForest, XGBoost, LogisticRegression) and a meta-learner.",
     "keywords": ["StackingClassifier", "cross_val_predict", "meta", "fit", "estimators"], "category": "ml"},
    {"prompt": "Write a custom PyTorch learning rate scheduler with linear warmup and cosine decay.",
     "keywords": ["LambdaLR", "warmup", "cosine", "scheduler", "step"], "category": "dl"},
    {"prompt": "Implement SHAP feature importance for a gradient boosting model with summary plot and waterfall plot.",
     "keywords": ["shap", "TreeExplainer", "summary_plot", "values"], "category": "ml"},
    {"prompt": "Write a complete Kaplan-Meier survival analysis with log-rank test comparing two groups.",
     "keywords": ["KaplanMeier", "log_rank", "survival", "fit", "lifelines"], "category": "stats"},
    {"prompt": "Implement a bidirectional LSTM with attention mechanism for text classification in PyTorch.",
     "keywords": ["LSTM", "attention", "bidirectional", "forward", "softmax"], "category": "dl"},
    {"prompt": "Write code for time series cross-validation with expanding window, fitting an ARIMA model at each fold and reporting RMSE.",
     "keywords": ["TimeSeriesSplit", "ARIMA", "rmse", "forecast", "fold"], "category": "stats"},
    {"prompt": "Implement Isolation Forest anomaly detection with contamination tuning and visualize the decision boundary on 2D data.",
     "keywords": ["IsolationForest", "contamination", "predict", "scatter", "anomaly"], "category": "ml"},
    {"prompt": "Build a complete text classification pipeline: TF-IDF vectorization, chi-squared feature selection, SVM classifier, and cross-validated evaluation.",
     "keywords": ["TfidfVectorizer", "chi2", "SVC", "Pipeline", "cross_val"], "category": "ml"},
    {"prompt": "Implement a variational autoencoder (VAE) in PyTorch with the reparameterization trick, KL divergence loss, and reconstruction loss.",
     "keywords": ["reparameterize", "kl_divergence", "mu", "log_var", "decoder"], "category": "dl"},
    {"prompt": "Write code to detect and handle multicollinearity using VIF, then compare Ridge, Lasso, and ElasticNet regression.",
     "keywords": ["VIF", "Ridge", "Lasso", "ElasticNet", "variance_inflation"], "category": "stats"},
    {"prompt": "Implement a GAN (Generative Adversarial Network) for generating synthetic tabular data with a generator and discriminator network.",
     "keywords": ["Generator", "Discriminator", "fake", "real", "adversarial"], "category": "dl"},
]


def eval_hard(gen_fn, label):
    print("\n" + "=" * 60)
    print(f"HARD EVAL: {label}")
    print("=" * 60)
    results = []
    passed = 0
    for i, check in enumerate(HARD_CHECKS):
        prompt = check["prompt"]
        keywords = check["keywords"]
        print(f"\n  [{i+1}/{len(HARD_CHECKS)}] {check['category']}")
        print(f"  Prompt: {prompt[:80]}...")
        start = time.time()
        response = gen_fn(prompt)
        elapsed = time.time() - start
        resp_lower = response.lower()
        kw_hits = {kw: kw.lower() in resp_lower for kw in keywords}
        kw_score = sum(kw_hits.values()) / len(kw_hits)
        failures = []
        if len(response) < 100:
            failures.append("too_short")
        if "<|im_end|>" in response or "<|im_start|>" in response:
            failures.append("special_token_leak")
        if "![" in response and response.count("![") >= 2:
            failures.append("image_artifact")
        has_code = ("def " in response or "class " in response or "import " in response)
        if not has_code:
            failures.append("no_code")
        ok = kw_score >= 0.5 and not failures and len(response) > 100
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status} | KW: {sum(kw_hits.values())}/{len(kw_hits)} | Len: {len(response)} | {elapsed:.1f}s")
        if not ok:
            missing = [k for k, v in kw_hits.items() if not v]
            if missing:
                print(f"  Missing: {missing}")
            if failures:
                print(f"  Flags: {failures}")
            print(f"  Preview: {response[:250]}...")
        results.append({"passed": ok, "kw_score": kw_score, "prompt": prompt})
    print(f"\n{label}: {passed}/{len(HARD_CHECKS)} passed")
    return passed, results


# ── Eval fine-tuned model ────────────────────────────────────────
ft_passed, ft_results = eval_hard(
    lambda p: generate(p, max_new_tokens=1024, temperature=0.1),
    "Fine-tuned Model"
)

# ── Free memory and load base model ─────────────────────────────
del model
gc.collect()
torch.cuda.empty_cache()

print("\nLoading base model for comparison...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,
    load_in_4bit=True,
    gpu_memory_utilization=0.85,
)
FastLanguageModel.for_inference(base_model)


def gen_base_hard(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = base_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = base_tokenizer(text, return_tensors="pt")
    device = next(base_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = base_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=base_tokenizer.eos_token_id,
            eos_token_id=base_tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return base_tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()


# ── Eval base model ─────────────────────────────────────────────
base_passed, base_results = eval_hard(gen_base_hard, "Base Model")


# ── Comparison ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HARD EVAL COMPARISON")
print("=" * 60)
print(f"Fine-tuned: {ft_passed}/{len(HARD_CHECKS)}")
print(f"Base model: {base_passed}/{len(HARD_CHECKS)}")
print(f"Lift: {ft_passed - base_passed:+d}")
print()

cats = {}
for ft_r, base_r in zip(ft_results, base_results):
    cat = [c for c in HARD_CHECKS if c["prompt"] == ft_r["prompt"]][0]["category"]
    if cat not in cats:
        cats[cat] = {"ft": 0, "base": 0, "total": 0}
    cats[cat]["total"] += 1
    if ft_r["passed"]:
        cats[cat]["ft"] += 1
    if base_r["passed"]:
        cats[cat]["base"] += 1

print(f"{'Category':<12} {'FT':>6} {'Base':>6}")
print("-" * 28)
for cat in sorted(cats):
    c = cats[cat]
    print(f"{cat:<12} {c['ft']}/{c['total']:>4} {c['base']}/{c['total']:>4}")

print()
print(f"{'Prompt':<55} {'FT':>4} {'Base':>4}")
print("-" * 67)
for ft_r, base_r in zip(ft_results, base_results):
    p = ft_r["prompt"][:52] + "..."
    f = "PASS" if ft_r["passed"] else "FAIL"
    b = "PASS" if base_r["passed"] else "FAIL"
    print(f"{p:<55} {f:>4} {b:>4}")

# ── Cleanup for next cell ──────────────────────────────────────
del base_model, base_tokenizer
gc.collect()
torch.cuda.empty_cache()
print("\nModels freed — GPU ready for next eval cell.")
