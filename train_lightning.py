#!/usr/bin/env python3
"""
DS-Tuned v4 — Lightning AI Training Script

Run on Lightning AI Studio with L4 GPU (24GB VRAM).
Single script: install deps → train → save → eval.

Usage:
  1. Create Lightning AI Studio with L4 GPU
  2. Upload train.jsonl + valid.jsonl to ~/data/
  3. Run: python train_lightning.py
  4. Adapter saved to ~/output/lora_adapter/
  5. Download the adapter folder when done
"""

import subprocess
import sys

# ── Install dependencies ─────────────────────────────────────────
print("Installing dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "unsloth", "datasets", "trl"],
    check=True,
)

import os
import json
import gc
import time
import re
import torch

os.environ["UNSLOTH_USE_FUSED_CE_LOSS"] = "0"
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"
os.environ["XFORMERS_DISABLED"] = "1"

# =====================================================================
# CONFIG
# =====================================================================

DATA_DIR = os.path.expanduser("~/data")
OUTPUT_DIR = os.path.expanduser("~/output/lora_adapter")

MODEL_NAME = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"

MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 1
LR = 3e-5
WARMUP_RATIO = 0.03

SYSTEM_PROMPT = (
    "You are an expert data science coding assistant. "
    "Respond ONLY with clean, runnable Python code. "
    "Use inline comments for explanation. "
    "No text outside code blocks."
)

# =====================================================================
# PHASE 1 — Training
# =====================================================================

from unsloth import FastLanguageModel

print(f"\nLoading {MODEL_NAME} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

model.print_trainable_parameters()

# ── Load data ─────────────────────────────────────────────────────
from datasets import load_dataset

train_path = os.path.join(DATA_DIR, "train.jsonl")
val_path = os.path.join(DATA_DIR, "valid.jsonl")

for p in [train_path, val_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found! Upload train.jsonl and valid.jsonl to {DATA_DIR}"
        )

with open(train_path) as f:
    sample = json.loads(f.readline())

if "messages" in sample:
    text_field = "messages"
    role_key = "role"
    content_key = "content"
elif "conversations" in sample:
    text_field = "conversations"
    role_key = "from"
    content_key = "value"
else:
    raise ValueError(f"Unknown data format. Keys found: {list(sample.keys())}")

print(f"Data format: {text_field}")

dataset = load_dataset("json", data_files={"train": train_path, "val": val_path})
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['val'])}")

# ── Tokenize with assistant-only masking ──────────────────────────

ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant",
            "human": "user", "gpt": "assistant"}

def tokenize_with_masking(examples):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
    marker_len = len(assistant_tokens)

    for convos in examples[text_field]:
        messages = []
        for msg in convos:
            role = ROLE_MAP.get(msg[role_key], msg[role_key])
            messages.append({"role": role, "content": msg[content_key]})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        labels = [-100] * len(input_ids)

        i = 0
        while i < len(input_ids):
            if input_ids[i] == im_start_id:
                candidate = input_ids[i + 1: i + 1 + marker_len]
                if candidate == assistant_tokens:
                    content_start = i + 1 + marker_len
                    content_end = content_start
                    while content_end < len(input_ids) and input_ids[content_end] != im_end_id:
                        content_end += 1
                    if content_end < len(input_ids):
                        content_end += 1
                    for j in range(content_start, min(content_end, len(labels))):
                        if attention_mask[j] == 1:
                            labels[j] = input_ids[j]
                    i = content_end
                    continue
            i += 1

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }

tokenized = dataset.map(
    tokenize_with_masking,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

sample = tokenized["train"][0]
n_loss = sum(1 for x in sample["labels"] if x != -100)
n_total = sum(1 for x in sample["attention_mask"] if x == 1)
print(f"Masking check: {n_loss}/{n_total} tokens have loss ({n_loss/max(n_total,1)*100:.0f}% are assistant)")

# ── Train ─────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

n_steps = (len(tokenized["train"]) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
print(f"\nTotal training steps: {n_steps}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    fp16=True,
    bf16=False,
    logging_steps=5,
    eval_strategy="no",
    save_strategy="no",
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    report_to="none",
    max_seq_length=MAX_SEQ_LEN,
    dataset_kwargs={"skip_prepare_dataset": True},
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized["train"],
    args=training_args,
)

print("Starting training...")
train_start = time.time()
result = trainer.train()
train_elapsed = time.time() - train_start
print(f"\nTraining complete! Final loss: {result.training_loss:.4f}")
print(f"Training time: {train_elapsed/3600:.1f} hours")

# ── Save adapter ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

config = {
    "base_model": MODEL_NAME,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "seq_length": MAX_SEQ_LEN,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
    "lr": LR,
    "train_loss": result.training_loss,
    "train_examples": len(tokenized["train"]),
    "val_examples": len(dataset["val"]),
    "train_time_hours": round(train_elapsed/3600, 2),
}
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\nAdapter saved to {OUTPUT_DIR}")
for fname in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"  {fname} ({size / 1024 / 1024:.1f} MB)")

# ── Free training resources ───────────────────────────────────────
del trainer
gc.collect()
torch.cuda.empty_cache()


# =====================================================================
# PHASE 2 — Hard Eval (FT vs Base)
# =====================================================================

print("\n" + "=" * 60)
print("PHASE 2: HARD EVAL")
print("=" * 60)

# Reload FT model for inference
del model
gc.collect()
torch.cuda.empty_cache()

print("Loading fine-tuned model for eval...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)
print("Ready!\n")


def make_generate(mdl, tok):
    def gen(prompt, max_new_tokens=1024, temperature=0.1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt")
        device = next(mdl.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=False,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()
    return gen


generate_ft = make_generate(model, tokenizer)

# Sanity check
print("Sanity check:")
print(generate_ft("Write a Python function that adds two numbers.", max_new_tokens=256))
print()

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


ft_passed, ft_results = eval_hard(
    lambda p: generate_ft(p, max_new_tokens=1024, temperature=0.1),
    "Fine-tuned Model"
)

# Free FT, load base
del model, generate_ft
gc.collect()
torch.cuda.empty_cache()

print("\nLoading base model for comparison...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(base_model)

generate_base = make_generate(base_model, base_tokenizer)

base_passed, base_results = eval_hard(generate_base, "Base Model")

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
    f_status = "PASS" if ft_r["passed"] else "FAIL"
    b_status = "PASS" if base_r["passed"] else "FAIL"
    print(f"{p:<55} {f_status:>4} {b_status:>4}")

del base_model, base_tokenizer, generate_base
gc.collect()
torch.cuda.empty_cache()


# =====================================================================
# PHASE 3 — Constraint Eval (FT vs Base)
# =====================================================================

print("\n" + "=" * 60)
print("PHASE 3: CONSTRAINT EVAL")
print("=" * 60)

print("Loading fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)

generate_ft = make_generate(model, tokenizer)

CONSTRAINT_TESTS = [
    {
        "id": "C01",
        "name": "Multi-step data cleaning",
        "prompt": (
            "Write a single function with type hints that: "
            "1) loads a CSV file, "
            "2) drops columns with more than 50% missing values, "
            "3) fills remaining numeric NaNs with the median, "
            "4) one-hot encodes all categorical columns, "
            "5) returns the cleaned DataFrame. "
            "No print statements. No explanations."
        ),
        "constraints": {
            "has_type_hints": r"def \w+\(.*:.*\)\s*->\s*",
            "loads_csv": r"read_csv",
            "drops_columns": r"drop|thresh",
            "fills_median": r"median|fillna",
            "one_hot": r"get_dummies|OneHotEncoder",
            "returns_df": r"\breturn\b",
            "no_print": lambda r: "print(" not in r,
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
    {
        "id": "C02",
        "name": "Complete ML pipeline with specific metrics",
        "prompt": (
            "Write code that: "
            "1) loads the breast cancer dataset from sklearn, "
            "2) splits 80/20 with random_state=42, "
            "3) trains a RandomForestClassifier with n_estimators=200, "
            "4) prints accuracy, precision, recall, and F1 score, "
            "5) plots a confusion matrix heatmap with seaborn, "
            "6) saves the model with joblib. "
            "All in one script. No explanations outside code."
        ),
        "constraints": {
            "breast_cancer": r"load_breast_cancer",
            "split_80_20": r"test_size\s*=\s*0\.2",
            "random_state_42": r"random_state\s*=\s*42",
            "rf_200": r"n_estimators\s*=\s*200",
            "accuracy": r"accuracy",
            "precision": r"precision",
            "recall": r"recall",
            "f1": r"f1",
            "confusion_matrix": r"confusion_matrix|ConfusionMatrix",
            "seaborn_heatmap": r"heatmap",
            "joblib_save": r"joblib\.dump|joblib\.save",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
    {
        "id": "C03",
        "name": "Statistical test with assumptions",
        "prompt": (
            "Write a function that performs a two-sample t-test: "
            "1) check normality with Shapiro-Wilk test first, "
            "2) check equal variances with Levene's test, "
            "3) if normal, use scipy ttest_ind (equal_var based on Levene result), "
            "4) if not normal, use Mann-Whitney U test instead, "
            "5) return a dict with test_used, statistic, p_value, and significant (bool at alpha=0.05). "
            "Include type hints. No explanations."
        ),
        "constraints": {
            "has_type_hints": r"def \w+\(.*:.*\)\s*->\s*",
            "shapiro": r"shapiro",
            "levene": r"levene",
            "ttest_ind": r"ttest_ind",
            "mannwhitney": r"mannwhitneyu|Mann.?Whitney",
            "equal_var_param": r"equal_var",
            "returns_dict": r"return\s*\{|return\s+dict",
            "alpha_005": r"0\.05",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
    {
        "id": "C04",
        "name": "PyTorch model with specific architecture",
        "prompt": (
            "Write a PyTorch neural network class called 'TabularNet' that: "
            "1) takes input_dim, hidden_dim, output_dim as __init__ args, "
            "2) has 3 hidden layers with BatchNorm and Dropout(0.3) after each, "
            "3) uses ReLU activations, "
            "4) has a forward method with proper type hints, "
            "5) includes a predict method that returns class labels (not logits). "
            "No explanations."
        ),
        "constraints": {
            "class_name": r"class TabularNet",
            "init_args": r"input_dim.*hidden_dim.*output_dim",
            "batchnorm": r"BatchNorm",
            "dropout_03": r"Dropout\(0\.3\)",
            "relu": r"ReLU|relu",
            "forward_method": r"def forward\(",
            "predict_method": r"def predict\(",
            "argmax": r"argmax",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
    {
        "id": "C05",
        "name": "EDA with specific visualizations",
        "prompt": (
            "Write an EDA function that takes a DataFrame and: "
            "1) prints shape, dtypes, and missing value counts, "
            "2) creates a figure with 4 subplots using plt.subplots(2,2), "
            "3) subplot 1: histogram of the first numeric column, "
            "4) subplot 2: boxplot of all numeric columns, "
            "5) subplot 3: correlation heatmap, "
            "6) subplot 4: bar chart of missing values per column, "
            "7) calls plt.tight_layout() and plt.savefig('eda_report.png'). "
            "No explanations."
        ),
        "constraints": {
            "takes_df": r"def \w+\(.*[Dd]ata[Ff]rame.*\)|def \w+\(\s*df",
            "prints_shape": r"\.shape",
            "prints_dtypes": r"\.dtypes|\.info\(\)",
            "missing_values": r"isnull|isna|missing",
            "subplots_2x2": r"subplots\s*\(\s*2\s*,\s*2",
            "histogram": r"hist|histplot",
            "boxplot": r"boxplot|box_plot",
            "heatmap": r"heatmap",
            "bar_chart": r"\.bar\(|barplot|bar_chart",
            "tight_layout": r"tight_layout",
            "savefig": r"savefig.*eda_report",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
    {
        "id": "C06",
        "name": "Cross-validated pipeline with preprocessing",
        "prompt": (
            "Write code using sklearn Pipeline that: "
            "1) imputes missing values with SimpleImputer(strategy='median'), "
            "2) scales features with StandardScaler, "
            "3) selects top 10 features with SelectKBest(f_classif, k=10), "
            "4) classifies with SVC(kernel='rbf'), "
            "5) evaluates with 5-fold cross_val_score, "
            "6) prints mean and std of scores. "
            "No explanations."
        ),
        "constraints": {
            "pipeline": r"Pipeline",
            "simple_imputer": r"SimpleImputer",
            "median_strategy": r"strategy\s*=\s*['\"]median['\"]",
            "standard_scaler": r"StandardScaler",
            "selectkbest": r"SelectKBest",
            "k_10": r"k\s*=\s*10",
            "f_classif": r"f_classif",
            "svc_rbf": r"SVC.*rbf|kernel\s*=\s*['\"]rbf['\"]",
            "cross_val": r"cross_val_score",
            "cv_5": r"cv\s*=\s*5",
            "prints_mean_std": r"mean|std",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
    {
        "id": "C07",
        "name": "Time series with specific steps",
        "prompt": (
            "Write code that: "
            "1) generates synthetic time series data with pandas date_range, "
            "2) performs ADF test for stationarity with statsmodels, "
            "3) if not stationary, differences the series, "
            "4) fits ARIMA(1,1,1) model, "
            "5) forecasts next 30 periods, "
            "6) plots actual vs forecast with matplotlib, "
            "7) prints AIC and RMSE. "
            "No explanations."
        ),
        "constraints": {
            "date_range": r"date_range",
            "adf_test": r"adfuller|ADF",
            "stationarity_check": r"p.?value|pvalue|0\.05",
            "diff": r"\.diff\(\)|differenc",
            "arima_111": r"ARIMA.*1.*1.*1|order\s*=\s*\(1",
            "forecast_30": r"forecast.*30|steps\s*=\s*30",
            "matplotlib_plot": r"plt\.plot|\.plot\(",
            "aic": r"\.aic|AIC",
            "rmse": r"rmse|RMSE|mean_squared_error",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
    {
        "id": "C08",
        "name": "Deep learning training with all best practices",
        "prompt": (
            "Write a PyTorch training function that: "
            "1) takes model, train_loader, val_loader, epochs, lr as args, "
            "2) uses Adam optimizer with the given lr, "
            "3) uses CrossEntropyLoss, "
            "4) implements early stopping with patience=5, "
            "5) tracks train and val loss per epoch in lists, "
            "6) saves best model weights with torch.save, "
            "7) returns a dict with train_losses, val_losses, best_epoch. "
            "Include type hints. No explanations."
        ),
        "constraints": {
            "type_hints": r"def \w+\(.*:.*\)\s*->\s*",
            "function_args": r"model.*train_loader.*val_loader.*epochs.*lr|model.*loader.*epoch",
            "adam": r"Adam",
            "cross_entropy": r"CrossEntropyLoss",
            "patience_5": r"patience.*5|5.*patience",
            "early_stopping": r"early.?stop|patience|best.*loss",
            "torch_save": r"torch\.save",
            "returns_dict": r"return\s*\{|return\s+dict",
            "tracks_losses": r"train.*loss|loss.*list|losses.*append",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
    {
        "id": "C09",
        "name": "Pandas operations chain",
        "prompt": (
            "Write a single chained pandas expression (no intermediate variables) that: "
            "1) reads 'sales.csv', "
            "2) filters rows where revenue > 1000, "
            "3) groups by 'region' and 'product', "
            "4) aggregates with sum of revenue and mean of quantity, "
            "5) sorts by revenue descending, "
            "6) resets the index, "
            "7) assigns to a variable called 'result'. "
            "One statement only. No explanations."
        ),
        "constraints": {
            "read_csv": r"read_csv.*sales",
            "filter_1000": r"revenue.*>.*1000|1000.*<.*revenue",
            "groupby": r"groupby",
            "region_product": r"region.*product|product.*region",
            "agg_sum": r"sum",
            "agg_mean": r"mean",
            "sort_desc": r"sort_values.*ascending\s*=\s*False",
            "reset_index": r"reset_index",
            "result_var": r"result\s*=",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 15,
        },
    },
    {
        "id": "C10",
        "name": "Comprehensive model evaluation",
        "prompt": (
            "Write a function called evaluate_classifier that: "
            "1) takes y_true and y_pred as args with type hints, "
            "2) computes accuracy, precision, recall, F1 (weighted), "
            "3) computes and plots ROC curve (requires y_prob arg too), "
            "4) computes AUC score, "
            "5) plots confusion matrix as a heatmap, "
            "6) creates a single figure with 2 subplots (ROC left, CM right), "
            "7) returns a dict with all metric values. "
            "No explanations."
        ),
        "constraints": {
            "func_name": r"def evaluate_classifier",
            "type_hints": r"def evaluate_classifier\(.*:.*\)",
            "accuracy": r"accuracy_score|accuracy",
            "precision": r"precision_score|precision",
            "recall": r"recall_score|recall",
            "f1_weighted": r"f1.*weighted|weighted.*f1",
            "roc_curve": r"roc_curve",
            "auc": r"auc|roc_auc",
            "confusion_matrix": r"confusion_matrix",
            "heatmap": r"heatmap",
            "subplots_1x2": r"subplots\s*\(\s*1\s*,\s*2",
            "returns_dict": r"return\s*\{",
            "no_explanation": lambda r: len(re.sub(r"```.*?```", "", r, flags=re.DOTALL).strip().split()) < 30,
        },
    },
]


def check_style(response):
    outside = re.sub(r"```.*?```", "", response, flags=re.DOTALL).strip()
    outside_words = len(outside.split()) if outside else 0
    total_words = len(response.split())
    return {
        "has_code_block": "```" in response,
        "has_imports": "import " in response,
        "has_inline_comments": bool(re.search(r"#\s+\w", response)),
        "code_ratio": 1 - (outside_words / max(total_words, 1)),
        "explanation_words": outside_words,
        "is_code_only": outside_words < 30,
    }


def run_constraint_eval(gen_fn, label):
    print("\n" + "=" * 60)
    print(f"CONSTRAINT EVAL: {label}")
    print("=" * 60)

    all_results = []
    total_constraints = 0
    total_met = 0
    style_scores = []

    for test in CONSTRAINT_TESTS:
        prompt = test["prompt"]
        constraints = test["constraints"]

        print(f"\n  [{test['id']}] {test['name']}")
        start = time.time()
        response = gen_fn(prompt)
        elapsed = time.time() - start

        met = 0
        details = {}
        for name, check in constraints.items():
            if callable(check):
                passed = check(response)
            else:
                passed = bool(re.search(check, response, re.IGNORECASE))
            details[name] = passed
            if passed:
                met += 1

        total_c = len(constraints)
        total_constraints += total_c
        total_met += met
        score = met / total_c

        style = check_style(response)
        style_scores.append(style)

        status = "PASS" if score >= 0.7 else "PARTIAL" if score >= 0.4 else "FAIL"
        print(f"  {status} | Constraints: {met}/{total_c} ({score:.0%})"
              f" | Style: {'code-only' if style['is_code_only'] else 'has-explanation'}"
              f" | Len: {len(response)} | {elapsed:.1f}s")

        failed = [k for k, v in details.items() if not v]
        if failed:
            print(f"  Missing: {failed}")

        all_results.append({
            "id": test["id"],
            "name": test["name"],
            "score": score,
            "met": met,
            "total": total_c,
            "style": style,
            "details": details,
        })

    overall = total_met / total_constraints
    avg_code_ratio = sum(s["code_ratio"] for s in style_scores) / len(style_scores)
    code_only_count = sum(1 for s in style_scores if s["is_code_only"])

    print(f"\n{'='*60}")
    print(f"{label} SUMMARY")
    print(f"{'='*60}")
    print(f"  Constraint score: {total_met}/{total_constraints} ({overall:.1%})")
    print(f"  Code-only compliance: {code_only_count}/{len(CONSTRAINT_TESTS)}")
    print(f"  Avg code ratio: {avg_code_ratio:.1%}")

    return {
        "label": label,
        "overall_score": round(overall, 4),
        "total_met": total_met,
        "total_constraints": total_constraints,
        "code_only_count": code_only_count,
        "avg_code_ratio": round(avg_code_ratio, 4),
        "results": all_results,
    }


ft_eval = run_constraint_eval(generate_ft, "Fine-tuned Model")

# Free and load base
del model, generate_ft
gc.collect()
torch.cuda.empty_cache()

print("\nLoading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(base_model)

generate_base = make_generate(base_model, base_tokenizer)

base_eval = run_constraint_eval(generate_base, "Base Model")

# Final comparison
print("\n" + "=" * 60)
print("CONSTRAINT EVAL COMPARISON")
print("=" * 60)
print(f"{'Metric':<30} {'Fine-tuned':>12} {'Base':>12} {'Delta':>8}")
print("-" * 65)
print(f"{'Constraint score':<30} {ft_eval['overall_score']:>11.1%} {base_eval['overall_score']:>11.1%} {ft_eval['overall_score']-base_eval['overall_score']:>+7.1%}")
print(f"{'Code-only compliance':<30} {ft_eval['code_only_count']:>10}/{len(CONSTRAINT_TESTS)} {base_eval['code_only_count']:>10}/{len(CONSTRAINT_TESTS)} {ft_eval['code_only_count']-base_eval['code_only_count']:>+7}")
print(f"{'Avg code ratio':<30} {ft_eval['avg_code_ratio']:>11.1%} {base_eval['avg_code_ratio']:>11.1%} {ft_eval['avg_code_ratio']-base_eval['avg_code_ratio']:>+7.1%}")

print(f"\n{'Test':<35} {'FT':>8} {'Base':>8} {'Delta':>8}")
print("-" * 62)
for ft_r, base_r in zip(ft_eval["results"], base_eval["results"]):
    name = ft_r["name"][:32] + "..." if len(ft_r["name"]) > 32 else ft_r["name"]
    ft_s = f"{ft_r['met']}/{ft_r['total']}"
    base_s = f"{base_r['met']}/{base_r['total']}"
    delta = ft_r["met"] - base_r["met"]
    print(f"{name:<35} {ft_s:>8} {base_s:>8} {delta:>+7}")

# Save results
results_path = os.path.join(os.path.expanduser("~/output"), "eval_results.json")
eval_output = {
    "hard_eval": {"ft_passed": ft_passed, "base_passed": base_passed, "total": len(HARD_CHECKS)},
    "constraint_eval": {
        "ft": {"score": ft_eval["overall_score"], "code_only": ft_eval["code_only_count"], "code_ratio": ft_eval["avg_code_ratio"]},
        "base": {"score": base_eval["overall_score"], "code_only": base_eval["code_only_count"], "code_ratio": base_eval["avg_code_ratio"]},
    },
}
with open(results_path, "w") as f:
    json.dump(eval_output, f, indent=2)
print(f"\nResults saved to {results_path}")

del base_model, base_tokenizer, generate_base
gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print(f"Adapter: {OUTPUT_DIR}")
print(f"Results: {results_path}")
