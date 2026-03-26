#!/usr/bin/env python3
"""Constraint-based eval — measures what fine-tuning actually changes.

Tests multi-constraint instruction following, style compliance,
and code quality. This is where fine-tuned models beat base models.

Cell 0: Install deps + restart kernel
Cell 1: Paste everything below
"""

# ── Cleanup any prior models from earlier cells ─────────────────
import gc
import torch

for name in list(globals()):
    obj = globals()[name]
    if hasattr(obj, 'parameters') and callable(getattr(obj, 'parameters', None)):
        try:
            del globals()[name]
        except Exception:
            pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Config ───────────────────────────────────────────────────────
import os
os.environ["UNSLOTH_USE_FUSED_CE_LOSS"] = "0"
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"

from unsloth import FastLanguageModel
from peft import PeftModel
import time
import re

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
FastLanguageModel.for_inference(model)
print("Ready!\n")


# ── Generate helper ──────────────────────────────────────────────
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
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()
    return gen


generate_ft = make_generate(model, tokenizer)


# ── Constraint checks ───────────────────────────────────────────
# Each test has a prompt with specific constraints.
# Constraints are either regex patterns or callable checks.
# Score = fraction of constraints met.

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


# ── Style compliance checks ──────────────────────────────────────
def check_style(response):
    """Check if response follows the system prompt style."""
    # Extract text outside code blocks
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


# ── Eval runner ──────────────────────────────────────────────────
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

        # Check constraints
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

        # Style check
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

    # Summary
    overall = total_met / total_constraints
    avg_code_ratio = sum(s["code_ratio"] for s in style_scores) / len(style_scores)
    code_only_count = sum(1 for s in style_scores if s["is_code_only"])

    print(f"\n{'='*60}")
    print(f"{label} SUMMARY")
    print(f"{'='*60}")
    print(f"  Constraint score: {total_met}/{total_constraints} ({overall:.1%})")
    print(f"  Code-only compliance: {code_only_count}/{len(CONSTRAINT_TESTS)}")
    print(f"  Avg code ratio: {avg_code_ratio:.1%}")
    print()

    return {
        "label": label,
        "overall_score": round(overall, 4),
        "total_met": total_met,
        "total_constraints": total_constraints,
        "code_only_count": code_only_count,
        "avg_code_ratio": round(avg_code_ratio, 4),
        "results": all_results,
    }


# ── Run fine-tuned eval ──────────────────────────────────────────
ft_eval = run_constraint_eval(generate_ft, "Fine-tuned Model")

# ── Free memory and load base ────────────────────────────────────
del model
gc.collect()
torch.cuda.empty_cache()

print("\nLoading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,
    load_in_4bit=True,
    gpu_memory_utilization=0.85,
)
FastLanguageModel.for_inference(base_model)

generate_base = make_generate(base_model, base_tokenizer)

# ── Run base eval ────────────────────────────────────────────────
base_eval = run_constraint_eval(generate_base, "Base Model")


# ── Final comparison ─────────────────────────────────────────────
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

# Final verdict
ft_pct = ft_eval["overall_score"] * 100
base_pct = base_eval["overall_score"] * 100
delta = ft_pct - base_pct
print(f"\n{'='*60}")
if delta > 0:
    print(f"RESULT: Fine-tuned model wins by {delta:.1f}% on constraint compliance")
elif delta < 0:
    print(f"RESULT: Base model wins by {-delta:.1f}% on constraint compliance")
else:
    print(f"RESULT: Tied on constraint compliance")
print(f"Fine-tuned: {ft_pct:.1f}% | Base: {base_pct:.1f}%")
print(f"{'='*60}")
