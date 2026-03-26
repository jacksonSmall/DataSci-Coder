#!/usr/bin/env python3
"""Download and filter public data science instruction datasets from HuggingFace."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert data science coding assistant. "
    "Respond ONLY with clean, runnable Python code. "
    "Use inline comments for explanation. "
    "No text outside code blocks."
)

DATASETS = [
    {
        "name": "iamtarun/python_code_instructions_18k_alpaca",
        "instruction_key": "instruction",
        "output_key": "output",
    },
    {
        "name": "sahil2801/CodeAlpaca-20k",
        "instruction_key": "instruction",
        "output_key": "output",
    },
    {
        "name": "nickrosh/Evol-Instruct-Code-80k-v1",
        "instruction_key": "instruction",
        "output_key": "output",
    },
    {
        "name": "TokenBender/code_instructions_122k_alpaca_style",
        "instruction_key": "instruction",
        "output_key": "output",
    },
    {
        "name": "theblackcat102/evol-codealpaca-v1",
        "instruction_key": "instruction",
        "output_key": "output",
    },
    {
        "name": "mlabonne/Evol-Instruct-Python-26k",
        "instruction_key": "instruction",
        "output_key": "output",
    },
]

# Keywords that suggest data-science relevance (case-insensitive matching)
DS_KEYWORDS = [
    "pandas", "numpy", "sklearn", "torch", "matplotlib", "seaborn",
    "scipy", "statsmodels", "tensorflow", "keras", "xgboost", "lightgbm",
    "data", "model", "train", "predict", "fit", "plot", "dataframe", "array",
    "regression", "classification", "clustering", "pipeline", "feature",
    "accuracy", "precision", "recall", "f1", "auc", "roc", "confusion_matrix",
    "cross_val", "GridSearchCV", "RandomizedSearchCV", "hyperparameter",
    "PCA", "t-SNE", "UMAP", "dimensionality", "embedding",
    "neural", "layer", "optimizer", "gradient", "backprop", "epoch",
    "tokenizer", "transformer", "attention", "encoder", "decoder",
    "bayesian", "hypothesis", "p-value", "confidence", "bootstrap",
    "time_series", "arima", "prophet", "forecast", "seasonal",
    "nlp", "text", "sentiment", "tfidf", "word2vec", "bert",
    "image", "cnn", "convolution", "resnet", "augmentation",
    "impute", "missing", "outlier", "anomaly", "normalization",
]

DS_PATTERN = re.compile("|".join(re.escape(kw) for kw in DS_KEYWORDS), re.IGNORECASE)


def looks_like_python(text: str) -> bool:
    """Check if the response contains Python code markers."""
    return bool(
        re.search(r"import\s+\w", text)
        or re.search(r"def\s+\w", text)
        or re.search(r"class\s+\w", text)
    )


def looks_like_code(text: str) -> bool:
    """Check if the text looks like code (has fences or indentation patterns)."""
    if "```" in text:
        return True
    # Count lines that start with whitespace (indented code) or common code patterns
    lines = text.strip().split("\n")
    code_lines = sum(
        1 for line in lines
        if line.startswith("    ") or line.startswith("\t")
        or line.strip().startswith("import ")
        or line.strip().startswith("from ")
        or line.strip().startswith("def ")
        or line.strip().startswith("class ")
        or line.strip().startswith("#")
        or "=" in line
    )
    return code_lines >= max(2, len(lines) * 0.3)


def is_ds_related(instruction: str, response: str) -> bool:
    """Check if instruction or response is data-science related."""
    combined = instruction + " " + response
    return bool(DS_PATTERN.search(combined))


def classify_category(instruction: str, response: str) -> str:
    """Assign a rough category based on keywords."""
    combined = (instruction + " " + response).lower()
    if any(kw in combined for kw in ("torch", "tensorflow", "keras", "neural", "deep learning", "nn.module", "cnn", "rnn", "lstm", "transformer", "autoencoder", "gan")):
        return "deep_learning"
    if any(kw in combined for kw in ("nlp", "tokeniz", "sentiment", "tfidf", "word2vec", "bert", "text classification", "named entity")):
        return "nlp"
    if any(kw in combined for kw in ("time_series", "arima", "prophet", "forecast", "seasonal", "autocorrelation")):
        return "time_series"
    if any(kw in combined for kw in ("scipy", "statsmodels", "statistic", "hypothesis", "p-value", "confidence interval", "bayesian", "anova", "chi-square")):
        return "statistics"
    if any(kw in combined for kw in ("matplotlib", "seaborn", "plot", "chart", "graph", "visual", "heatmap", "subplot")):
        return "visualization"
    if any(kw in combined for kw in ("pandas", "dataframe", "csv", "merge", "groupby", "pivot")):
        return "data_wrangling"
    if any(kw in combined for kw in ("sklearn", "train", "predict", "fit", "model", "xgboost", "lightgbm", "classification", "clustering")):
        return "machine_learning"
    if any(kw in combined for kw in ("numpy", "array", "matrix", "linalg")):
        return "numerical_computing"
    return "general_ds"


def filter_example(instruction: str, response: str) -> bool:
    """Apply all filter criteria. Returns True if example should be kept."""
    if not instruction or not response:
        return False
    resp_len = len(response)
    if resp_len < 80 or resp_len > 6000:
        return False
    if not looks_like_python(response):
        return False
    if not is_ds_related(instruction, response):
        return False
    if not looks_like_code(response):
        return False
    return True


def format_example(instruction: str, response: str, dataset_name: str) -> dict:
    """Convert to our training format."""
    short_name = dataset_name.split("/")[-1]
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": instruction.strip()},
            {"from": "gpt", "value": response.strip()},
        ],
        "category": classify_category(instruction, response),
        "source_id": f"public:{short_name}",
    }


def download_and_filter(max_per_dataset: int, dry_run: bool) -> list[dict]:
    """Download datasets, filter, and return formatted examples."""
    from datasets import load_dataset

    all_examples = []

    for ds_info in DATASETS:
        name = ds_info["name"]
        instr_key = ds_info["instruction_key"]
        out_key = ds_info["output_key"]

        print(f"\n--- Loading {name} ---")
        try:
            ds = load_dataset(name, split="train")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            continue

        print(f"  Total rows: {len(ds)}")

        # Check available columns
        cols = ds.column_names
        print(f"  Columns: {cols}")
        if instr_key not in cols or out_key not in cols:
            print(f"  Missing expected columns, skipping.")
            continue

        kept = 0
        skipped = 0
        for row in ds:
            if kept >= max_per_dataset:
                break
            instruction = row.get(instr_key, "") or ""
            # Some datasets have an 'input' field that extends the instruction
            extra_input = row.get("input", "") or ""
            if extra_input:
                instruction = instruction.strip() + "\n\n" + extra_input.strip()
            response = row.get(out_key, "") or ""

            if filter_example(instruction, response):
                example = format_example(instruction, response, name)
                all_examples.append(example)
                kept += 1
            else:
                skipped += 1

        print(f"  Kept: {kept}, Skipped: {skipped}")

    return all_examples


def main():
    parser = argparse.ArgumentParser(description="Download public DS instruction data")
    parser.add_argument("--max-per-dataset", type=int, default=3000,
                        help="Max examples to keep per dataset (default: 3000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Filter and report counts but don't write output file")
    args = parser.parse_args()

    examples = download_and_filter(args.max_per_dataset, args.dry_run)

    print(f"\n=== Total examples collected: {len(examples)} ===")

    # Category breakdown
    cats = {}
    for ex in examples:
        c = ex["category"]
        cats[c] = cats.get(c, 0) + 1
    print("\nCategory breakdown:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")

    # Source breakdown
    sources = {}
    for ex in examples:
        s = ex["source_id"]
        sources[s] = sources.get(s, 0) + 1
    print("\nSource breakdown:")
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n}")

    if args.dry_run:
        print("\n[DRY RUN] No file written.")
        return

    out_dir = Path(__file__).parent / "data" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "public_examples.jsonl"

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
