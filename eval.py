#!/usr/bin/env python3
"""Step 3.5: Model evaluation — run after training, before serving.

Run on jarch. Evaluates model quality using held-out validation data
and hand-crafted sanity checks. Optionally compares against the base
model to measure fine-tuning lift.

Usage:
    python eval.py                          # Full eval (val sample + sanity checks)
    python eval.py --quick                  # Sanity checks only (~2 min)
    python eval.py --val-samples 50         # More val examples
    python eval.py --adapter outputs/lora_adapter  # Custom adapter path
    python eval.py --compare-base           # Also run base model for comparison
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_ADAPTER = BASE_DIR / "outputs" / "lora_adapter"
VAL_DATA = BASE_DIR / "data" / "training" / "val_full.jsonl"
RESULTS_PATH = BASE_DIR / "outputs" / "eval_results.json"

# ── Sanity check prompts ──────────────────────────────────────────────

SANITY_CHECKS = [
    {
        "prompt": "Write a Python function that computes cosine similarity between two vectors.",
        "keywords": ["def ", "np.", "dot", "norm"],
        "category": "machine_learning",
        "type": "code",
    },
    {
        "prompt": "Write code to train a random forest classifier and evaluate with cross-validation.",
        "keywords": ["RandomForest", "cross_val", "fit", "score"],
        "category": "machine_learning",
        "type": "code",
    },
    {
        "prompt": "Show me how to load a CSV with pandas, handle missing values, and plot a heatmap.",
        "keywords": ["read_csv", "fillna", "heatmap", "plt"],
        "category": "machine_learning",
        "type": "code",
    },
    {
        "prompt": "Write Python code to perform PCA on a dataset and plot explained variance.",
        "keywords": ["PCA", "fit", "explained_variance", "plot"],
        "category": "statistics",
        "type": "code",
    },
    {
        "prompt": "Implement a simple feedforward neural network for classification using PyTorch.",
        "keywords": ["nn.Module", "Linear", "forward", "def "],
        "category": "deep_learning",
        "type": "code",
    },
    {
        "prompt": "Write code to split data into train/test sets, scale features, and train a logistic regression.",
        "keywords": ["train_test_split", "Scaler", "LogisticRegression", "fit"],
        "category": "machine_learning",
        "type": "code",
    },
    {
        "prompt": "Show me how to create a confusion matrix and classification report in sklearn.",
        "keywords": ["confusion_matrix", "classification_report", "predict", "import"],
        "category": "machine_learning",
        "type": "code",
    },
    {
        "prompt": "Write Python code to perform k-means clustering and visualize the clusters.",
        "keywords": ["KMeans", "fit", "labels_", "plt"],
        "category": "machine_learning",
        "type": "code",
    },
    {
        "prompt": "Implement gradient descent from scratch in Python for linear regression.",
        "keywords": ["def ", "gradient", "learning_rate", "for "],
        "category": "machine_learning",
        "type": "code",
    },
    {
        "prompt": "Write code to build a data preprocessing pipeline with sklearn Pipeline.",
        "keywords": ["Pipeline", "Scaler", "fit", "transform"],
        "category": "machine_learning",
        "type": "code",
    },
]


# ── Failure mode detection ────────────────────────────────────────────

def detect_failures(prompt: str, response: str) -> list[str]:
    """Detect common failure modes in a generated response."""
    flags = []

    # Empty or near-empty
    if len(response.strip()) < 20:
        flags.append("empty_or_near_empty")

    # Prompt echoed back
    if prompt.strip().lower() in response.lower() and len(response) < len(prompt) * 2:
        flags.append("prompt_echo")

    # Repetition detection: find repeated phrases appearing many times
    # Use higher thresholds for code (common patterns like imports repeat naturally)
    words = response.split()
    if len(words) > 20:
        for ngram_len, threshold in [(5, 5), (8, 4), (12, 3)]:
            seen = defaultdict(int)
            for i in range(len(words) - ngram_len + 1):
                phrase = " ".join(words[i:i + ngram_len])
                seen[phrase] += 1
            max_repeats = max(seen.values()) if seen else 0
            if max_repeats >= threshold:
                flags.append(f"repetition_{ngram_len}gram")
                break

    # Special token leakage from fine-tuning
    if "<|im_end|>" in response or "<|im_start|>" in response:
        flags.append("special_token_leak")
    if "![" in response and response.count("![") >= 2:
        flags.append("image_artifact")

    # Off-topic: check if response shares any significant words with prompt
    prompt_words = set(w.lower() for w in re.findall(r'\w{4,}', prompt))
    response_words = set(w.lower() for w in re.findall(r'\w{4,}', response))
    if prompt_words and response_words:
        overlap = prompt_words & response_words
        if len(overlap) == 0 and len(response) > 50:
            flags.append("possibly_off_topic")

    return flags


# ── Metrics ───────────────────────────────────────────────────────────

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_keyword_recall(reference: str, response: str) -> float:
    """Extract key terms from reference and check recall in response."""
    # Extract significant words (4+ chars, not stopwords)
    stopwords = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "there", "will", "would", "could", "should", "about",
        "which", "when", "what", "where", "also", "more", "than", "into",
        "some", "other", "each", "your", "these", "those", "then", "here",
        "very", "just", "most", "such", "like", "over", "only", "being",
        "between", "after", "before", "during", "does", "make", "made",
    }
    ref_words = set(w.lower() for w in re.findall(r'\w{4,}', reference))
    ref_words -= stopwords
    if not ref_words:
        return 1.0

    response_lower = response.lower()
    found = sum(1 for w in ref_words if w in response_lower)
    return found / len(ref_words)


# ── Evaluation runners ────────────────────────────────────────────────

def run_validation_eval(model, tokenizer, max_tokens: int, n_samples: int) -> dict:
    """Evaluate on sampled validation examples."""
    from inference import generate_response as _generate

    print(f"\n{'='*60}")
    print(f"Validation Set Evaluation ({n_samples} samples)")
    print(f"{'='*60}")

    # Load and sample validation data
    examples = []
    with open(VAL_DATA) as f:
        for line in f:
            examples.append(json.loads(line))

    if n_samples < len(examples):
        # Stratified-ish sample: pick proportionally from each category
        by_cat = defaultdict(list)
        for ex in examples:
            by_cat[ex.get("category", "unknown")].append(ex)

        sampled = []
        for cat, cat_examples in by_cat.items():
            n = max(1, round(n_samples * len(cat_examples) / len(examples)))
            sampled.extend(random.sample(cat_examples, min(n, len(cat_examples))))

        # Trim or pad to target
        if len(sampled) > n_samples:
            sampled = random.sample(sampled, n_samples)
        examples = sampled

    results_by_cat = defaultdict(list)
    all_results = []

    for i, ex in enumerate(examples):
        convs = ex["conversations"]
        prompt = next(c["value"] for c in convs if c["from"] == "human")
        reference = next(c["value"] for c in convs if c["from"] == "gpt")
        category = ex.get("category", "unknown")

        print(f"  [{i+1}/{len(examples)}] {category}: {prompt[:60]}...", end=" ", flush=True)
        start = time.time()
        response = _generate(model, tokenizer, prompt, max_tokens=max_tokens, temperature=0.1)
        elapsed = time.time() - start

        rouge_l = compute_rouge_l(reference, response)
        kw_recall = compute_keyword_recall(reference, response)
        failures = detect_failures(prompt, response)

        result = {
            "prompt": prompt[:200],
            "category": category,
            "response_len_chars": len(response),
            "response_len_tokens": len(response.split()),
            "rouge_l": round(rouge_l, 4),
            "keyword_recall": round(kw_recall, 4),
            "failures": failures,
            "time_s": round(elapsed, 1),
        }
        all_results.append(result)
        results_by_cat[category].append(result)

        status = "OK" if not failures else f"FLAGS: {failures}"
        print(f"ROUGE-L={rouge_l:.3f} kw={kw_recall:.3f} ({elapsed:.1f}s) {status}")

    # Aggregate by category
    print(f"\n{'─'*70}")
    print(f"{'Category':<20} {'Count':>5} {'Avg ROUGE-L':>12} {'Avg KW Recall':>14} {'Avg Len':>8} {'Flags':>6}")
    print(f"{'─'*70}")

    category_summary = {}
    for cat in sorted(results_by_cat.keys()):
        cat_results = results_by_cat[cat]
        n = len(cat_results)
        avg_rouge = sum(r["rouge_l"] for r in cat_results) / n
        avg_kw = sum(r["keyword_recall"] for r in cat_results) / n
        avg_len = sum(r["response_len_chars"] for r in cat_results) / n
        n_flags = sum(1 for r in cat_results if r["failures"])

        print(f"{cat:<20} {n:>5} {avg_rouge:>12.3f} {avg_kw:>14.3f} {avg_len:>8.0f} {n_flags:>6}")
        category_summary[cat] = {
            "count": n,
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_keyword_recall": round(avg_kw, 4),
            "avg_response_len": round(avg_len),
            "flagged": n_flags,
        }

    # Overall
    total = len(all_results)
    overall_rouge = sum(r["rouge_l"] for r in all_results) / total
    overall_kw = sum(r["keyword_recall"] for r in all_results) / total
    overall_len = sum(r["response_len_chars"] for r in all_results) / total
    total_flags = sum(1 for r in all_results if r["failures"])

    print(f"{'─'*70}")
    print(f"{'OVERALL':<20} {total:>5} {overall_rouge:>12.3f} {overall_kw:>14.3f} {overall_len:>8.0f} {total_flags:>6}")

    return {
        "category_summary": category_summary,
        "details": all_results,
        "overall": {
            "count": total,
            "avg_rouge_l": round(overall_rouge, 4),
            "avg_keyword_recall": round(overall_kw, 4),
            "avg_response_len": round(overall_len),
            "flagged": total_flags,
        },
    }


def run_sanity_checks(model, tokenizer, max_tokens: int) -> dict:
    """Run hand-crafted sanity check prompts."""
    from inference import generate_response as _generate

    print(f"\n{'='*60}")
    print("Sanity Checks")
    print(f"{'='*60}")

    results = []
    pass_count = 0

    for i, check in enumerate(SANITY_CHECKS):
        prompt = check["prompt"]
        expected_kw = check["keywords"]

        print(f"\n  [{i+1}/{len(SANITY_CHECKS)}] ({check['type']}/{check['category']})")
        print(f"  Prompt: {prompt}")

        start = time.time()
        response = _generate(model, tokenizer, prompt, max_tokens=max_tokens, temperature=0.1)
        elapsed = time.time() - start

        # Check keywords
        response_lower = response.lower()
        kw_hits = {kw: kw.lower() in response_lower for kw in expected_kw}
        kw_score = sum(kw_hits.values()) / len(kw_hits) if kw_hits else 0

        # Detect failures
        failures = detect_failures(prompt, response)

        # Pass if: >=50% keywords hit, no critical failures, response is substantial
        passed = kw_score >= 0.5 and not failures and len(response) > 50

        result = {
            "prompt": prompt,
            "type": check["type"],
            "category": check["category"],
            "passed": passed,
            "keyword_hits": kw_hits,
            "keyword_score": round(kw_score, 3),
            "response_len": len(response),
            "response_preview": response[:300],
            "failures": failures,
            "time_s": round(elapsed, 1),
        }
        results.append(result)
        if passed:
            pass_count += 1

        status = "PASS" if passed else "FAIL"
        print(f"  Result: {status} | Keywords: {sum(kw_hits.values())}/{len(kw_hits)} | Len: {len(response)} | {elapsed:.1f}s")
        if not passed:
            missing = [k for k, v in kw_hits.items() if not v]
            if missing:
                print(f"  Missing keywords: {missing}")
            if failures:
                print(f"  Failure flags: {failures}")
            print(f"  Response preview: {response[:200]}...")

    print(f"\n{'─'*60}")
    print(f"Sanity Checks: {pass_count}/{len(SANITY_CHECKS)} passed")

    return {
        "passed": pass_count,
        "total": len(SANITY_CHECKS),
        "details": results,
    }


# ── Main ──────────────────────────────────────────────────────────────

def load_base_model(seq_len: int = 512):
    """Load the un-finetuned base model for comparison."""
    from unsloth import FastLanguageModel

    BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    print(f"\nLoading base model {BASE_MODEL} for comparison...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Base model loaded!")
    return model, tokenizer


def print_comparison(label: str, finetuned: dict, base: dict):
    """Print a side-by-side comparison table."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {label}")
    print(f"{'='*60}")

    if "overall" in finetuned and "overall" in base:
        # Validation comparison
        ft = finetuned["overall"]
        bs = base["overall"]
        print(f"{'Metric':<20} {'Fine-tuned':>12} {'Base':>12} {'Delta':>12}")
        print(f"{'─'*56}")
        for key, name in [
            ("avg_rouge_l", "ROUGE-L"),
            ("avg_keyword_recall", "KW Recall"),
            ("avg_response_len", "Avg Length"),
            ("flagged", "Flagged"),
        ]:
            ft_val = ft[key]
            bs_val = bs[key]
            delta = ft_val - bs_val
            sign = "+" if delta > 0 else ""
            fmt = ".3f" if isinstance(ft_val, float) else "d"
            print(f"{name:<20} {ft_val:>12{fmt}} {bs_val:>12{fmt}} {sign}{delta:>11{fmt}}")
    else:
        # Sanity check comparison
        ft_pass = finetuned["passed"]
        bs_pass = base["passed"]
        total = finetuned["total"]
        print(f"  Fine-tuned: {ft_pass}/{total} passed")
        print(f"  Base model: {bs_pass}/{total} passed")
        print(f"  Lift: {ft_pass - bs_pass:+d}")

        # Per-check comparison
        print(f"\n{'Prompt':<55} {'FT':>4} {'Base':>5}")
        print(f"{'─'*66}")
        for ft_d, bs_d in zip(finetuned["details"], base["details"]):
            ft_status = "PASS" if ft_d["passed"] else "FAIL"
            bs_status = "PASS" if bs_d["passed"] else "FAIL"
            prompt_short = ft_d["prompt"][:53]
            print(f"{prompt_short:<55} {ft_status:>4} {bs_status:>5}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--quick", action="store_true",
                        help="Sanity checks only (skip validation set)")
    parser.add_argument("--val-samples", type=int, default=20,
                        help="Number of validation examples to sample (default: 20)")
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER),
                        help="Path to LoRA adapter")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max new tokens (defaults to seq_len from training config)")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate base model for A/B comparison")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load fine-tuned model (reusing inference.py pattern)
    from inference import load_model, generate_response
    model, tokenizer, seq_len = load_model(args.adapter)
    max_tokens = args.max_tokens if args.max_tokens else seq_len

    random.seed(42)
    all_results = {"adapter": args.adapter, "mode": "quick" if args.quick else "full"}
    start_time = time.time()

    # ── Fine-tuned model evaluation ──
    print("\n" + "="*60)
    print("EVALUATING: Fine-tuned model")
    print("="*60)

    sanity = run_sanity_checks(model, tokenizer, max_tokens)
    all_results["sanity_checks"] = sanity

    val = None
    if not args.quick:
        val = run_validation_eval(model, tokenizer, max_tokens, args.val_samples)
        all_results["validation"] = val

    # ── Base model comparison ──
    if args.compare_base:
        # Free fine-tuned model memory before loading base
        import torch
        del model
        torch.cuda.empty_cache()

        base_model, base_tokenizer = load_base_model(seq_len)

        print("\n" + "="*60)
        print("EVALUATING: Base model (no fine-tuning)")
        print("="*60)

        # Temporarily monkey-patch generate_response to use base model's tokenizer
        import inference
        _orig_generate = inference.generate_response

        random.seed(42)  # Reset seed for identical sampling
        base_sanity = run_sanity_checks(base_model, base_tokenizer, max_tokens)
        all_results["base_sanity_checks"] = base_sanity

        base_val = None
        if not args.quick:
            random.seed(42)
            base_val = run_validation_eval(base_model, base_tokenizer, max_tokens, args.val_samples)
            all_results["base_validation"] = base_val

        # Print comparisons
        print_comparison("Sanity Checks", sanity, base_sanity)
        if val and base_val:
            print_comparison("Validation Set", val, base_val)

        # Free base model, reload fine-tuned for any downstream use
        del base_model, base_tokenizer
        torch.cuda.empty_cache()

    elapsed_total = time.time() - start_time
    all_results["total_time_s"] = round(elapsed_total, 1)

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Sanity checks: {sanity['passed']}/{sanity['total']} passed")
    if "validation" in all_results:
        v = all_results["validation"]["overall"]
        print(f"Validation:    ROUGE-L={v['avg_rouge_l']:.3f}  KW-recall={v['avg_keyword_recall']:.3f}  Avg-len={v['avg_response_len']}  Flagged={v['flagged']}/{v['count']}")
    if args.compare_base:
        bs = all_results["base_sanity_checks"]
        print(f"\nBase sanity:   {bs['passed']}/{bs['total']} passed")
        if "base_validation" in all_results:
            bv = all_results["base_validation"]["overall"]
            print(f"Base val:      ROUGE-L={bv['avg_rouge_l']:.3f}  KW-recall={bv['avg_keyword_recall']:.3f}  Avg-len={bv['avg_response_len']}  Flagged={bv['flagged']}/{bv['count']}")


if __name__ == "__main__":
    main()
