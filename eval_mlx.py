#!/usr/bin/env python3
"""Model evaluation using MLX on Apple Silicon.

Usage:
    python eval_mlx.py                          # Full eval
    python eval_mlx.py --quick                  # Sanity checks only
    python eval_mlx.py --compare-base           # Compare with base model
    python eval_mlx.py --val-samples 50         # More val examples
"""

import argparse
import json
import random
import time
from pathlib import Path

# Reuse all the evaluation logic from eval.py
from eval import (
    SANITY_CHECKS,
    detect_failures,
    compute_keyword_recall,
    VAL_DATA,
)
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DEFAULT_ADAPTER = BASE_DIR / "outputs" / "mlx_coder_adapter"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
RESULTS_PATH = BASE_DIR / "outputs" / "eval_mlx_results.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model (MLX)")
    parser.add_argument("--quick", action="store_true",
                        help="Sanity checks only")
    parser.add_argument("--val-samples", type=int, default=20)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--compare-base", action="store_true")
    return parser.parse_args()


def run_sanity_checks(model, tokenizer, max_tokens: int) -> dict:
    """Run sanity check prompts."""
    from inference_mlx import generate_response

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
        response = generate_response(model, tokenizer, prompt, max_tokens=max_tokens, temperature=0.1)
        elapsed = time.time() - start

        response_lower = response.lower()
        kw_hits = {kw: kw.lower() in response_lower for kw in expected_kw}
        kw_score = sum(kw_hits.values()) / len(kw_hits) if kw_hits else 0

        failures = detect_failures(prompt, response)
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

    return {"passed": pass_count, "total": len(SANITY_CHECKS), "details": results}


def run_validation_eval(model, tokenizer, max_tokens: int, n_samples: int) -> dict:
    """Evaluate on validation set."""
    from inference_mlx import generate_response

    print(f"\n{'='*60}")
    print(f"Validation Set Evaluation ({n_samples} samples)")
    print(f"{'='*60}")

    examples = []
    with open(VAL_DATA) as f:
        for line in f:
            examples.append(json.loads(line))

    if n_samples < len(examples):
        by_cat = defaultdict(list)
        for ex in examples:
            by_cat[ex.get("category", "unknown")].append(ex)

        sampled = []
        for cat, cat_examples in by_cat.items():
            n = max(1, round(n_samples * len(cat_examples) / len(examples)))
            sampled.extend(random.sample(cat_examples, min(n, len(cat_examples))))

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
        response = generate_response(model, tokenizer, prompt, max_tokens=max_tokens, temperature=0.1)
        elapsed = time.time() - start

        kw_recall = compute_keyword_recall(reference, response)
        failures = detect_failures(prompt, response)

        result = {
            "prompt": prompt[:200],
            "category": category,
            "response_len_chars": len(response),
            "keyword_recall": round(kw_recall, 4),
            "failures": failures,
            "time_s": round(elapsed, 1),
        }
        all_results.append(result)
        results_by_cat[category].append(result)

        status = "OK" if not failures else f"FLAGS: {failures}"
        print(f"kw={kw_recall:.3f} ({elapsed:.1f}s) {status}")

    # Aggregate
    print(f"\n{'─'*70}")
    print(f"{'Category':<20} {'Count':>5} {'Avg KW Recall':>14} {'Avg Len':>8} {'Flags':>6}")
    print(f"{'─'*70}")

    category_summary = {}
    for cat in sorted(results_by_cat.keys()):
        cat_results = results_by_cat[cat]
        n = len(cat_results)
        avg_kw = sum(r["keyword_recall"] for r in cat_results) / n
        avg_len = sum(r["response_len_chars"] for r in cat_results) / n
        n_flags = sum(1 for r in cat_results if r["failures"])

        print(f"{cat:<20} {n:>5} {avg_kw:>14.3f} {avg_len:>8.0f} {n_flags:>6}")
        category_summary[cat] = {
            "count": n,
            "avg_keyword_recall": round(avg_kw, 4),
            "avg_response_len": round(avg_len),
            "flagged": n_flags,
        }

    total = len(all_results)
    overall_kw = sum(r["keyword_recall"] for r in all_results) / total
    overall_len = sum(r["response_len_chars"] for r in all_results) / total
    total_flags = sum(1 for r in all_results if r["failures"])

    print(f"{'─'*70}")
    print(f"{'OVERALL':<20} {total:>5} {overall_kw:>14.3f} {overall_len:>8.0f} {total_flags:>6}")

    return {
        "category_summary": category_summary,
        "details": all_results,
        "overall": {
            "count": total,
            "avg_keyword_recall": round(overall_kw, 4),
            "avg_response_len": round(overall_len),
            "flagged": total_flags,
        },
    }


def main():
    args = parse_args()

    from inference_mlx import load_model
    model, tokenizer = load_model(args.model, args.adapter)

    random.seed(42)
    all_results = {"adapter": args.adapter, "mode": "quick" if args.quick else "full"}
    start_time = time.time()

    # Fine-tuned eval
    print("\n" + "="*60)
    print("EVALUATING: Fine-tuned model (MLX)")
    print("="*60)

    sanity = run_sanity_checks(model, tokenizer, args.max_tokens)
    all_results["sanity_checks"] = sanity

    val = None
    if not args.quick:
        val = run_validation_eval(model, tokenizer, args.max_tokens, args.val_samples)
        all_results["validation"] = val

    # Base model comparison
    if args.compare_base:
        import mlx.core as mx
        del model
        mx.metal.clear_cache()

        print("\n" + "="*60)
        print("EVALUATING: Base model (no fine-tuning)")
        print("="*60)

        base_model, base_tokenizer = load_model(args.model, adapter_path=None)

        random.seed(42)
        base_sanity = run_sanity_checks(base_model, base_tokenizer, args.max_tokens)
        all_results["base_sanity_checks"] = base_sanity

        if not args.quick:
            random.seed(42)
            base_val = run_validation_eval(base_model, base_tokenizer, args.max_tokens, args.val_samples)
            all_results["base_validation"] = base_val

        # Comparison summary
        print(f"\n{'='*60}")
        print("COMPARISON: Sanity Checks")
        print(f"{'='*60}")
        print(f"  Fine-tuned: {sanity['passed']}/{sanity['total']} passed")
        print(f"  Base model: {base_sanity['passed']}/{base_sanity['total']} passed")
        print(f"  Lift: {sanity['passed'] - base_sanity['passed']:+d}")

    elapsed_total = time.time() - start_time
    all_results["total_time_s"] = round(elapsed_total, 1)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Sanity checks: {sanity['passed']}/{sanity['total']} passed")
    if val:
        v = val["overall"]
        print(f"Validation:    KW-recall={v['avg_keyword_recall']:.3f}  Avg-len={v['avg_response_len']}  Flagged={v['flagged']}/{v['count']}")


if __name__ == "__main__":
    main()
