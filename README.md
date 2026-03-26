# Data Science Assistant — Fine-tuned LLM

A domain-specific language model fine-tuned on curated data science content. Built end-to-end: data collection, cleaning, training, evaluation, and serving.

## What it does

Fine-tunes **Qwen2.5-1.5B-Instruct** using QLoRA on ~1,500 curated examples from the "Daily Dose of Data Science" newsletter. The resulting model answers questions about ML, deep learning, LLMs, RAG, AI agents, MLOps, and more — with practical explanations and code.

## Architecture

```
Gmail API → fetch → clean → format → train (QLoRA) → eval → serve
  (Mac)      (Mac)   (Mac)   (Mac)    (jarch GPU)   (jarch) (jarch)
```

| Component | Script | What it does |
|-----------|--------|-------------|
| Fetch | `fetch_emails.py` | Pull emails from Gmail, scrape linked articles, download images |
| Clean | `clean_data.py` | Normalize text, detect sections/categories, filter noise |
| Format | `format_training_data.py` | Generate Q&A pairs (7 strategies), quality filter, stratified split |
| Train | `train.py` | QLoRA fine-tuning on GTX 1660 Ti (6GB VRAM) |
| Eval | `eval.py` | Sanity checks, ROUGE-L, keyword recall, failure detection, base model comparison |
| Serve | `inference.py` | Interactive CLI or FastAPI HTTP server |
| Update | `update.py` | Daily cron pipeline: fetch new emails, retrain weekly |

## Training details

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| Method | QLoRA (r=8, alpha=16, dropout=0.05) |
| Target modules | All attention + MLP layers |
| Training examples | ~1,500 (stratified 90/10 split) |
| Effective batch size | 16 (batch=1, grad_accum=16) |
| Learning rate | 2e-4 (cosine schedule, 5% warmup) |
| Epochs | 2 |
| Sequence length | 512 |
| Hardware | GTX 1660 Ti (6GB VRAM), CUDA 12.1 |
| Label masking | Loss on assistant tokens only |

## Training data

Sourced from 286 newsletter issues covering:

| Category | Examples | Share |
|----------|----------|-------|
| Agents | 879 | 52% |
| LLMs | 352 | 21% |
| RAG | 139 | 8% |
| Machine Learning | 101 | 6% |
| MCP | 49 | 3% |
| Deep Learning | 49 | 3% |
| MLOps | 40 | 2% |
| Statistics | 36 | 2% |
| Prompt Engineering | 24 | 1% |
| Graph ML | 16 | 1% |

Seven generation strategies: subject overviews, section Q&A, code-focused pairs, how-to guides, comparisons, article summaries, and multi-turn conversations.

Quality filters remove: short/empty responses, URL-heavy content, newsletter boilerplate (headers, sponsor blocks, promotional CTAs), duplicate examples.

## Evaluation

```bash
python eval.py                  # Full eval: sanity checks + validation sample
python eval.py --quick          # Sanity checks only (~2 min)
python eval.py --compare-base   # Compare fine-tuned vs base model
python eval.py --val-samples 50 # Evaluate more validation examples
```

Metrics:
- **ROUGE-L** — lexical overlap with reference answers
- **Keyword recall** — key terms from reference present in output
- **Failure detection** — empty responses, repetition, prompt echo, off-topic
- **Sanity checks** — 10 hand-crafted prompts across explain/code/compare/how-to types

## Quick start

```bash
# On jarch (GPU server)
pip install -r requirements-jarch.txt

# Train
python train.py                     # ~6.5h on 1660 Ti
python train.py --test              # Quick test with 10 examples

# Evaluate
python eval.py --quick              # Sanity checks
python eval.py --compare-base       # Full eval with base model comparison

# Serve
python inference.py --mode cli      # Interactive CLI
python inference.py --mode server   # FastAPI on port 8000
```

```bash
# From Mac (access server)
ssh -L 8000:localhost:8000 jarch    # Tunnel for API access
curl -X POST localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain how RAG works"}'
```

## Project structure

```
├── fetch_emails.py             # Step 0: Gmail data collection
├── clean_data.py               # Step 1: Text cleaning & structuring
├── format_training_data.py     # Step 2: Training pair generation
├── train.py                    # Step 3: QLoRA fine-tuning
├── eval.py                     # Step 3.5: Model evaluation
├── inference.py                # Step 4: CLI + API serving
├── update.py                   # Step 5: Daily pipeline orchestrator
├── requirements.txt            # Mac dependencies
├── requirements-jarch.txt      # jarch (GPU server) dependencies
├── data/
│   ├── emails/                 # Raw email JSONs
│   ├── clean_emails/           # Cleaned & structured emails
│   ├── training/               # Generated training data (JSONL)
│   └── images/                 # Downloaded article images
└── outputs/
    ├── lora_adapter/           # Trained LoRA weights
    └── eval_results.json       # Evaluation results
```

## Known limitations

- **Category imbalance**: Agents (52%) and LLMs (21%) dominate training data. Model will be weaker on underrepresented categories (NLP, Graph ML, Statistics).
- **Sequence length 512**: Responses are capped at training sequence length. Complex topics may get truncated.
- **Single GPU**: Training on 6GB VRAM constrains model size and batch size.
- **Newsletter source**: Training data reflects one newsletter's style, coverage, and potential biases.
