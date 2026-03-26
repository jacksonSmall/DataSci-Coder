# DataSci-Coder — Fine-Tuned LLM for Data Science Code Generation

A QLoRA fine-tuned [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) optimized for data science code generation. Outputs clean, runnable Python code with zero explanatory text.

Model weights on HuggingFace: [jsmall12/DataSci-Coder-14B-LoRA](https://huggingface.co/jsmall12/DataSci-Coder-14B-LoRA)

## Results

| Metric | Fine-tuned | Base Model | Delta |
|--------|:---:|:---:|:---:|
| Hard Eval (12 complex tasks) | 12/12 | 12/12 | Tie |
| Constraint Compliance | 93.3% | 91.4% | **+1.9%** |
| Code-Only Compliance | 10/10 | 6/10 | **+67%** |
| Code Ratio | 100% | 87.9% | **+12.1%** |

The fine-tuned model's biggest win is instruction compliance: when told "respond with code only," the base model adds explanatory text 40% of the time. The fine-tuned model follows the instruction 100% of the time.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit` |
| Method | QLoRA (4-bit quantization) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Targets | q/k/v/o_proj, gate/up/down_proj |
| Trainable Parameters | 68.8M / 14.8B (0.46%) |
| Training Examples | 10,795 |
| Epochs | 1 |
| Final Loss | 0.5933 |
| Training Time | 1.9 hours on NVIDIA L40S |
| Precision | bfloat16 |
| Optimizer | Paged AdamW 8-bit |
| Learning Rate | 3e-5 (cosine schedule) |
| Effective Batch Size | 16 (1 x 16 grad accum) |

## Training Data

10,795 curated data science instruction-response pairs from:
- 6 public HuggingFace datasets (CodeAlpaca, Evol-Instruct, etc.)
- University coursework (statistics, ML, deep learning)
- Hand-curated examples

All examples filtered for Python code quality, data science relevance, and response length. Categories: machine learning, deep learning, statistics, data wrangling, visualization, NLP, time series, numerical computing.

## Evaluation

### Hard Eval — 12 Complex Tasks

All 12 tasks produced correct, complete, runnable implementations:

| Category | Tasks | Score |
|----------|-------|:---:|
| Statistics | Bayesian A/B testing, Kaplan-Meier survival analysis, time series CV + ARIMA, VIF + Ridge/Lasso/ElasticNet | 4/4 |
| Machine Learning | Stacking ensemble, SHAP importance, Isolation Forest, TF-IDF + SVM pipeline | 4/4 |
| Deep Learning | LR scheduler (warmup + cosine), BiLSTM + attention, VAE, GAN | 4/4 |

### Constraint Eval — 10 Multi-Constraint Tests

| Test | FT | Base | Delta |
|------|:---:|:---:|:---:|
| C01 Multi-step data cleaning | 8/8 | 8/8 | 0 |
| C02 Complete ML pipeline | 12/12 | 12/12 | 0 |
| C03 Statistical hypothesis test | 9/9 | 7/9 | **+2** |
| C04 PyTorch architecture | 9/9 | 7/9 | **+2** |
| C05 EDA visualizations | 11/12 | 10/12 | **+1** |
| C06 Cross-validated pipeline | 12/12 | 12/12 | 0 |
| C07 Time series ARIMA | 9/10 | 10/10 | -1 |
| C08 DL training function | 8/10 | 8/10 | 0 |
| C09 Pandas method chain | 10/10 | 10/10 | 0 |
| C10 Model evaluation | 10/13 | 12/13 | -2 |
| **Total** | **98/105** | **96/105** | **+2** |

## Usage

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="jsmall12/DataSci-Coder-14B-LoRA",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are an expert data science coding assistant. Respond ONLY with clean, runnable Python code. Use inline comments for explanation. No text outside code blocks."},
    {"role": "user", "content": "Write a function to train a logistic regression model with sklearn and print the classification report."},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.15,
        use_cache=False,
    )

response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## Project Structure

```
├── format_training_data.py     # Build training JSONL from multiple sources
├── download_public_data.py     # Download & filter public HuggingFace datasets
├── format_class_data.py        # Format university coursework examples
├── clean_data.py               # Text cleaning & structuring
├── kaggle_notebook_v3.py       # Full Kaggle training + eval notebook (T4 x2)
├── train_lightning.ipynb        # Lightning AI training + eval notebook (L40S)
├── train_lightning.py           # Standalone training script
├── train.py                     # Local training script
├── eval.py                      # Evaluation framework
├── inference.py                 # Interactive CLI inference
├── kaggle_eval.py               # Kaggle hard eval
├── kaggle_constraint_eval.py    # Kaggle constraint eval
├── kaggle_eval_full.py          # Combined Kaggle eval
├── requirements.txt             # Python dependencies
└── requirements-jarch.txt       # GPU server dependencies
```

## Hardware Requirements

- **Minimum:** ~10GB VRAM (4-bit quantized)
- **Recommended:** 24GB+ VRAM (L4, A100, etc.)
- Tested on: NVIDIA L40S (44GB), NVIDIA T4 x2 (15GB each)

## License

Apache 2.0
