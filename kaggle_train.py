#!/usr/bin/env python3
"""Kaggle T4 GPU training notebook — Qwen2.5-Coder-14B QLoRA.

Copy this into a Kaggle notebook cell. Requirements:
- GPU T4 (or T4 x2) accelerator enabled
- Internet enabled (to download model)
- Dataset uploaded with train.jsonl and valid.jsonl

Instructions:
1. Upload train.jsonl + valid.jsonl as a Kaggle dataset
2. Create new notebook, enable GPU T4, enable Internet
3. Add your dataset
4. Paste this entire file as a single code cell and run
5. When done, download the adapter from /kaggle/working/lora_adapter/
"""

# ── Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run(
    ["pip", "install", "-q", "unsloth", "datasets", "trl"],
    check=True,
)

# ── Config ────────────────────────────────────────────────────────────
import os
os.environ["UNSLOTH_USE_FUSED_CE_LOSS"] = "0"
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"

# Change this to match your Kaggle dataset path
DATASET_PATH = "/kaggle/input/datasci-training-data"
OUTPUT_DIR = "/kaggle/working/lora_adapter"

# Model — 14B is max for T4 16GB with QLoRA
# If you get OOM, fall back to 7B by changing the line below
MODEL_NAME = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
# MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"  # fallback

MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 1        # reduced for 2048 seq len on T4
GRAD_ACCUM = 16       # effective batch = 16
EPOCHS = 3            # 3 epochs for larger dataset
LR = 3e-5             # slightly lower LR for more data + epochs
WARMUP_RATIO = 0.03

# ── Load model ────────────────────────────────────────────────────────
from unsloth import FastLanguageModel

print(f"Loading {MODEL_NAME} ...")
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

# ── Load data ─────────────────────────────────────────────────────────
from datasets import load_dataset
import os

train_path = os.path.join(DATASET_PATH, "train.jsonl")
val_path = os.path.join(DATASET_PATH, "valid.jsonl")

# Check which format the data is in
import json
with open(train_path) as f:
    sample = json.loads(f.readline())

if "messages" in sample:
    # MLX format — convert to conversations format for tokenization
    text_field = "messages"
    role_key = "role"
    content_key = "content"
    role_map = {"system": "system", "user": "user", "assistant": "assistant"}
elif "conversations" in sample:
    text_field = "conversations"
    role_key = "from"
    content_key = "value"
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}

print(f"Data format: {text_field}")

dataset = load_dataset("json", data_files={"train": train_path, "val": val_path})
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['val'])}")

# ── Tokenize with assistant-only masking ──────────────────────────────

def find_assistant_ranges(input_ids, tokenizer):
    """Find token ranges for assistant responses (loss only on these)."""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_marker = tokenizer.encode("assistant\n", add_special_tokens=False)
    marker_len = len(assistant_marker)

    ranges = []
    i = 0
    while i < len(input_ids):
        if input_ids[i] == im_start_id:
            candidate = input_ids[i + 1: i + 1 + marker_len]
            if candidate == assistant_marker:
                content_start = i + 1 + marker_len
                content_end = content_start
                while content_end < len(input_ids) and input_ids[content_end] != im_end_id:
                    content_end += 1
                if content_end < len(input_ids):
                    content_end += 1
                ranges.append((content_start, content_end))
                i = content_end
                continue
        i += 1
    return ranges


def tokenize_with_masking(examples):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for convos in examples[text_field]:
        messages = []
        for msg in convos:
            role = role_map.get(msg[role_key], msg[role_key])
            messages.append({"role": role, "content": msg[content_key]})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Mask everything except assistant responses
        labels = [-100] * len(input_ids)
        for start, end in find_assistant_ranges(input_ids, tokenizer):
            for j in range(start, min(end, len(labels))):
                labels[j] = input_ids[j]

        for j in range(len(labels)):
            if attention_mask[j] == 0:
                labels[j] = -100

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

# Verify masking
sample = tokenized["train"][0]
n_loss = sum(1 for x in sample["labels"] if x != -100)
n_total = sum(1 for x in sample["attention_mask"] if x == 1)
print(f"Sample masking: {n_loss}/{n_total} tokens have loss ({n_loss/max(n_total,1)*100:.0f}% are assistant)")

# ── Train ─────────────────────────────────────────────────────────────
from transformers import TrainingArguments, Trainer

n_steps = (len(tokenized["train"]) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
print(f"\nTotal training steps: {n_steps}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")

training_args = TrainingArguments(
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
    eval_strategy="steps",
    eval_steps=n_steps // 2 if n_steps > 10 else n_steps,
    save_strategy="no",
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    report_to="none",
)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    args=training_args,
)

print("Starting training...")
result = trainer.train()
print(f"\nTraining complete! Final loss: {result.training_loss:.4f}")

# ── Save adapter ──────────────────────────────────────────────────────
print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save config for reference
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
    "val_examples": len(tokenized["val"]),
}
import json
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("\nDone! Download the adapter from: /kaggle/working/lora_adapter/")
print("Files to download:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size / 1024 / 1024:.1f} MB)")
