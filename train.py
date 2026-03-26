#!/usr/bin/env python3
"""Step 3: QLoRA fine-tuning with Unsloth on GTX 1660 Ti (6GB VRAM).

Key design decisions for 6GB:
- Qwen2.5-1.5B-Instruct (4-bit) — fits in ~2GB
- LoRA r=8, targeting all attn+MLP layers — ~9M trainable params
- Batch=1, grad_accum=16, seq_len=512
- Proper label masking: loss only on assistant tokens
- Standard HF Trainer (Unsloth's fused CE has shape bugs)
- No mid-training eval or checkpoint saves (OOM on fp32 conversion)

Run on jarch (Arch Linux, CUDA 12.1).
Usage:
    python train.py                    # Full training
    python train.py --test             # Test with 10 examples first
    python train.py --seq-len 1024     # Increase if VRAM allows
"""

import argparse
import json
import os
from pathlib import Path

os.environ["UNSLOTH_USE_FUSED_CE_LOSS"] = "0"
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("--test", action="store_true", help="Test with 10 examples")
    parser.add_argument("--seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default="data/training")
    parser.add_argument("--output-dir", type=str, default="outputs/lora_adapter")
    return parser.parse_args()


def find_assistant_ranges(input_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """Find token ranges corresponding to assistant responses.

    For Qwen2.5 chat template, assistant content is between
    <|im_start|>assistant\n ... <|im_end|>

    Returns list of (start, end) index pairs for assistant content tokens.
    """
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


def main():
    args = parse_args()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers import TrainingArguments, Trainer

    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / args.data_dir
    OUTPUT_DIR = BASE_DIR / args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model with max_seq_length={args.seq_len}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        max_seq_length=args.seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("Loading training data...")
    train_path = str(DATA_DIR / "train_chat.jsonl")
    val_path = str(DATA_DIR / "val_chat.jsonl")

    dataset = load_dataset("json", data_files={"train": train_path, "val": val_path})

    if args.test:
        dataset["train"] = dataset["train"].select(range(min(10, len(dataset["train"]))))
        dataset["val"] = dataset["val"].select(range(min(5, len(dataset["val"]))))
        print(f"Test mode: {len(dataset['train'])} train, {len(dataset['val'])} val")

    def tokenize_with_masking(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for convos in examples["conversations"]:
            messages = []
            for msg in convos:
                role_map = {"system": "system", "human": "user", "gpt": "assistant"}
                role = role_map.get(msg["from"], msg["from"])
                messages.append({"role": role, "content": msg["value"]})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=args.seq_len,
                padding="max_length",
            )

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            labels = [-100] * len(input_ids)
            assistant_ranges = find_assistant_ranges(input_ids, tokenizer)
            for start, end in assistant_ranges:
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

    # Verify masking works
    sample = tokenized["train"][0]
    n_total = sum(1 for x in sample["labels"] if x != -100)
    n_all = sum(1 for x in sample["attention_mask"] if x == 1)
    print(f"Sample masking: {n_total}/{n_all} tokens have loss ({n_total/max(n_all,1)*100:.0f}% are assistant)")

    print(f"Train examples: {len(tokenized['train'])}")
    print(f"Val examples: {len(tokenized['val'])}")

    # No eval or checkpoint saves during training — both trigger fp32
    # conversion that OOMs on 6GB VRAM. We save once at the end instead.
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_strategy="no",
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
        args=training_args,
    )

    print("Starting training...")
    result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Final train loss: {result.training_loss:.4f}")

    # Save adapter only at the end — avoids mid-training OOM from
    # checkpoint serialization (which converts weights to fp32)
    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    config = {
        "base_model": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seq_length": args.seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "label_masking": "assistant_only",
        "train_loss": result.training_loss,
    }
    (OUTPUT_DIR / "training_config.json").write_text(json.dumps(config, indent=2))

    print("Done!")


if __name__ == "__main__":
    main()
