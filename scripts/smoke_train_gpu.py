"""Quick GPU smoke test — run from repo root: python scripts/smoke_train_gpu.py"""
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROTATION = 0
MAX_LENGTH = 256
BATCH_SIZE = 4

train_path = f"data/splits/rotation_{ROTATION}/train.csv"
val_path = f"data/splits/rotation_{ROTATION}/val_indist.csv"
raw = load_dataset("csv", data_files={"train": train_path, "validation": val_path})
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-base", use_fast=True, local_files_only=True
)


def prep(batch):
    texts = [f"{t or ''} {te or ''}".strip() for t, te in zip(batch["title"], batch["text"])]
    return {"text": texts, "labels": batch["label"]}


def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)
    enc["labels"] = batch["labels"]
    return enc


ds = raw.map(prep, batched=True, remove_columns=raw["train"].column_names)
ds = ds.map(tokenize, batched=True, remove_columns=["text"])
print("tokenized", len(ds["train"]), "rows")

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=2, local_files_only=True
).cuda()
collator = DataCollatorWithPadding(tokenizer=tokenizer)
args = TrainingArguments(
    output_dir="./_smoke",
    per_device_train_batch_size=BATCH_SIZE,
    max_steps=5,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    dataloader_num_workers=0,
    logging_steps=1,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"].select(range(64)),
    data_collator=collator,
)
trainer.train()
print("OK — peak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))
