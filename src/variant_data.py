"""Dataset helpers for multitask DeBERTa training."""

from __future__ import annotations

import os

import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding

from deberta_multitask import build_source_map


def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def prep_text_labels_batch(batch):
    """Text + label only (test splits; no generator / source_ids)."""
    pieces = []
    for t, te in zip(batch["title"], batch["text"]):
        piece = (safe_str(t) + " " + safe_str(te)).strip()
        pieces.append(piece if piece else safe_str(te))
    return {"text": pieces, "labels": [int(x) for x in batch["label"]]}


def prep_batch(batch, source_map: dict[str, int]):
    pieces = []
    source_ids = []
    for t, te, gen in zip(batch["title"], batch["text"], batch["generator"]):
        piece = (safe_str(t) + " " + safe_str(te)).strip()
        pieces.append(piece if piece else safe_str(te))
        g = str(gen).strip()
        if g not in source_map:
            raise KeyError(f"Unknown generator {g!r}; known: {list(source_map)}")
        source_ids.append(source_map[g])
    return {
        "text": pieces,
        "labels": [int(x) for x in batch["label"]],
        "source_ids": source_ids,
    }


def tokenize_batch(batch, tokenizer, max_length: int):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    enc["labels"] = batch["labels"]
    if "source_ids" in batch:
        enc["source_ids"] = batch["source_ids"]
    return enc


def load_train_val_datasets(
    project_dir: str,
    rotation: int,
    tokenizer,
    max_length: int,
):
    train_path = os.path.join(
        project_dir, "data", "splits", f"rotation_{rotation}", "train.csv"
    )
    val_path = os.path.join(
        project_dir, "data", "splits", f"rotation_{rotation}", "val_indist.csv"
    )
    source_map, generators = build_source_map(train_path)
    raw = load_dataset("csv", data_files={"train": train_path, "validation": val_path})

    def prep_fn(batch):
        return prep_batch(batch, source_map)

    train_ds = raw["train"].map(prep_fn, batched=True, remove_columns=raw["train"].column_names)
    train_ds = train_ds.map(
        lambda b: tokenize_batch(b, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    val_ds = raw["validation"].map(prep_fn, batched=True, remove_columns=raw["validation"].column_names)
    val_ds = val_ds.map(
        lambda b: tokenize_batch(b, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    return {"train": train_ds, "validation": val_ds}, source_map, generators


def load_test_dataset(project_dir: str, rotation: int, split_name: str, tokenizer, max_length: int):
    """Test/eval splits: labels only (cross-gen rows may use held-out generators)."""
    test_path = os.path.join(
        project_dir, "data", "splits", f"rotation_{rotation}", f"{split_name}.csv"
    )
    raw = load_dataset("csv", data_files={"test": test_path})["test"]
    ds = raw.map(prep_text_labels_batch, batched=True, remove_columns=raw.column_names)
    return ds.map(
        lambda b: tokenize_batch(b, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )


class MultitaskDataCollator(DataCollatorWithPadding):
    """Pad input_ids and stack labels / source_ids."""

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        has_sources = "source_ids" in features[0]
        source_ids = [f.pop("source_ids") for f in features] if has_sources else None
        batch = super().__call__(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if source_ids is not None:
            batch["source_ids"] = torch.tensor(source_ids, dtype=torch.long)
        return batch
