"""Score train / val / test splits for saved checkpoints."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.chdir(PROJECT)

from eval_summary import score_binary  # noqa: E402

MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 256
BATCH = 8
SPLITS = ("train", "val_indist", "test_indist", "test_crossgen")


def prep(batch):
    pieces = []
    for t, te in zip(batch["title"], batch["text"]):
        t, te = str(t or "").strip(), str(te or "").strip()
        pieces.append((t + " " + te).strip() or te)
    return {"text": pieces, "labels": [int(x) for x in batch["label"]]}


def load_split(rotation: int, name: str):
    path = os.path.join("data", "splits", f"rotation_{rotation}", f"{name}.csv")
    raw = load_dataset("csv", data_files={"d": path})["d"]
    return raw.map(prep, batched=True, remove_columns=raw.column_names)


def tokenize(ds, tok):
    def fn(batch):
        enc = tok(batch["text"], truncation=True, max_length=MAX_LEN, padding=False)
        enc["labels"] = batch["labels"]
        return enc

    return ds.map(fn, batched=True, remove_columns=["text"])


def score_ckpt(ckpt: str, ds, tok, coll):
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.float32)
    if torch.cuda.is_available():
        model = model.cuda()
    out = Trainer(model=model, processing_class=tok, data_collator=coll).predict(ds)
    preds = np.argmax(out.predictions, axis=-1)
    probs = torch.softmax(torch.tensor(out.predictions, dtype=torch.float32), dim=-1).numpy()[:, 1]
    s = score_binary(out.label_ids, preds, probs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return s["f1"], s["auc"]


def main():
    variants = sys.argv[1:] if len(sys.argv) > 1 else ["deberta_ce", "contrastive"]
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    coll = DataCollatorWithPadding(tokenizer=tok)

    for var in variants:
        print(f"\n===== {var} =====")
        print(f"{'rotation':<10} {'split':<16} {'F1':>8} {'AUC':>8}")
        rows = []
        for r in range(4):
            ckpt = os.path.join("checkpoints", var, f"rotation_{r}", "best_hf")
            if not os.path.isdir(ckpt):
                print(f"  rotation_{r}: missing {ckpt}")
                continue
            for split in SPLITS:
                ds = tokenize(load_split(r, split), tok)
                f1, auc = score_ckpt(ckpt, ds, tok, coll)
                rows.append((r, split, f1, auc))
                print(f"  rotation_{r:<4} {split:<16} {f1:>8.4f} {auc:>8.4f}")

        if rows:
            print(f"\n  {'AVERAGE':<10} {'split':<16} {'F1':>8} {'AUC':>8}")
            for split in SPLITS:
                sub = [x for x in rows if x[1] == split]
                print(
                    f"  {'(avg)':<10} {split:<16} "
                    f"{np.mean([x[2] for x in sub]):>8.4f} {np.mean([x[3] for x in sub]):>8.4f}"
                )


if __name__ == "__main__":
    main()
