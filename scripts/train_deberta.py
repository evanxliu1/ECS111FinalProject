"""
Train DeBERTa baseline from the terminal (no Jupyter).

Usage (from repo root):
  python scripts/train_deberta.py --rotation 0
  python scripts/train_deberta.py --rotation 0 --eval
  python scripts/train_deberta.py --all --eval
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.environ.get("ECS111_PROJECT_DIR", os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from eval_summary import (  # noqa: E402
    DEBERTA_CE_SUMMARY_TITLE,
    make_rotation_row,
    print_summary,
    save_summary_csv,
    score_binary,
    summary_csv_path,
)

MODEL_NAME = "microsoft/deberta-v3-base"
EPOCHS = 3
LR = 2e-5
SEED = 42


def resolve_project_dir() -> str:
    if os.path.isdir(os.path.join(PROJECT_DIR, "data", "splits")):
        return PROJECT_DIR
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "data", "splits")):
        return cwd
    raise FileNotFoundError(
        "Could not find data/splits. cd to ECS111FinalProject or set ECS111_PROJECT_DIR."
    )


def training_hparams(fp32: bool = False) -> dict:
    local_gpu = torch.cuda.is_available() and not os.path.isdir("/content/drive/MyDrive")
    # Default fp32 on local Windows — bf16 often yields loss=0 / grad_norm=nan on RTX 4060
    if local_gpu and not fp32:
        fp32 = True
    use_bf16 = (not fp32) and local_gpu and torch.cuda.is_bf16_supported()
    if local_gpu:
        return {
            "max_length": 256,
            "batch_size": 4,
            "grad_accum": 4,
            "use_bf16": use_bf16,
            "eval_batch_size": 8,
        }
    return {
        "max_length": 384,
        "batch_size": 16,
        "grad_accum": 1,
        "use_bf16": use_bf16,
        "eval_batch_size": 16,
    }


def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def build_tokenizer():
    # Avoid Hub calls during training (fixes "Server disconnected" on Windows)
    try:
        return AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=True, local_files_only=True
        )
    except OSError:
        print("Tokenizer not cached — downloading once from Hugging Face...")
        return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def prep_batch(batch):
    pieces = []
    for t, te in zip(batch["title"], batch["text"]):
        piece = (safe_str(t) + " " + safe_str(te)).strip()
        pieces.append(piece if piece else safe_str(te))
    return {"text": pieces, "labels": batch["label"]}


def tokenize_batch(batch, tokenizer, max_length):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    enc["labels"] = [int(x) for x in batch["labels"]]
    return enc


def load_split_csv(project_dir: str, rotation: int, split_name: str):
    path = os.path.join(
        project_dir, "data", "splits", f"rotation_{rotation}", f"{split_name}.csv"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return load_dataset("csv", data_files={"data": path})["data"]


def prepare_dataset(raw, tokenizer, max_length):
    drop_cols = raw.column_names
    ds = raw.map(prep_batch, batched=True, remove_columns=drop_cols)
    return ds.map(
        lambda b: tokenize_batch(b, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def checkpoint_has_nan(ckpt_dir: str) -> bool:
    if not os.path.isdir(ckpt_dir):
        return False
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir, dtype=torch.float32
    )
    bad = torch.isnan(model.classifier.weight).any().item()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bool(bad)


def warn_bad_checkpoint(ckpt_dir: str) -> None:
    print("\n*** BAD CHECKPOINT (NaN weights) — from an old bf16/failed run ***")
    print(f"Delete this folder, then retrain:\n  {ckpt_dir}\n")
    print("PowerShell:")
    print(f'  Remove-Item -Recurse -Force "{ckpt_dir}"')
    print("Then:")
    print("  python scripts/train_deberta.py --rotation 0")
    print("(fp32 is automatic on local GPU)\n")


def load_classifier_model(ckpt: str | None = None):
    """Transformers 5.x defaults to float16 — that breaks fine-tuning on RTX GPUs."""
    kwargs = {
        "num_labels": 2,
        "id2label": {0: "human", 1: "ai"},
        "label2id": {"human": 0, "ai": 1},
        "problem_type": "single_label_classification",
        "dtype": torch.float32,
    }
    name = ckpt or MODEL_NAME
    load_kw = dict(kwargs)
    if ckpt is None:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                name, local_files_only=True, **load_kw
            )
        except OSError:
            print("Model not cached — downloading once from Hugging Face...")
            model = AutoModelForSequenceClassification.from_pretrained(name, **load_kw)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name, **load_kw)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def train_rotation(project_dir: str, rotation: int, hp: dict) -> str:
    set_seed()
    train_path = os.path.join(
        project_dir, "data", "splits", f"rotation_{rotation}", "train.csv"
    )
    val_path = os.path.join(
        project_dir, "data", "splits", f"rotation_{rotation}", "val_indist.csv"
    )
    out_dir = os.path.join(project_dir, "checkpoints", "deberta_ce", f"rotation_{rotation}")
    save_path = os.path.join(out_dir, "best_hf")
    if os.path.isdir(out_dir):
        import shutil

        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Train rotation_{rotation} ===")
    print("train:", train_path)
    print("val:  ", val_path)
    print("out:  ", save_path)

    tokenizer = build_tokenizer()
    raw = load_dataset(
        "csv", data_files={"train": train_path, "validation": val_path}
    )
    ds = prepare_dataset(raw["train"], tokenizer, hp["max_length"])
    ds_val = prepare_dataset(raw["validation"], tokenizer, hp["max_length"])
    ds = {"train": ds, "validation": ds_val}

    model = load_classifier_model()
    print("model dtype:", next(model.parameters()).dtype)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(
            torch.tensor(logits, dtype=torch.float32), dim=-1
        ).numpy()[:, 1]
        out = {"f1_ai": f1_score(labels, preds, pos_label=1)}
        try:
            out["roc_auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            out["roc_auc"] = float("nan")
        return out

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=LR,
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        max_grad_norm=1.0,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_ai",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        report_to="none",
        bf16=hp["use_bf16"],
        fp16=False,
        dataloader_num_workers=0,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model = trainer.model
    if torch.cuda.is_available():
        model = model.float()
    if torch.isnan(model.classifier.weight).any():
        raise RuntimeError(
            "Training produced NaN weights — delete checkpoints/deberta_ce and retry."
        )
    model = model.float()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("Saved:", save_path)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_path


def predict_split(model, tokenizer, collator, dataset, batch_size: int, project_dir: str):
    predict_args = TrainingArguments(
        output_dir=os.path.join(project_dir, "_predict_tmp"),
        per_device_eval_batch_size=batch_size,
        report_to="none",
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=predict_args,
        processing_class=tokenizer,
        data_collator=collator,
    )
    out = trainer.predict(dataset)
    logits = out.predictions
    labels = out.label_ids
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    ).numpy()[:, 1]
    return labels, preds, probs


def publish_summary_table(project_dir: str, rows: list, header: str | None = None) -> None:
    """Print teammate-format table + update CSV after each rotation."""
    if not rows:
        return
    if header:
        print(header)
    df = print_summary(DEBERTA_CE_SUMMARY_TITLE, rows)
    out_csv = summary_csv_path(project_dir)
    save_summary_csv(df, out_csv)
    print("Saved:", out_csv)


def eval_rotation(project_dir: str, rotation: int, hp: dict, ckpt: str | None = None):
    ckpt = ckpt or os.path.join(
        project_dir, "checkpoints", "deberta_ce", f"rotation_{rotation}", "best_hf"
    )
    if not os.path.isdir(ckpt):
        print(f"skip eval rotation_{rotation}: no checkpoint at {ckpt}")
        return None
    if checkpoint_has_nan(ckpt):
        warn_bad_checkpoint(ckpt)
        return None

    print(f"\n=== Eval rotation_{rotation} ===")
    tokenizer = build_tokenizer()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = load_classifier_model(ckpt)

    bs = hp["eval_batch_size"]
    results = {}
    for split_name in ("test_indist", "test_crossgen"):
        print(f"  loading {split_name}...")
        raw = load_split_csv(project_dir, rotation, split_name)
        ds = prepare_dataset(raw, tokenizer, hp["max_length"])
        print(f"  predicting {split_name} ({len(ds)} rows)...")
        y, p, pr = predict_split(model, tokenizer, collator, ds, bs, project_dir)
        results[split_name] = score_binary(y, p, pr)
        del ds, raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    si, sc = results["test_indist"], results["test_crossgen"]
    row = make_rotation_row(rotation, si["f1"], si["auc"], sc["f1"], sc["auc"])
    print(
        f"  rotation_{rotation}: "
        f"in-dist F1={si['f1']:.4f} AUC={si['auc']:.4f} | "
        f"cross-gen F1={sc['f1']:.4f} AUC={sc['auc']:.4f}"
    )
    return row


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa CE baseline locally")
    parser.add_argument("--rotation", type=int, default=0, help="LOGO rotation 0-3")
    parser.add_argument("--all", action="store_true", help="Run rotations 0-3")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate test_indist + test_crossgen after training",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only run test evaluation on existing checkpoint",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Train in full precision (default on local GPU; use if bf16 breaks)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing checkpoint for this rotation before training",
    )
    args = parser.parse_args()

    project_dir = resolve_project_dir()
    os.chdir(project_dir)
    hp = training_hparams(fp32=args.fp32)

    print("PROJECT_DIR:", project_dir)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"settings: batch={hp['batch_size']}, accum={hp['grad_accum']}, "
        f"max_len={hp['max_length']}, bf16={hp['use_bf16']}"
    )

    rotations = list(range(4)) if args.all else [args.rotation]
    total = len(rotations)
    rows = []

    for i, r in enumerate(rotations):
        ckpt = None
        if not args.eval_only:
            ckpt_dir = os.path.join(
                project_dir, "checkpoints", "deberta_ce", f"rotation_{r}", "best_hf"
            )
            if args.force and os.path.isdir(ckpt_dir):
                import shutil

                shutil.rmtree(ckpt_dir)
                print("Removed old checkpoint:", ckpt_dir)
            elif checkpoint_has_nan(ckpt_dir):
                warn_bad_checkpoint(ckpt_dir)
                print("Use --force to delete and retrain, or delete the folder manually.\n")
                continue
            ckpt = train_rotation(project_dir, r, hp)
        if args.eval or args.eval_only:
            row = eval_rotation(project_dir, r, hp, ckpt)
            if row:
                rows.append(row)
                publish_summary_table(
                    project_dir,
                    rows,
                    header=f"\n>>> Summary after rotation_{r} ({len(rows)}/{total} done)",
                )
        elif not args.eval_only:
            # Train-only: still print a one-line status per rotation
            print(f"\n>>> Finished training rotation_{r} ({i + 1}/{total} done)")


if __name__ == "__main__":
    main()
