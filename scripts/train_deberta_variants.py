"""
Train DeBERTa LOGO variants (base / contrastive / adversarial / both).

Usage (from repo root):
  python scripts/train_deberta_variants.py --variant contrastive --rotation 0
  python scripts/train_deberta_variants.py --variant all --all --eval
  python scripts/train_deberta_variants.py --variant both --all --eval --skip-existing

Skips `base` training if checkpoints/deberta_ce/rotation_{r}/best_hf exists (same as CE baseline).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer, TrainingArguments

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.environ.get("ECS111_PROJECT_DIR", os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
sys.path.insert(0, SCRIPT_DIR)

from deberta_multitask import VARIANTS, DebertaMultitaskModel, MultitaskConfig  # noqa: E402
from eval_summary import (  # noqa: E402
    make_rotation_row,
    print_summary,
    save_summary_csv,
    score_binary,
    summary_csv_path,
    variant_summary_title,
)
from multitask_trainer import MultitaskTrainer  # noqa: E402
from variant_data import (  # noqa: E402
    MultitaskDataCollator,
    load_test_dataset,
    load_train_val_datasets,
)

MODEL_NAME = "microsoft/deberta-v3-base"
EPOCHS = 3
LR = 2e-5
SEED = 42
LAMBDA_C = 0.5
LAMBDA_A = 0.5
PROJ_DIM = 128


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


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def build_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def checkpoint_dir(project_dir: str, variant: str, rotation: int) -> str:
    return os.path.join(project_dir, "checkpoints", variant, f"rotation_{rotation}", "best_hf")


def has_valid_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "pytorch_model.bin")):
        return False
    if not os.path.isfile(os.path.join(path, "multitask_config.json")):
        return False
    try:
        model = DebertaMultitaskModel.from_pretrained(path, map_location="cpu")
        ok = not torch.isnan(model.classifier.weight).any().item()
        del model
        return ok
    except Exception:
        return False


def train_variant_rotation(
    project_dir: str,
    variant: str,
    rotation: int,
    hp: dict,
    force: bool = False,
) -> str | None:
    save_path = checkpoint_dir(project_dir, variant, rotation)

    if variant == "base":
        ce_path = os.path.join(project_dir, "checkpoints", "deberta_ce", f"rotation_{rotation}", "best_hf")
        if os.path.isdir(ce_path) and not force:
            print(f"skip base rotation_{rotation}: using existing deberta_ce at {ce_path}")
            return ce_path

    if not force and has_valid_checkpoint(save_path):
        print(f"skip {variant} rotation_{rotation}: checkpoint exists at {save_path}")
        return save_path

    set_seed()
    out_dir = os.path.join(project_dir, "checkpoints", variant, f"rotation_{rotation}")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Train {variant} rotation_{rotation} ===")
    print("out:", save_path)

    tokenizer = build_tokenizer()
    datasets, source_map, generators = load_train_val_datasets(
        project_dir, rotation, tokenizer, hp["max_length"]
    )
    print("sources:", generators, "->", source_map)

    config = MultitaskConfig(
        model_name=MODEL_NAME,
        variant=variant,
        num_labels=2,
        num_sources=len(generators),
        proj_dim=PROJ_DIM,
        lambda_c=LAMBDA_C,
        lambda_a=LAMBDA_A,
        source_map=source_map,
    )
    model = DebertaMultitaskModel(config)
    if torch.cuda.is_available():
        model = model.cuda()
    print("model dtype:", next(model.parameters()).dtype)

    collator = MultitaskDataCollator(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()[:, 1]
        out = {"f1_ai": f1_score(labels, preds, pos_label=1, zero_division=0)}
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

    trainer = MultitaskTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        grl_gamma=config.grl_gamma,
        grl_max_lambda=config.grl_max_lambda,
    )
    trainer.train()
    model = trainer.model
    if torch.cuda.is_available():
        model = model.float()
    if torch.isnan(model.classifier.weight).any():
        raise RuntimeError(f"NaN weights for {variant} rotation_{rotation}")

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Saved:", save_path)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_path


def eval_variant_rotation(project_dir: str, variant: str, rotation: int, hp: dict):
    if variant == "base":
        mt_ckpt = checkpoint_dir(project_dir, "base", rotation)
        ce_ckpt = os.path.join(
            project_dir, "checkpoints", "deberta_ce", f"rotation_{rotation}", "best_hf"
        )
        if has_valid_checkpoint(mt_ckpt):
            ckpt = mt_ckpt
        elif os.path.isdir(ce_ckpt):
            import train_deberta as ce_train

            return ce_train.eval_rotation(project_dir, rotation, hp, ce_ckpt)
        else:
            print(f"skip eval base rotation_{rotation}: no checkpoint")
            return None
    else:
        ckpt = checkpoint_dir(project_dir, variant, rotation)

    if not has_valid_checkpoint(ckpt):
        print(f"skip eval {variant} rotation_{rotation}: no checkpoint at {ckpt}")
        return None

    print(f"\n=== Eval {variant} rotation_{rotation} ===")
    tokenizer = build_tokenizer()
    config = MultitaskConfig.load(ckpt)
    model = DebertaMultitaskModel.from_pretrained(ckpt)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    collator = MultitaskDataCollator(tokenizer=tokenizer)
    predict_args = TrainingArguments(
        output_dir=os.path.join(project_dir, "_predict_tmp"),
        per_device_eval_batch_size=hp["eval_batch_size"],
        report_to="none",
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
    )
    trainer = MultitaskTrainer(
        model=model,
        args=predict_args,
        processing_class=tokenizer,
        data_collator=collator,
    )

    results = {}
    for split_name in ("test_indist", "test_crossgen"):
        print(f"  loading {split_name}...")
        ds = load_test_dataset(project_dir, rotation, split_name, tokenizer, hp["max_length"])
        print(f"  predicting {split_name} ({len(ds)} rows)...")
        out = trainer.predict(ds)
        logits = out.predictions
        labels = out.label_ids
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()[:, 1]
        results[split_name] = score_binary(labels, preds, probs)
        del ds

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    si, sc = results["test_indist"], results["test_crossgen"]
    print(
        f"  rotation_{rotation}: in-dist F1={si['f1']:.4f} cross-gen F1={sc['f1']:.4f} "
        f"gap={si['f1'] - sc['f1']:.4f}"
    )
    return make_rotation_row(rotation, si["f1"], si["auc"], sc["f1"], sc["auc"])


def publish_variant_summary(project_dir: str, variant: str, rows: list, header: str | None = None):
    if not rows:
        return
    if header:
        print(header)
    title = variant_summary_title(variant)
    df = print_summary(title, rows)
    out_csv = summary_csv_path(project_dir, variant)
    save_summary_csv(df, out_csv)
    print("Saved:", out_csv)


def parse_variants(variant_arg: str) -> list[str]:
    if variant_arg == "all":
        return list(VARIANTS)
    if variant_arg not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS} or 'all'")
    return [variant_arg]


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa proposal variants")
    parser.add_argument(
        "--variant",
        default="contrastive",
        help="base | contrastive | adversarial | both | all",
    )
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Run rotations 0-3")
    parser.add_argument("--eval", action="store_true", help="Eval test splits after training")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip train if checkpoint exists")
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoints")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    project_dir = resolve_project_dir()
    os.chdir(project_dir)
    hp = training_hparams(fp32=args.fp32)
    variants = parse_variants(args.variant)
    rotations = list(range(4)) if args.all else [args.rotation]

    print("PROJECT_DIR:", project_dir)
    print("variants:", variants)
    print("rotations:", rotations)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"settings: batch={hp['batch_size']}, accum={hp['grad_accum']}, "
        f"max_len={hp['max_length']}, bf16={hp['use_bf16']}, "
        f"lambda_c={LAMBDA_C}, lambda_a={LAMBDA_A}"
    )

    for variant in variants:
        rows = []
        for r in rotations:
            ckpt = None
            if not args.eval_only:
                if args.skip_existing and not args.force:
                    ckpt = train_variant_rotation(project_dir, variant, r, hp, force=False)
                else:
                    ckpt = train_variant_rotation(project_dir, variant, r, hp, force=args.force)
            if args.eval or args.eval_only:
                row = eval_variant_rotation(project_dir, variant, r, hp)
                if row:
                    rows.append(row)
                    publish_variant_summary(
                        project_dir,
                        variant,
                        rows,
                        header=f"\n>>> {variant} after rotation_{r} ({len(rows)}/{len(rotations)})",
                    )


if __name__ == "__main__":
    main()
