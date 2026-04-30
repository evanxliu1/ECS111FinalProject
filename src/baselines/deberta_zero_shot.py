"""DeBERTa-v3-base (frozen) + Logistic Regression baseline.

No fine-tuning: extract mean-pooled embeddings from a frozen encoder,
train an LR on top.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ReviewDataset
from src.models.encoder import Encoder, make_tokenizer


@torch.inference_mode()
def _embed(encoder, loader, device) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    feats, labels = [], []
    for batch in tqdm(loader, desc="embed"):
        h = encoder(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        feats.append(h.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "n": int(len(y)),
    }


def run_rotation(
    rotation: int,
    splits_dir: str = "data/splits",
    encoder_name: str = "microsoft/deberta-v3-base",
    max_len: int = 256,
    batch_size: int = 32,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_name).to(device)
    tokenizer = make_tokenizer(encoder_name)

    rdir = Path(splits_dir) / f"rotation_{rotation}"
    train_df = pd.read_parquet(rdir / "train.parquet")
    indist_df = pd.read_parquet(rdir / "test_indist.parquet")
    cross_df = pd.read_parquet(rdir / "test_crossgen.parquet")

    gen_map = {"human": 0}  # only label is used; gen_label not needed for baseline

    def _loader(df):
        ds = ReviewDataset(df, tokenizer, gen_map, max_len=max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    X_tr, y_tr = _embed(encoder, _loader(train_df), device)
    X_in, y_in = _embed(encoder, _loader(indist_df), device)
    X_cr, y_cr = _embed(encoder, _loader(cross_df), device)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr); X_in = scaler.transform(X_in); X_cr = scaler.transform(X_cr)

    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(X_tr, y_tr)
    p_in = clf.predict_proba(X_in)[:, 1]
    p_cr = clf.predict_proba(X_cr)[:, 1]

    return {
        "test_indist": _metrics(y_in, p_in),
        "test_crossgen": _metrics(y_cr, p_cr),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=str, default="data/splits")
    ap.add_argument("--rotations", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", type=str, default="results/deberta_zero_shot.json")
    args = ap.parse_args()

    all_results = {}
    for r in args.rotations:
        print(f"\nRotation {r}")
        res = run_rotation(r, args.splits_dir)
        print(json.dumps(res, indent=2))
        all_results[f"rotation_{r}"] = res

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
