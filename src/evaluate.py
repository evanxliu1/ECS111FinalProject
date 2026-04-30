"""Evaluate a trained checkpoint on in-dist and cross-gen test sets."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import ReviewDataset
from src.models.detector import Detector
from src.models.encoder import make_tokenizer


@torch.inference_mode()
def _predict(model, loader, device):
    import numpy as np
    model.eval()
    labels, probs = [], []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        p = torch.softmax(out.cls_logits, dim=-1)[:, 1].cpu().numpy()
        labels.append(batch["label"].numpy()); probs.append(p)
    return np.concatenate(labels), np.concatenate(probs)


def _metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "n": int(len(y)),
    }


def evaluate_checkpoint(ckpt_dir: str, splits_dir: str, rotation: int) -> dict:
    ckpt = Path(ckpt_dir)
    with (ckpt / "config.yaml").open() as f:
        config = yaml.safe_load(f)
    with (ckpt / "gen_map.json").open() as f:
        gen_map = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = make_tokenizer(config["encoder"])

    model = Detector(
        encoder_name=config["encoder"],
        projection_dim=config["projection_dim"],
        num_generators=config["num_generators"],
        use_contrastive=config["losses"]["contrastive"] > 0,
        use_adversarial=config["losses"]["adversarial"] > 0,
    ).to(device)
    model.load_state_dict(torch.load(ckpt / "best.pt", map_location=device))

    rdir = Path(splits_dir) / f"rotation_{rotation}"
    results: dict = {}

    for split_name in ("test_indist", "test_crossgen"):
        df = pd.read_parquet(rdir / f"{split_name}.parquet")
        ds = ReviewDataset(df, tokenizer, gen_map, max_len=config["max_len"])
        loader = DataLoader(ds, batch_size=config["train"]["batch_size"] * 2, shuffle=False)
        y, p = _predict(model, loader, device)
        results[split_name] = _metrics(y, p)

        # Per-generator breakdown
        per_gen = {}
        for g in df["generator"].unique():
            mask = (df["generator"] == g) | (df["generator"] == "human")
            sub = df[mask].reset_index(drop=True)
            sub_ds = ReviewDataset(sub, tokenizer, gen_map, max_len=config["max_len"])
            sub_loader = DataLoader(sub_ds, batch_size=config["train"]["batch_size"] * 2, shuffle=False)
            yy, pp = _predict(model, sub_loader, device)
            per_gen[g] = _metrics(yy, pp)
        results[f"{split_name}_per_generator"] = per_gen

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--splits-dir", type=str, default="data/splits")
    p.add_argument("--rotation", type=int, required=True)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    res = evaluate_checkpoint(args.ckpt_dir, args.splits_dir, args.rotation)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
