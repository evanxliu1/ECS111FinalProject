"""Single training run: one config × one LOGO rotation."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ReviewDataset, build_generator_label_map
from src.models.detector import Detector, compute_loss
from src.models.encoder import make_tokenizer
from src.models.losses import SupConLoss


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step: int):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def evaluate_split(model, loader, device) -> dict:
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        out = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        probs = torch.softmax(out.cls_logits, dim=-1)[:, 1]
        all_labels.append(batch["label"].numpy())
        all_probs.append(probs.cpu().numpy())
    import numpy as np
    y = np.concatenate(all_labels)
    p = np.concatenate(all_probs)
    pred = (p >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "n": int(len(y)),
    }


def train_one(
    config: dict,
    rotation: int,
    splits_dir: str,
    out_dir: str,
    max_steps: int | None = None,
    debug: bool = False,
) -> dict:
    cfg_train = config["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg_train["seed"])

    rdir = Path(splits_dir) / f"rotation_{rotation}"
    train_df = pd.read_parquet(rdir / "train.parquet")
    val_df = pd.read_parquet(rdir / "val_indist.parquet")

    gen_map = build_generator_label_map(train_df)
    print(f"Generator label map: {gen_map}")

    tokenizer = make_tokenizer(config["encoder"])
    train_ds = ReviewDataset(train_df, tokenizer, gen_map, max_len=config["max_len"])
    val_ds = ReviewDataset(val_df, tokenizer, gen_map, max_len=config["max_len"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg_train["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_train["batch_size"] * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    use_contrastive = config["losses"]["contrastive"] > 0
    use_adversarial = config["losses"]["adversarial"] > 0
    model = Detector(
        encoder_name=config["encoder"],
        projection_dim=config["projection_dim"],
        num_generators=config["num_generators"],
        use_contrastive=use_contrastive,
        use_adversarial=use_adversarial,
    ).to(device)

    encoder_params = list(model.encoder.parameters())
    head_params = model.head_parameters()
    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": cfg_train["lr_encoder"]},
            {"params": head_params, "lr": cfg_train["lr_heads"]},
        ],
        weight_decay=cfg_train["weight_decay"],
    )

    n_steps = (len(train_loader) // cfg_train["grad_accum"]) * cfg_train["epochs"]
    if max_steps:
        n_steps = min(n_steps, max_steps)
    n_warmup = int(n_steps * cfg_train["warmup_ratio"])
    scheduler = cosine_schedule_with_warmup(optimizer, n_warmup, n_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg_train["fp16"] and device.type == "cuda")
    supcon = SupConLoss() if use_contrastive else None

    best_f1, best_state = -1.0, None
    step = 0
    optimizer.zero_grad()

    for epoch in range(cfg_train["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for micro_step, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            gen_labels = batch["gen_label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg_train["fp16"] and device.type == "cuda"):
                out = model(ids, mask)
                loss, parts = compute_loss(
                    out, labels, gen_labels,
                    lambda_ce=config["losses"]["ce"],
                    lambda_contrastive=config["losses"]["contrastive"],
                    lambda_adversarial=config["losses"]["adversarial"],
                    supcon=supcon,
                )
                loss = loss / cfg_train["grad_accum"]

            scaler.scale(loss).backward()

            if (micro_step + 1) % cfg_train["grad_accum"] == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                pbar.set_postfix(parts)

                if step % cfg_train["eval_every"] == 0 or (debug and step >= 5):
                    metrics = evaluate_split(model, val_loader, device)
                    print(f"\n[step {step}] val: {metrics}")
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    model.train()

                if max_steps and step >= max_steps:
                    break
        if max_steps and step >= max_steps:
            break

    # Final eval if we never hit one above
    if best_state is None:
        metrics = evaluate_split(model, val_loader, device)
        best_f1 = metrics["f1"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out_path = Path(out_dir) / config["name"] / f"rotation_{rotation}"
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path / "best.pt")
    with (out_path / "gen_map.json").open("w") as f:
        json.dump(gen_map, f)
    with (out_path / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f)

    return {"best_val_f1": best_f1, "ckpt": str(out_path / "best.pt")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--rotation", type=int, required=True)
    p.add_argument("--splits-dir", type=str, default="data/splits")
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = train_one(
        config=config, rotation=args.rotation,
        splits_dir=args.splits_dir, out_dir=args.out_dir,
        max_steps=args.max_steps, debug=args.debug,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
