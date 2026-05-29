"""Print train/val/test scores to inspect overfitting."""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)


def load_val_history(variant: str, rotation: int) -> list[dict]:
    pattern = os.path.join("checkpoints", variant, f"rotation_{rotation}", "**", "trainer_state.json")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        return []
    latest = max(paths, key=os.path.getmtime)
    with open(latest, encoding="utf-8") as f:
        state = json.load(f)
    return [x for x in state.get("log_history", []) if "eval_f1_ai" in x]


def main():
    print("=" * 72)
    print("TEST SET SCORES (from runs/*_summary.csv)")
    print("Large Gap (F1) = in-dist much better than cross-gen (generalization drop)")
    print("=" * 72)
    for path in sorted(glob.glob("runs/*_summary.csv")):
        name = os.path.basename(path).replace("_summary.csv", "")
        df = pd.read_csv(path)
        print(f"\n--- {name} ---")
        cols = ["Rotation", "In-Dist F1", "In-Dist AUC", "Cross-Gen F1", "Cross-Gen AUC", "Gap (F1)"]
        print(df[cols].to_string(index=False))

    print("\n" + "=" * 72)
    print("VALIDATION DURING TRAINING (val_indist.csv, end of each epoch)")
    print("=" * 72)
    for variant in ["deberta_ce", "contrastive", "adversarial", "both"]:
        print(f"\n--- {variant} ---")
        rows = []
        for r in range(4):
            evals = load_val_history(variant, r)
            if not evals:
                rows.append({"rotation": r, "note": "no trainer_state"})
                continue
            for i, e in enumerate(evals, 1):
                rows.append(
                    {
                        "rotation": r,
                        "epoch": round(float(e.get("epoch", 0)), 2),
                        "eval_loss": round(float(e.get("eval_loss", 0)), 4),
                        "eval_f1_ai": round(float(e.get("eval_f1_ai", 0)), 4),
                        "eval_auc": round(float(e.get("eval_roc_auc", 0)), 4),
                    }
                )
        if rows and "note" not in rows[0]:
            pdf = pd.DataFrame(rows)
            for r in range(4):
                sub = pdf[pdf["rotation"] == r]
                if sub.empty:
                    print(f"  rotation_{r}: (missing)")
                else:
                    print(f"  rotation_{r}:")
                    print(sub.to_string(index=False))
        else:
            print("  (no checkpoint logs found)")


if __name__ == "__main__":
    main()
