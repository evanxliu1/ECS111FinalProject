"""TF-IDF + Logistic Regression baseline.

Train on a rotation's training set, evaluate in-dist and cross-gen.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


def _metrics(y, p, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "n": int(len(y)),
    }


def run_rotation(rotation: int, splits_dir: str = "data/splits") -> dict:
    rdir = Path(splits_dir) / f"rotation_{rotation}"
    train = pd.read_parquet(rdir / "train.parquet")
    test_indist = pd.read_parquet(rdir / "test_indist.parquet")
    test_cross = pd.read_parquet(rdir / "test_crossgen.parquet")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            min_df=2,
            sublinear_tf=True,
        )),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)),
    ])
    pipe.fit(train["text"].astype(str), train["label"].astype(int))

    out: dict = {}
    for name, df in [("test_indist", test_indist), ("test_crossgen", test_cross)]:
        p = pipe.predict_proba(df["text"].astype(str))[:, 1]
        out[name] = _metrics(df["label"].astype(int).values, p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=str, default="data/splits")
    ap.add_argument("--rotations", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", type=str, default="results/tfidf_lr.json")
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
