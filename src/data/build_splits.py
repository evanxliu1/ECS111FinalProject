"""Build leave-one-generator-out (LOGO) splits.

For each rotation r in 0..3:
  - hold out generator r as the cross-gen test set
  - train on human + the other 3 generators (1:1 balanced)
  - in-dist val/test are held-out portions of the 3 train generators + matched human

Critically maintains a *disjoint human pool* between training and the cross-gen
test set, so cross-gen evaluation isn't contaminated.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENERATORS = ["gpt5mini", "granite", "gemma", "qwen"]


def _load_all(human_path: str, gen_dir: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    human = pd.read_parquet(human_path)
    gens = {g: pd.read_parquet(f"{gen_dir}/{g}.parquet") for g in GENERATORS}
    return human, gens


def _split_indist(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10 train/val/test for an in-dist generator's reviews."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return (
        df.iloc[:n_train],
        df.iloc[n_train:n_train + n_val],
        df.iloc[n_train + n_val:],
    )


def build_rotation(
    rotation: int,
    human: pd.DataFrame,
    gens: dict[str, pd.DataFrame],
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    held_out = GENERATORS[rotation]
    train_gens = [g for g in GENERATORS if g != held_out]

    rng = np.random.default_rng(seed + rotation)

    # 1) Carve human pool: 2500 reserved for cross-gen test, rest for in-dist
    human_shuffled = human.sample(frac=1.0, random_state=seed + rotation).reset_index(drop=True)
    human_crossgen = human_shuffled.iloc[:2500]
    human_indist_pool = human_shuffled.iloc[2500:].reset_index(drop=True)

    # 2) Per-generator 80/10/10 split for the 3 in-dist generators
    train_chunks, val_chunks, test_chunks = [], [], []
    for g in train_gens:
        tr, va, te = _split_indist(gens[g], seed=seed + rotation)
        train_chunks.append(tr); val_chunks.append(va); test_chunks.append(te)

    ai_train = pd.concat(train_chunks, ignore_index=True)
    ai_val = pd.concat(val_chunks, ignore_index=True)
    ai_test = pd.concat(test_chunks, ignore_index=True)

    # 3) Balance human side 1:1 against AI counts, sampling from the in-dist human pool
    h_pool = human_indist_pool.sample(frac=1.0, random_state=seed + rotation).reset_index(drop=True)
    n_tr, n_va, n_te = len(ai_train), len(ai_val), len(ai_test)
    assert n_tr + n_va + n_te <= len(h_pool), "Not enough human reviews to balance"
    h_train = h_pool.iloc[:n_tr]
    h_val = h_pool.iloc[n_tr:n_tr + n_va]
    h_test = h_pool.iloc[n_tr + n_va:n_tr + n_va + n_te]

    # 4) Cross-gen test = held-out generator + reserved human pool
    ai_crossgen = gens[held_out]

    def _combine(*dfs: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat(dfs, ignore_index=True)
        return out.sample(frac=1.0, random_state=seed + rotation).reset_index(drop=True)

    return {
        "train": _combine(h_train, ai_train),
        "val_indist": _combine(h_val, ai_val),
        "test_indist": _combine(h_test, ai_test),
        "test_crossgen": _combine(human_crossgen, ai_crossgen),
        "_meta": pd.DataFrame([{
            "rotation": rotation,
            "held_out": held_out,
            "train_gens": ",".join(train_gens),
            "n_train": len(h_train) + len(ai_train),
            "n_val": len(h_val) + len(ai_val),
            "n_test_indist": len(h_test) + len(ai_test),
            "n_test_crossgen": len(human_crossgen) + len(ai_crossgen),
        }]),
    }


def write_rotation(splits: dict[str, pd.DataFrame], out_dir: str, rotation: int) -> None:
    rdir = Path(out_dir) / f"rotation_{rotation}"
    rdir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(rdir / f"{name.lstrip('_')}.parquet", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--human", type=str, default="data/raw/human_reviews.parquet")
    p.add_argument("--gen-dir", type=str, default="data/generated")
    p.add_argument("--out-dir", type=str, default="data/splits")
    p.add_argument("--rotation", type=int, default=None, help="Build only this rotation (0-3); default builds all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    human, gens = _load_all(args.human, args.gen_dir)
    rotations = [args.rotation] if args.rotation is not None else list(range(len(GENERATORS)))

    for r in rotations:
        splits = build_rotation(r, human, gens, seed=args.seed)
        write_rotation(splits, args.out_dir, r)

        meta = splits["_meta"].iloc[0]
        print(f"\nRotation {r} (held out: {meta['held_out']})")
        print(f"  train: {meta['n_train']}, val: {meta['n_val']}, "
              f"test_indist: {meta['n_test_indist']}, test_crossgen: {meta['n_test_crossgen']}")

        if args.debug:
            train_ids = set(splits["train"]["review_id"])
            cg_ids = set(splits["test_crossgen"]["review_id"])
            assert train_ids.isdisjoint(cg_ids), "LEAKAGE: train and cross-gen share reviews"
            print("  ✓ no review_id leakage between train and cross-gen test")


if __name__ == "__main__":
    main()
