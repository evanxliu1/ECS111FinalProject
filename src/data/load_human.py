"""Load and filter Amazon Reviews 2023 to build the human review pool.

Filters reviews to before ChatGPT's public release (2022-11-30) to ensure
the human pool is genuinely human-written.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Cutoff: ChatGPT public release was 2022-11-30
PRE_CHATGPT_CUTOFF_MS = 1669766400000  # 2022-11-30 00:00:00 UTC in ms

CATEGORIES = [
    "All_Beauty",
    "Books",
    "Electronics",
    "Home_and_Kitchen",
    "Sports_and_Outdoors",
    "Toys_and_Games",
    "Pet_Supplies",
    "Office_Products",
]

MIN_WORDS = 20
MAX_WORDS = 400


def _stream_category(category: str, target_n: int, seed: int) -> list[dict]:
    """Stream a category, filter by date and length, return up to target_n rows."""
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    rows = []
    for ex in tqdm(ds, desc=category, total=target_n * 4):
        ts = ex.get("timestamp")
        text = ex.get("text") or ""
        if not ts or ts >= PRE_CHATGPT_CUTOFF_MS:
            continue
        wc = len(text.split())
        if wc < MIN_WORDS or wc > MAX_WORDS:
            continue
        rows.append({
            "review_id": f"{category}_{ex.get('user_id', '')}_{ts}",
            "category": category,
            "parent_asin": ex.get("parent_asin"),
            "rating": ex.get("rating"),
            "title": ex.get("title"),
            "text": text,
            "timestamp": ts,
            "label": 0,        # 0 = human, 1 = AI
            "generator": "human",
        })
        if len(rows) >= target_n:
            break
    return rows


def build_human_pool(
    n_total: int = 10_000,
    seed: int = 42,
    out_path: str = "data/raw/human_reviews.parquet",
) -> pd.DataFrame:
    per_cat = n_total // len(CATEGORIES)
    extra = n_total - per_cat * len(CATEGORIES)

    all_rows: list[dict] = []
    for i, cat in enumerate(CATEGORIES):
        target = per_cat + (1 if i < extra else 0)
        all_rows.extend(_stream_category(cat, target, seed + i))

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset="review_id").sample(frac=1, random_state=seed).reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} human reviews to {out_path}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/raw/human_reviews.parquet")
    p.add_argument("--limit", type=int, default=None, help="Smoke test: load this many total only")
    args = p.parse_args()

    n = args.limit if args.limit else args.n
    df = build_human_pool(n_total=n, seed=args.seed, out_path=args.out)
    print(df.head())
    print(f"\nCategory counts:\n{df['category'].value_counts()}")
    print(f"\nWord count stats: min={df['text'].str.split().str.len().min()}, "
          f"max={df['text'].str.split().str.len().max()}, "
          f"mean={df['text'].str.split().str.len().mean():.1f}")


if __name__ == "__main__":
    main()
