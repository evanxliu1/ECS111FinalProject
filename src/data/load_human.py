"""Load and filter Amazon Reviews 2023 to build the human review pool.

The McAuley-Lab/Amazon-Reviews-2023 HF dataset uses a deprecated loading script,
so we stream the underlying JSONL files directly from HF over HTTP and parse
line-by-line, filtering as we go. This avoids downloading entire multi-GB files.

Filters reviews to before ChatGPT's public release (2022-11-30).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# 2022-11-30 00:00:00 UTC in milliseconds — ChatGPT public release
PRE_CHATGPT_CUTOFF_MS = 1669766400000

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

HF_BASE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories"
)


def _stream_category(category: str, target_n: int) -> list[dict]:
    """HTTP-stream a category's JSONL, filter, return up to target_n rows."""
    url = f"{HF_BASE}/{category}.jsonl"
    rows: list[dict] = []

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        pbar = tqdm(desc=category, total=target_n, unit="rev")
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

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
                "label": 0,
                "generator": "human",
            })
            pbar.update(1)
            if len(rows) >= target_n:
                break
        pbar.close()
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
        all_rows.extend(_stream_category(cat, target))

    df = pd.DataFrame(all_rows)
    df = (
        df.drop_duplicates(subset="review_id")
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} human reviews to {out_path}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/raw/human_reviews.parquet")
    p.add_argument("--limit", type=int, default=None, help="Smoke test override")
    args = p.parse_args()

    n = args.limit if args.limit else args.n
    df = build_human_pool(n_total=n, seed=args.seed, out_path=args.out)
    print(df.head())
    print(f"\nCategory counts:\n{df['category'].value_counts()}")
    wc = df["text"].str.split().str.len()
    print(f"\nWord count: min={wc.min()}, max={wc.max()}, mean={wc.mean():.1f}")


if __name__ == "__main__":
    main()
