"""Generate AI reviews using OpenAI's API (GPT-5 mini)."""
import argparse
import os
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.data.prompts import build_prompt, sample_word_count

load_dotenv()


def _load_seed_products(human_path: str, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(human_path)
    df = df.drop_duplicates(subset="parent_asin").sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def generate_openai(
    n: int = 2500,
    model: str = "gpt-5-mini",
    human_path: str = "data/raw/human_reviews.parquet",
    out_path: str = "data/generated/gpt5mini.parquet",
    seed: int = 42,
) -> pd.DataFrame:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rng = random.Random(seed)

    seeds_df = _load_seed_products(human_path, n, seed)
    human_lengths = pd.read_parquet(human_path)["text"].str.split().str.len().tolist()

    rows: list[dict] = []
    for _, prod in tqdm(seeds_df.iterrows(), total=len(seeds_df), desc=f"openai/{model}"):
        target_wc = sample_word_count(rng, human_lengths)
        prompt = build_prompt(
            product_title=prod["title"],
            category=prod["category"],
            rating=prod["rating"],
            target_word_count=target_wc,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(target_wc * 2.5),
                temperature=0.9,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"error on {prod['parent_asin']}: {e}")
            time.sleep(2.0)
            continue

        rows.append({
            "review_id": f"gpt5mini_{prod['parent_asin']}_{seed}",
            "category": prod["category"],
            "parent_asin": prod["parent_asin"],
            "rating": prod["rating"],
            "title": prod["title"],
            "text": text,
            "timestamp": None,
            "label": 1,
            "generator": "gpt5mini",
        })

    out = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Saved {len(out)} reviews to {out_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--model", type=str, default="gpt-5-mini")
    p.add_argument("--human", type=str, default="data/raw/human_reviews.parquet")
    p.add_argument("--out", type=str, default="data/generated/gpt5mini.parquet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    generate_openai(
        n=args.n, model=args.model, human_path=args.human,
        out_path=args.out, seed=args.seed,
    )


if __name__ == "__main__":
    main()
