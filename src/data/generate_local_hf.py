"""Generate AI reviews using local HuggingFace models on Colab T4 (4-bit).

Used for: IBM Granite 4.1 8B, Google Gemma-4 E4B-it, Qwen3.6 small variant.
"""
import argparse
import gc
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.data.prompts import build_prompt, sample_word_count

# Generator slot -> default HF model id. Override on command line if needed.
GENERATOR_MODELS = {
    "granite": "ibm-granite/granite-4.1-8b",         # pure text, Apache-2.0
    "gemma": "google/gemma-4-E4B-it",                 # multimodal; we use text-only
    "qwen": "Qwen/Qwen2.5-7B-Instruct",               # pure text, Apache-2.0
}


def _load_model(hf_id: str, four_bit: bool = True):
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Pick the right model class based on architecture (multimodal vs causal-LM)
    cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    arch_str = " ".join(getattr(cfg, "architectures", []) or []).lower()
    if "imagetexttotext" in arch_str or "vision" in arch_str or hasattr(cfg, "vision_config"):
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()
    return tok, model


def _format_chat(tok, prompt: str) -> str:
    """Use the model's chat template if available; otherwise raw prompt."""
    msgs = [{"role": "user", "content": prompt}]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt


@torch.inference_mode()
def generate_local(
    generator: str,
    n: int = 2500,
    hf_id: str | None = None,
    human_path: str = "data/raw/human_reviews.parquet",
    out_dir: str = "data/generated",
    seed: int = 42,
    four_bit: bool = True,
) -> pd.DataFrame:
    hf_id = hf_id or GENERATOR_MODELS[generator]
    rng = random.Random(seed)

    df_human = pd.read_parquet(human_path)
    seeds_df = (
        df_human.drop_duplicates(subset="parent_asin")
        .sample(n=n, random_state=seed)
        .reset_index(drop=True)
    )
    human_lengths = df_human["text"].str.split().str.len().tolist()

    tok, model = _load_model(hf_id, four_bit=four_bit)

    rows: list[dict] = []
    for _, prod in tqdm(seeds_df.iterrows(), total=len(seeds_df), desc=generator):
        target_wc = sample_word_count(rng, human_lengths)
        prompt = build_prompt(
            product_title=prod["title"],
            category=prod["category"],
            rating=prod["rating"],
            target_word_count=target_wc,
        )
        chat = _format_chat(tok, prompt)
        inputs = tok(chat, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=int(target_wc * 2.5),
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
        generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        rows.append({
            "review_id": f"{generator}_{prod['parent_asin']}_{seed}",
            "category": prod["category"],
            "parent_asin": prod["parent_asin"],
            "rating": prod["rating"],
            "title": prod["title"],
            "text": generated,
            "timestamp": None,
            "label": 1,
            "generator": generator,
        })

    del model
    gc.collect()
    torch.cuda.empty_cache()

    out_df = pd.DataFrame(rows)
    out_path = Path(out_dir) / f"{generator}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} reviews to {out_path}")
    return out_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generator", type=str, required=True, choices=list(GENERATOR_MODELS.keys()))
    p.add_argument("--hf-id", type=str, default=None)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--human", type=str, default="data/raw/human_reviews.parquet")
    p.add_argument("--out-dir", type=str, default="data/generated")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()

    generate_local(
        generator=args.generator,
        n=args.n,
        hf_id=args.hf_id,
        human_path=args.human,
        out_dir=args.out_dir,
        seed=args.seed,
        four_bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
