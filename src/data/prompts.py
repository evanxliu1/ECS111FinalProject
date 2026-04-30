"""Prompt construction for AI review generation.

Single template parameterized by product metadata. Word count is sampled
to match the human review length distribution to avoid trivial detection
by length alone.
"""
import random

PROMPT_TEMPLATE = (
    "Write a {rating}-star Amazon product review for: {product_title}\n"
    "Category: {category}\n"
    "Length: about {target_word_count} words.\n"
    "Voice: a real customer who bought this product. No emojis, no hashtags.\n"
    "Output only the review body — no title, no preamble."
)


def sample_word_count(rng: random.Random, human_lengths: list[int] | None = None) -> int:
    """Sample a target word count matching the human distribution if provided."""
    if human_lengths:
        return rng.choice(human_lengths)
    # Fallback: rough lognormal-ish distribution typical for product reviews
    return int(rng.lognormvariate(mu=4.0, sigma=0.6))


def build_prompt(product_title: str, category: str, rating: float, target_word_count: int) -> str:
    return PROMPT_TEMPLATE.format(
        product_title=product_title or "this product",
        category=category.replace("_", " "),
        rating=int(round(rating)) if rating else 5,
        target_word_count=target_word_count,
    )
