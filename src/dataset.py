"""PyTorch Dataset wrapping the parquet split files."""
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_generator_label_map(train_df: pd.DataFrame) -> dict[str, int]:
    """human -> 0, then each unique AI generator gets 1..N in sorted order."""
    gens = sorted(g for g in train_df["generator"].unique() if g != "human")
    mapping = {"human": 0}
    for i, g in enumerate(gens, start=1):
        mapping[g] = i
    return mapping


class ReviewDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        gen_label_map: dict[str, int],
        max_len: int = 256,
    ):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        # Map any unseen generator (e.g., the held-out one in cross-gen test) to 0;
        # we never use gen_labels for cross-gen eval anyway.
        self.gen_labels = [gen_label_map.get(g, 0) for g in df["generator"].tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "gen_label": torch.tensor(self.gen_labels[idx], dtype=torch.long),
        }
