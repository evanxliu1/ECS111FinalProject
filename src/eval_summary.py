"""Shared evaluation summary tables (TF-IDF, DeBERTa, etc.)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

# rotation r holds out GENERATORS[r] (must match 02_build_splits.ipynb)
GENERATORS = ["gpt5mini", "deepseek", "gemma", "qwen"]

# Match teammate table titles, e.g. "--- TF-IDF + LOGREG BASELINE SUMMARY ---"
TFIDF_SUMMARY_TITLE = "--- TF-IDF + LOGREG BASELINE SUMMARY ---"
DEBERTA_CE_SUMMARY_TITLE = "--- DEBERTA + CE BASELINE SUMMARY ---"
VARIANT_DEBERTA_CE = "deberta_ce"

VARIANT_DISPLAY_NAMES = {
    "base": "DEBERTA + CE (BASE)",
    "contrastive": "DEBERTA + CONTRASTIVE",
    "adversarial": "DEBERTA + ADVERSARIAL",
    "both": "DEBERTA + CONTRASTIVE + ADVERSARIAL",
}


def baseline_summary_title(method_label: str) -> str:
    """Build a teammate-compatible summary banner."""
    return f"--- {method_label} BASELINE SUMMARY ---"


def _round4(value: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return round(float(value), 4)


SUMMARY_COLUMNS = [
    "Rotation",
    "In-Dist F1",
    "In-Dist AUC",
    "Cross-Gen F1",
    "Cross-Gen AUC",
    "Gap (F1)",
    *[f"{g}_F1" for g in GENERATORS],
]


def held_out_for_rotation(rotation: int) -> str:
    return GENERATORS[rotation]


def score_binary(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray | None = None) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    out: dict[str, float] = {
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
    }
    if probs is None:
        out["auc"] = float("nan")
    else:
        probs = np.asarray(probs, dtype=float)
        try:
            out["auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            out["auc"] = float("nan")
    return out


def make_rotation_row(
    rotation: int,
    indist_f1: float,
    indist_auc: float,
    crossgen_f1: float,
    crossgen_auc: float,
) -> dict[str, Any]:
    held = held_out_for_rotation(rotation)
    indist_f1 = _round4(indist_f1)
    indist_auc = _round4(indist_auc)
    crossgen_f1 = _round4(crossgen_f1)
    crossgen_auc = _round4(crossgen_auc)
    row: dict[str, Any] = {
        "Rotation": f"rotation_{rotation}",
        "In-Dist F1": indist_f1,
        "In-Dist AUC": indist_auc,
        "Cross-Gen F1": crossgen_f1,
        "Cross-Gen AUC": crossgen_auc,
        "Gap (F1)": _round4(indist_f1 - crossgen_f1),
    }
    for g in GENERATORS:
        row[f"{g}_F1"] = crossgen_f1 if g == held else np.nan
    return row


def rows_to_dataframe(rows: list[dict[str, Any]], *, include_average: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not rows or not include_average:
        return df

    avg = {
        "Rotation": "AVERAGE",
        "In-Dist F1": _round4(df["In-Dist F1"].mean()),
        "In-Dist AUC": _round4(df["In-Dist AUC"].mean()),
        "Cross-Gen F1": _round4(df["Cross-Gen F1"].mean()),
        "Cross-Gen AUC": _round4(df["Cross-Gen AUC"].mean()),
        "Gap (F1)": _round4(df["Gap (F1)"].mean()),
    }
    for g in GENERATORS:
        col = f"{g}_F1"
        avg[col] = _round4(df[col].mean(skipna=True))
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def print_summary(
    title: str,
    rows: list[dict[str, Any]],
    *,
    float_fmt: str = ".4f",
    include_average: bool = True,
) -> pd.DataFrame:
    """Print a teammate-compatible summary table and return the DataFrame."""
    df = rows_to_dataframe(rows, include_average=include_average)
    print()
    print(title)
    print()
    if df.empty:
        print("(no rotation rows yet)")
        print()
        return df
    # index=True matches teammate console output (rows 0..n)
    print(df.to_string(index=True, float_format=lambda x: format(x, float_fmt)))
    print()
    return df


def summary_csv_path(project_dir: str, variant: str = VARIANT_DEBERTA_CE) -> str:
    return os.path.join(project_dir, "runs", f"{variant}_summary.csv")


def variant_summary_title(variant: str) -> str:
    label = VARIANT_DISPLAY_NAMES.get(variant, variant.upper())
    return f"--- {label} SUMMARY ---"


def save_summary_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
