"""
Print the DeBERTa summary table (same format as TF-IDF + LogReg baseline).

Usage (from repo root):
  python scripts/print_deberta_summary.py
  python scripts/print_deberta_summary.py --demo   # format-only example for your teammate
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.environ.get("ECS111_PROJECT_DIR", os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from eval_summary import (  # noqa: E402
    DEBERTA_CE_SUMMARY_TITLE,
    make_rotation_row,
    print_summary,
    save_summary_csv,
    summary_csv_path,
)

# Reuse eval logic from train_deberta without importing the full training stack at startup
from train_deberta import (  # noqa: E402
    eval_rotation,
    publish_summary_table,
    resolve_project_dir,
    training_hparams,
)


def demo_rows():
    """Example numbers in teammate table shape (for sharing format before real runs)."""
    return [
        make_rotation_row(0, 0.8500, 0.9900, 0.7800, 0.9700),
        make_rotation_row(1, 0.8400, 0.9880, 0.7600, 0.9650),
        make_rotation_row(2, 0.8600, 0.9920, 0.7400, 0.9600),
        make_rotation_row(3, 0.8550, 0.9910, 0.8000, 0.9750),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Print a format-only example table (no checkpoints required)",
    )
    args = parser.parse_args()

    if args.demo:
        rows = demo_rows()
    else:
        project_dir = resolve_project_dir()
        os.chdir(project_dir)
        hp = training_hparams()
        rows = []
        for rotation in range(4):
            row = eval_rotation(project_dir, rotation, hp)
            if row:
                rows.append(row)
                publish_summary_table(
                    project_dir,
                    rows,
                    header=f"\n>>> Summary after rotation_{rotation} ({len(rows)}/4 done)",
                )
        return

    print_summary(DEBERTA_CE_SUMMARY_TITLE, rows)


if __name__ == "__main__":
    main()
