"""Generate figures for the final report from the recorded LOGO results.

All numbers are transcribed from the baseline notebooks (TF-IDF + LogReg,
frozen-DeBERTa + LogReg) and the DeBERTa multitask variant evaluation tables.
Outputs PDF figures into report/figures/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(__file__))

# (model label, in-dist F1, cross-gen F1, cross-gen AUC)
MODELS = [
    ("TF-IDF\n+ LogReg", 0.9709, 0.9558, 0.9940),
    ("Frozen DeBERTa\n+ LogReg", 0.9657, 0.9437, 0.9882),
    ("DeBERTa + CE\n(base)", 0.9933, 0.9839, 0.9991),
    ("+ Contrastive", 0.9924, 0.9875, 0.9994),
    ("+ Adversarial", 0.9847, 0.9832, 0.9964),
    ("+ Both", 0.9901, 0.9831, 0.9954),
]

labels = [m[0] for m in MODELS]
indist = np.array([m[1] for m in MODELS])
crossgen = np.array([m[2] for m in MODELS])
gap = indist - crossgen

# ---- Figure 1: in-dist vs cross-gen F1 ----
x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 3.8))
b1 = ax.bar(x - w / 2, indist, w, label="In-distribution F1", color="#4C72B0")
b2 = ax.bar(x + w / 2, crossgen, w, label="Cross-generator F1", color="#DD8452")
ax.set_ylim(0.92, 1.0)
ax.set_ylabel("F1 (positive class = AI)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(loc="lower left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_title("In-distribution vs cross-generator F1 (averaged over 4 LOGO rotations)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f1_comparison.pdf"))
plt.close(fig)

# ---- Figure 2: generalization gap ----
fig, ax = plt.subplots(figsize=(7.0, 3.4))
colors = ["#C44E52" if g > 0.015 else "#55A868" for g in gap]
bars = ax.bar(x, gap, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("In-dist minus cross-gen F1")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_title("Cross-generator generalization gap (lower is better)")
for b, g in zip(bars, gap):
    ax.text(b.get_x() + b.get_width() / 2, g + 0.0008, f"{g:+.4f}",
            ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, max(gap) * 1.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gap.pdf"))
plt.close(fig)

# ---- Figure 3: per-held-out-generator cross-gen F1, the 4 variants ----
# rows: rotation/held-out; cols: CE, Contrastive, Adversarial, Both
HELD = ["gpt5mini\n(rot 0)", "deepseek\n(rot 1)", "gemma\n(rot 2)", "qwen\n(rot 3)"]
CE = [0.9912, 0.9843, 0.9681, 0.9918]
CON = [0.9902, 0.9861, 0.9808, 0.9930]
ADV = [0.9875, 0.9831, 0.9815, 0.9807]
BOTH = [0.9844, 0.9802, 0.9754, 0.9926]
series = [("CE (base)", CE), ("Contrastive", CON), ("Adversarial", ADV), ("Both", BOTH)]
xx = np.arange(len(HELD))
ww = 0.2
fig, ax = plt.subplots(figsize=(8.2, 3.8))
palette = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
for i, (name, vals) in enumerate(series):
    ax.bar(xx + (i - 1.5) * ww, vals, ww, label=name, color=palette[i])
ax.set_ylim(0.96, 1.0)
ax.set_ylabel("Cross-generator F1")
ax.set_xticks(xx)
ax.set_xticklabels(HELD, fontsize=8)
ax.legend(fontsize=8, ncol=4, loc="lower center")
ax.grid(axis="y", alpha=0.3)
ax.set_title("Cross-generator F1 by held-out generator")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_generator.pdf"))
plt.close(fig)

# ---- Figure 4: word-count distributions, human vs generators ----
def word_counts(texts):
    return np.array([len(str(t).split()) for t in texts])

human = pd.read_csv(os.path.join(ROOT, "data", "raw", "human_reviews.csv"))
ai = pd.read_csv(os.path.join(ROOT, "data", "generated", "ai_reviews.csv"))
groups = [("Human", word_counts(human["text"]), "#444444")]
gpal = {"gpt5mini": "#4C72B0", "deepseek": "#DD8452",
        "gemma": "#55A868", "qwen": "#8172B3"}
for g, c in gpal.items():
    groups.append((g, word_counts(ai[ai["generator"] == g]["text"]), c))

bins = np.linspace(0, 200, 41)
fig, ax = plt.subplots(figsize=(7.6, 3.8))
for name, wc, color in groups:
    style = "-" if name == "Human" else "--"
    lw = 2.2 if name == "Human" else 1.3
    ax.hist(wc, bins=bins, density=True, histtype="step",
            color=color, lw=lw, ls=style, label=f"{name} (mean {wc.mean():.1f})")
ax.set_xlabel("Review length (words)")
ax.set_ylabel("Density")
ax.set_xlim(0, 200)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_title("Review length distribution: human vs each generator")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "wordcounts.pdf"))
plt.close(fig)

print("wrote figures to", OUT)
