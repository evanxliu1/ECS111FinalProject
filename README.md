# Cross-Generator AI Review Detector — ECS111 Final Project

Building a binary classifier (human vs AI) for Amazon product reviews that
**generalizes across LLM generators** — i.e. it still works when tested on
reviews from a model it never saw during training. The planned method adds
contrastive loss + a generator-adversarial head on top of a DeBERTa encoder.

## How this repo works

- **Code** (the notebooks) lives in git — GitHub is the source of truth.
- **Data** (the CSVs under `data/`) is tracked in git too, so the dataset
  travels with the repo.
- **Training artifacts** (`checkpoints/`, `runs/`, `wandb/`) are gitignored —
  they're large and regenerated.

Everything runs as self-contained Colab notebooks. There is no `src/` package
and no `requirements.txt`; each notebook installs exactly what it needs.

### Setup

Run `notebooks/00_setup.ipynb` first, once per Colab session. It mounts Google
Drive and clones (or pulls) this repo into `MyDrive/ECS111FinalProject`, so code
and data sit together in one folder. Edit and `git push` code from your laptop;
re-run `00_setup` in Colab to `git pull` the latest.

## Pipeline status

### Done — data collection and splitting

| Notebook | What it does |
|---|---|
| `00_setup.ipynb` | Mounts Drive, clones/pulls the repo into Drive. Run once per session. |
| `01_collect_human_reviews.ipynb` | Streams the McAuley-Lab **Amazon Reviews 2023** dataset from HuggingFace, filters to **pre-2022-11-30** reviews (before ChatGPT's public release, so confidently human) of 20–400 words, and reservoir-samples **10,000** reviews evenly across 8 product categories. Per-category checkpoints make it resumable. Output: `data/raw/human_reviews.csv`. |
| `01b_generate_gpt5mini.ipynb` | Samples 2,500 products from the human pool and prompts **GPT-5 mini** (OpenAI API) to write a customer-style review for each, with target lengths drawn from the human length distribution. Parallel, resume-safe. Appends to `data/generated/ai_reviews.csv`. |
| `01c_generate_open_models.ipynb` | Same idea for three more generators — **DeepSeek-V4-Flash**, **Gemma-4-31B-it**, **Qwen3.6-35B-A3B** — via the HuggingFace Inference Providers router (OpenAI-compatible, hosted, no GPU needed). Set `GENERATOR`, Run All, repeat. Appends to the same `data/generated/ai_reviews.csv`. |
| `02_build_splits.ipynb` | Builds the **4 leave-one-generator-out (LOGO) rotations**. Each rotation holds one generator out entirely as the *cross-generator* test set, trains on human + the other three, and keeps a **disjoint human pool** so cross-gen evaluation isn't contaminated. Writes `data/splits/rotation_{0..3}/{train,val_indist,test_indist,test_crossgen}.csv`. |

The final dataset is ~10,000 human + ~10,000 AI reviews (2,500 from each of 4
generators). The original proposal listed IBM Granite as the 4th generator;
it was swapped for DeepSeek-V4-Flash because no Granite model is served through
the HF router.

### Next — modeling and evaluation (not yet built)

To be written as new notebooks:

1. **Train all variants** — 4 rotations × 4 model variants
   (base / + contrastive loss / + adversarial head / both) = 16 runs on a
   DeBERTa-v3-base encoder.
2. **Baselines** — TF-IDF + logistic regression, and DeBERTa with no
   fine-tuning, for comparison.
3. **Evaluate** — aggregate into the in-distribution vs cross-generator
   comparison table (F1 and ROC-AUC), plus a per-held-out-generator breakdown.
   The gap between the in-dist and cross-gen columns is what the method is
   trying to close.
