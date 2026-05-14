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
and data sit together in one folder.

**Workflow:** edit and commit notebooks from your **laptop**, then `git push`.
In **Colab**, treat the repo as read-only — just re-run `00_setup` to `git pull`
the latest. Colab auto-commits every save ("Created using Colab"), so editing
notebooks there will fork history; don't.

## Pipeline status

### Done — data collection and splitting

| Notebook | What it does |
|---|---|
| `00_setup.ipynb` | Mounts Drive, clones/pulls the repo into Drive. Run once per session. |
| `01_collect_human_reviews.ipynb` | Streams the McAuley-Lab **Amazon Reviews 2023** dataset from HuggingFace, filters to **pre-2022-11-30** reviews (before ChatGPT's public release, so confidently human) of 20–400 words, **English only** (`langdetect`), and reservoir-samples ~10,000 reviews evenly across 8 product categories. Per-category checkpoints make it resumable. Output: `data/raw/human_reviews.csv`. |
| `01b_generate_gpt5mini.ipynb` | Samples 2,500 products from the human pool and prompts **GPT-5 mini** (OpenAI API) to write a customer-style review for each, with target lengths drawn from the human length distribution. Parallel, resume-safe. Appends to `data/generated/ai_reviews.csv`. |
| `01c_generate_open_models.ipynb` | Same idea for three more generators — **DeepSeek-V4-Flash**, **Gemma-4-31B-it**, **Qwen3.6-35B-A3B** — via the HuggingFace Inference Providers router (OpenAI-compatible, hosted, no GPU needed). Set `GENERATOR`, Run All, repeat. Appends to the same `data/generated/ai_reviews.csv`. |
| `02_build_splits.ipynb` | Builds the **4 leave-one-generator-out (LOGO) rotations**. Writes `data/splits/rotation_{0..3}/{train,val_indist,test_indist,test_crossgen}.csv` + `meta.csv`. |

### Dataset

~10,000 human + ~10,000 AI reviews (≈2,500 from each of 4 generators). Exact
counts drift slightly — non-English entries were dropped (≈0.3% human, ≈0.1% AI)
and some contaminated AI generations removed — so nothing downstream assumes a
round 2,500 or 10,000.

Notes:
- The original proposal listed **IBM Granite** as the 4th generator; it was
  swapped for **DeepSeek-V4-Flash** because no Granite model is served through
  the HF router.
- Qwen3.6 and DeepSeek-V4 are reasoning models — `01c` disables thinking
  (`chat_template_kwargs`) and strips any leftover reasoning traces so the
  reviews are clean.

### How the LOGO splits work

The project measures **cross-generator robustness**: can a detector trained on
some generators catch reviews from one it never saw? So the train→evaluate
pipeline runs 4 times, each rotation holding out one generator.

For a given rotation, the 3 *training* generators' reviews are split 80/10/10:

- **`train.csv`** — 80% of the 3 in-dist generators' AI + an equal count of
  human reviews (**1:1 balanced**).
- **`val_indist.csv`** — 10%, for tuning / early stopping (1:1).
- **`test_indist.csv`** — 10%, held-out AI from generators seen in training
  (1:1). The "easy" test.
- **`test_crossgen.csv`** — the **entire held-out generator** + a human pool.
  The "hard" test — a generator never seen in training.

The human pool is carved **after** the AI split: it takes exactly as many
humans as the AI in-dist count (for 1:1 balancing), and whatever is left over
becomes the cross-gen human pool. This means (a) the split builder adapts to
any human count instead of assuming 10,000, and (b) the cross-gen humans are
**disjoint** from training humans — so cross-gen evaluation measures
generalization to an unseen *generator*, not memorization of human text. The
notebook asserts no `review_id` overlap between `train` and `test_crossgen`.

### Next — modeling and evaluation (not yet built)

The detector is a **DeBERTa-v3-base encoder** with up to three heads stacked on
top. Each of the three robustness components from the proposal is a head + a
loss term; the four model variants just toggle which are active.

#### The model (define in `03_train.ipynb`)

- **Encoder** — `microsoft/deberta-v3-base`. Mean-pool (or take `[CLS]`) the
  token embeddings into a single review vector `h`.
- **Classification head** — `Linear(hidden, 2)` on `h`, trained with
  cross-entropy on the human/AI `label`. This *is* the detector; the other two
  heads only exist to shape the encoder.
- **Contrastive projection head** — `Linear(hidden, 128)` → L2-normalize,
  trained with **supervised contrastive loss** using the binary `label` as the
  class. Pulls all-AI together and all-human together in embedding space
  regardless of generator — "what AI writing looks like in general."
- **Generator-adversarial head** — `Linear(hidden, n_sources)` predicting
  *which source* wrote the review (the 3 training generators + `human` = 4
  classes), fed through a **Gradient Reversal Layer (GRL)**: identity on the
  forward pass, negates+scales the gradient on the backward pass. Minimizing
  this head's loss therefore *maximizes* generator confusion in the encoder,
  forcing it to drop generator-specific style cues.

Total loss: `L = L_ce + λ_c · L_supcon + λ_a · L_adv`. The GRL handles the
adversarial sign flip, so `L_adv` is just cross-entropy on the source label.

The four variants = which terms are active:

| variant | active loss |
|---|---|
| `base` | `L_ce` |
| `contrastive` | `L_ce + λ_c·L_supcon` |
| `adversarial` | `L_ce + λ_a·L_adv` |
| `both` | all three |

#### Training — `03_train.ipynb`

- Loop over **4 rotations × 4 variants = 16 runs**; skip any `(variant,
  rotation)` whose checkpoint already exists, so it survives Colab disconnects.
- Per run: read `data/splits/rotation_{r}/train.csv` + `val_indist.csv`,
  tokenize (`max_length≈384`), train with AdamW (encoder lr ≈ 2e-5), ~3 epochs,
  batch ≈ 16 on a T4. Select the best epoch on `val_indist` F1; save to
  `checkpoints/{variant}/rotation_{r}/best.pt` (gitignored, persists on Drive).
- Put hyperparameters in a config cell: `lr`, `epochs`, `batch_size`,
  `max_length`, `λ_c`, `λ_a`, `proj_dim`. Ramp the GRL `λ_a` from 0 upward over
  training (DANN-style) — adversarial training is unstable at full strength
  from step 0.
- ~30–45 min/run on a T4 → ~10 hrs total; split across sessions.

#### Baselines — `04_evaluate.ipynb`

- **TF-IDF + Logistic Regression** — fit TF-IDF on `train.csv` text, LogReg on
  top, per rotation.
- **DeBERTa, no fine-tuning** — freeze the encoder, train *only* a linear
  classifier on the pooled features (a linear probe). Shows what fine-tuning
  buys over raw DeBERTa features.

#### Evaluation — `04_evaluate.ipynb`

- For every checkpoint, score `test_indist` and `test_crossgen` with **F1
  (positive class = AI)** and **ROC-AUC**.
- Average each metric across the 4 rotations.
- Build the proposal's comparison table: rows = the 6 models (2 baselines + 4
  variants), columns = in-dist vs cross-gen F1/AUC.
- Add a **per-held-out-generator breakdown** (cross-gen F1 for each held-out
  generator) — shows which generators are hardest to catch unseen.
- Headline number = the **in-dist minus cross-gen gap**; the contrastive +
  adversarial components should shrink it.

#### Implementation notes

- DeBERTa-v3 needs `sentencepiece` installed for its tokenizer.
- Supervised contrastive loss needs ≥2 examples per class per batch — batches
  mix human+AI so that holds; just don't drop to a tiny batch size.
- Adversarial source labels come from the `generator` column (`human` + the 3
  training generators). Build the label map **per rotation**, since the
  held-out generator isn't present in training.
- Set seeds (torch / numpy / python) per run so the 16 runs are reproducible.
